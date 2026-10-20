#!/usr/bin/env python3
"""Build an interpolation-certified nonlocal Lyman-alpha response table."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

if __package__:
    from . import lmhd, lya_rt, sv25_paper
else:
    import lmhd
    import lya_rt
    import sv25_paper


_COSMOLOGY: lmhd.Cosmology | None = None
_IONIZATION: lmhd.PeeblesIonization | None = None


class _TabulatedIonization:
    def __init__(self, path: str) -> None:
        rows = np.genfromtxt(path, delimiter=",", comments="#")
        if rows.ndim != 2 or rows.shape[1] < 2 or rows.shape[0] < 3:
            raise ValueError(f"invalid ionization-history table: {path}")
        order = np.argsort(rows[:, 0])
        self.redshift = np.asarray(rows[order, 0], dtype=float)
        self.ionization = np.asarray(rows[order, 1], dtype=float)
        if not (
            np.all(np.isfinite(self.redshift))
            and np.all(np.isfinite(self.ionization))
            and np.all(np.diff(self.redshift) > 0.0)
        ):
            raise ValueError(f"non-finite or non-monotone ionization history: {path}")

    def __call__(self, redshift: float) -> float:
        # The production fixture is total x_e.  The configured z<=1700 window
        # is below He recombination, where x_e equals x_HII to the accuracy
        # required by this first-order response table.
        value = np.interp(redshift, self.redshift, self.ionization)
        return float(np.clip(value, 1.0e-14, 1.0 - 1.0e-14))


def _initialize_worker(
    ionization_model: str = "peebles",
    ionization_history: str = "",
    cosmology_model: str = "campaign",
) -> None:
    global _COSMOLOGY, _IONIZATION
    _COSMOLOGY = (
        sv25_paper.paper_cosmology()
        if cosmology_model == "paper"
        else lmhd.Cosmology()
    )
    if ionization_model == "external":
        _IONIZATION = _TabulatedIonization(ionization_history)
    elif ionization_model in ("peebles", "recfast-v2"):
        _IONIZATION = lmhd.PeeblesIonization(
            _COSMOLOGY,
            zmax=3600.0,
            zmin=400.0,
            samples=10000,
            recfast_v2=ionization_model == "recfast-v2",
        )
    else:
        raise ValueError(f"unknown ionization model: {ionization_model}")


def _row(task: tuple[float, np.ndarray, float, float, int]) -> dict[str, object]:
    redshift, positive_k, xmax, bins_per_width, lmax = task
    if _COSMOLOGY is None or _IONIZATION is None:
        _initialize_worker()
    assert _COSMOLOGY is not None and _IONIZATION is not None
    background = lya_rt.lya_background(_COSMOLOGY, redshift, _IONIZATION)
    homogeneous = lya_rt.solve_homogeneous_line(
        background,
        xmax=xmax,
        bins_per_width=bins_per_width,
        method="sparse",
    )
    rates = []
    diagnostics = []
    for k_mpc in np.concatenate(([0.0], positive_k)):
        bases = {
            name: lya_rt.solve_scalar_basis(
                homogeneous,
                float(k_mpc),
                name,
                lmax=lmax,
                bins_per_width=bins_per_width,
                method="sparse",
            )
            for name in ("A", "B", "C")
        }
        response = lya_rt.recombination_response(
            {
                "background": background,
                "Abar": bases["A"].line_average,
                "Bbar": bases["B"].line_average,
                "Cbar": bases["C"].line_average,
            },
            _COSMOLOGY,
        )
        rates.append(
            (
                response.density_rate,
                response.ionization_rate,
                response.velocity_divergence_rate,
                response.temperature_rate,
            )
        )
        diagnostics.append(
            (
                response.peebles_factor,
                response.A_effective,
                response.B_effective,
                response.escape_response,
            )
        )
    values = np.asarray(rates)
    delta = values[1:] - values[0]
    return {
        "redshift": redshift,
        "xe": background.xe,
        "local_rates": values[0],
        "delta_rates": delta,
        "diagnostics": np.asarray(diagnostics),
        "homogeneous_residual": homogeneous.residual_max,
    }


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _bilinear_uniform(
    values: np.ndarray,
    redshifts: np.ndarray,
    positive_k: np.ndarray,
    redshift: float,
    k_mpc: float,
) -> np.ndarray:
    uz = math.log1p(redshift)
    u0 = math.log1p(float(redshifts[0]))
    du = math.log1p(float(redshifts[1])) - u0
    z_coordinate = np.clip((uz - u0) / du, 0.0, redshifts.size - 1.0)
    zi = min(int(math.floor(z_coordinate)), redshifts.size - 2)
    zf = z_coordinate - zi
    logk = math.log10(k_mpc)
    logk0 = math.log10(float(positive_k[0]))
    dlogk = math.log10(float(positive_k[1])) - logk0
    k_coordinate = np.clip(
        (logk - logk0) / dlogk, 0.0, positive_k.size - 1.0
    )
    ki = min(int(math.floor(k_coordinate)), positive_k.size - 2)
    kf = k_coordinate - ki
    result = np.empty(values.shape[2])
    signs = (-1.0, -1.0, 1.0, 1.0)
    for component, sign in enumerate(signs[: values.shape[2]]):
        signed = sign * values[:, :, component]
        corners = signed[zi : zi + 2, ki : ki + 2]
        if np.any(corners <= 0.0):
            raise ValueError("response table violates its analytic component sign")
        logarithm = np.log(signed)
        lower = (
            (1.0 - kf) * logarithm[zi, ki]
            + kf * logarithm[zi, ki + 1]
        )
        upper = (
            (1.0 - kf) * logarithm[zi + 1, ki]
            + kf * logarithm[zi + 1, ki + 1]
        )
        result[component] = sign * math.exp((1.0 - zf) * lower + zf * upper)
    return result


def _write_tsv(
    path: Path,
    redshifts: np.ndarray,
    positive_k: np.ndarray,
    local_rates: np.ndarray,
    delta_rates: np.ndarray,
) -> None:
    fieldnames = [
        "redshift",
        "k_mpc",
        "local_density_rate_s",
        "local_ionization_rate_s",
        "local_velocity_divergence_rate_s",
        "local_temperature_rate_s",
        "delta_density_rate_s",
        "delta_ionization_rate_s",
        "delta_velocity_divergence_rate_s",
        "delta_temperature_rate_s",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fieldnames, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for iz, redshift in enumerate(redshifts):
            for ik, k_mpc in enumerate(positive_k):
                writer.writerow(
                    {
                        "redshift": f"{redshift:.17g}",
                        "k_mpc": f"{k_mpc:.17g}",
                        "local_density_rate_s": f"{local_rates[iz, 0]:.17g}",
                        "local_ionization_rate_s": f"{local_rates[iz, 1]:.17g}",
                        "local_velocity_divergence_rate_s": f"{local_rates[iz, 2]:.17g}",
                        "local_temperature_rate_s": f"{local_rates[iz, 3]:.17g}",
                        "delta_density_rate_s": f"{delta_rates[iz, ik, 0]:.17g}",
                        "delta_ionization_rate_s": f"{delta_rates[iz, ik, 1]:.17g}",
                        "delta_velocity_divergence_rate_s": f"{delta_rates[iz, ik, 2]:.17g}",
                        "delta_temperature_rate_s": f"{delta_rates[iz, ik, 3]:.17g}",
                    }
                )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--zmin", type=float, default=500.0)
    parser.add_argument("--zmax", type=float, default=3500.0)
    parser.add_argument("--nz", type=int, default=121)
    parser.add_argument("--kmin", type=float, default=100.0)
    parser.add_argument("--kmax", type=float, default=1.0e6)
    parser.add_argument("--nk", type=int, default=65)
    parser.add_argument("--xmax", type=float, default=1000.0)
    parser.add_argument("--bins-per-width", type=float, default=2.0)
    parser.add_argument("--lmax", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--cosmology", choices=("campaign", "paper"), default="campaign"
    )
    parser.add_argument(
        "--ionization-history",
        type=Path,
        help="CSV with redshift and x_e columns for --ionization-model=external",
    )
    parser.add_argument(
        "--ionization-model",
        choices=("peebles", "recfast-v2", "external"),
        default="peebles",
    )
    args = parser.parse_args()
    if not (0 < args.zmin < args.zmax and args.nz >= 3):
        parser.error("redshift grid must have 0 < zmin < zmax and nz >= 3")
    if not (0 < args.kmin < args.kmax and args.nk >= 3):
        parser.error("wavenumber grid must have 0 < kmin < kmax and nk >= 3")
    if args.ionization_model == "external" and args.ionization_history is None:
        parser.error("--ionization-model=external requires --ionization-history")
    if args.ionization_model != "external" and args.ionization_history is not None:
        parser.error("--ionization-history is valid only with --ionization-model=external")
    if args.ionization_history is not None and not args.ionization_history.is_file():
        parser.error(f"ionization history does not exist: {args.ionization_history}")
    history_path = (
        str(args.ionization_history.resolve())
        if args.ionization_history is not None
        else ""
    )
    args.output.mkdir(parents=True, exist_ok=True)
    redshifts = np.expm1(
        np.linspace(math.log1p(args.zmin), math.log1p(args.zmax), args.nz)
    )
    positive_k = np.logspace(
        math.log10(args.kmin), math.log10(args.kmax), args.nk
    )
    tasks = [
        (float(redshift), positive_k, args.xmax, args.bins_per_width, args.lmax)
        for redshift in redshifts
    ]
    start = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_initialize_worker,
        initargs=(args.ionization_model, history_path, args.cosmology),
    ) as executor:
        rows = list(executor.map(_row, tasks))
    wall_seconds = time.perf_counter() - start
    local_rates = np.asarray([row["local_rates"] for row in rows])
    delta_rates = np.asarray([row["delta_rates"] for row in rows])
    diagnostics = np.asarray([row["diagnostics"] for row in rows])
    xe = np.asarray([row["xe"] for row in rows])
    table_path = args.output / "response_table.tsv"
    _write_tsv(table_path, redshifts, positive_k, local_rates, delta_rates)
    np.savez_compressed(
        args.output / "response_table.npz",
        redshift=redshifts,
        k_mpc=positive_k,
        xe=xe,
        local_rates=local_rates,
        delta_rates=delta_rates,
        diagnostics=diagnostics,
    )

    # Direct midpoint checks exercise both table dimensions away from anchors.
    validation_tasks = []
    validation_locations = []
    index_pairs = []
    for fraction in (0.1, 0.3, 0.5, 0.7, 0.9):
        pair = (
            min(int(round(fraction * (args.nz - 2))), args.nz - 2),
            min(int(round(fraction * (args.nk - 2))), args.nk - 2),
        )
        if pair not in index_pairs:
            index_pairs.append(pair)
    for zi, ki in index_pairs:
        zmid = math.sqrt((1.0 + redshifts[zi]) * (1.0 + redshifts[zi + 1])) - 1.0
        kmid = math.sqrt(positive_k[ki] * positive_k[ki + 1])
        validation_locations.append((zmid, kmid))
        validation_tasks.append(
            (zmid, np.asarray([kmid]), args.xmax, args.bins_per_width, args.lmax)
        )
    with ProcessPoolExecutor(
        max_workers=min(args.workers, len(validation_tasks)),
        initializer=_initialize_worker,
        initargs=(args.ionization_model, history_path, args.cosmology),
    ) as executor:
        validation_rows = list(executor.map(_row, validation_tasks))
    scales = np.maximum(np.max(np.abs(delta_rates), axis=(0, 1)), 1.0e-300)
    expected_signs = np.asarray([-1.0, -1.0, 1.0, 1.0])
    signed_rates = delta_rates * expected_signs[None, None, :]
    sign_locations = np.argwhere(signed_rates <= 0.0)
    sign_violations = [
        {
            "redshift": float(redshifts[iz]),
            "k_mpc": float(positive_k[ik]),
            "component": (
                "density", "ionization", "velocity_divergence", "temperature"
            )[component],
            "value": float(delta_rates[iz, ik, component]),
        }
        for iz, ik, component in sign_locations
    ]
    validation = []
    for location, direct_row in zip(validation_locations, validation_rows):
        direct = np.asarray(direct_row["delta_rates"])[0]
        try:
            interpolated = _bilinear_uniform(
                delta_rates, redshifts, positive_k, location[0], location[1]
            )
        except ValueError:
            interpolated = np.full(4, np.nan)
        normalized_error = np.abs(interpolated - direct) / np.maximum(
            np.abs(direct), 1.0e-5 * scales
        )
        validation.append(
            {
                "redshift": location[0],
                "k_mpc": location[1],
                "direct": direct.tolist(),
                "interpolated": interpolated.tolist(),
                "relative_error": normalized_error.tolist(),
                "max_relative_error": float(np.max(normalized_error)),
            }
        )
    max_interpolation_error = max(row["max_relative_error"] for row in validation)
    sign_ok = not sign_violations
    interpolation_ok = math.isfinite(max_interpolation_error) and max_interpolation_error < 0.01
    digest = hashlib.sha256(table_path.read_bytes()).hexdigest()
    metadata = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "PASS" if sign_ok and interpolation_ok else "FAIL",
        "paper": "Schiff & Venumadhav, arXiv:2506.16517v2",
        "response": (
            "Lyman-alpha first-order free-electron rate including the SV25 "
            "delta-alpha_B baryon-temperature term; k=0 subtracted"
        ),
        "continuum_channel": "not included (paper classifies it as subdominant)",
        "solver": "second-order sparse trapezoidal BVP",
        "interpolation": "bilinear signed-log magnitude in ln(1+z), ln(k)",
        "sign_convention": "delta_dot_xe=-delta_dot_x1s; fixed by short-wavelength limit",
        "analytic_sign_gate": "PASS" if sign_ok else "FAIL",
        "sign_violations": sign_violations,
        "ionization_history": (
            {
                "model": "external tabulation",
                "path": history_path,
                "sha256": hashlib.sha256(args.ionization_history.read_bytes()).hexdigest(),
                "validity": "x_e treated as x_HII; use only below helium recombination",
            }
            if args.ionization_history is not None
            else {
                "model": (
                    "independent current-cosmology RECFAST-v2 ODE"
                    if args.ionization_model == "recfast-v2"
                    else "independent pure-Peebles three-level atom"
                ),
                "cosmology": args.cosmology,
            }
        ),
        "redshift_grid": redshifts.tolist(),
        "k_mpc_grid": positive_k.tolist(),
        "xmax_doppler_widths": args.xmax,
        "bins_per_doppler_width": args.bins_per_width,
        "lmax": args.lmax,
        "workers": args.workers,
        "cosmology": args.cosmology,
        "wall_seconds": wall_seconds,
        "max_boundary_residual": max(float(row["homogeneous_residual"]) for row in rows),
        "max_midpoint_interpolation_error": max_interpolation_error,
        "midpoint_validation": validation,
        "table_sha256": digest,
    }
    _atomic_json(args.output / "metadata.json", metadata)
    print(
        f"response table {metadata['status']}: {args.nz}x{args.nk}, "
        f"wall={wall_seconds:.1f}s interpolation={max_interpolation_error:.3e}",
        flush=True,
    )
    return 0 if metadata["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
