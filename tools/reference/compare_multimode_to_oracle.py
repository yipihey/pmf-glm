#!/usr/bin/env python3
"""Compare a weak multimode Vespa spectrum with independent P1 transfers."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

if __package__:
    from . import lmhd, lya_rt
else:
    import lmhd
    import lya_rt


FIELDS = (
    "B_vector", "gas_delta", "dm_delta", "photon_delta",
    "photon_velocity", "photon_baryon_slip",
)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def complex_value(payload: dict[str, float]) -> complex:
    return complex(float(payload["real"]), float(payload["imag"]))


def p1_state(mode: dict[str, object]) -> np.ndarray:
    fields = mode["mode_state"]
    names = (
        "b_x_over_B0", "b_y_over_B0", "delta_b",
        "v_b_x", "v_b_y", "v_b_z", "delta_gamma",
        "v_gamma_x", "v_gamma_y", "v_gamma_z",
        "delta_dm", "v_dm_z", "delta_xe", "delta_temperature",
    )
    return np.asarray([complex_value(fields[name]) for name in names])


def expected_power(
    result: lmhd.EvolutionResult, field: str, index: int,
    velocity_unit: float,
) -> float:
    if field == "B_vector":
        return float(
            abs(result.field("b_x_over_B0")[index]) ** 2
            + abs(result.field("b_y_over_B0")[index]) ** 2
        )
    if field == "gas_delta":
        return float(abs(result.field("delta_b")[index]) ** 2)
    if field == "dm_delta":
        return float(abs(result.field("delta_dm")[index]) ** 2)
    if field == "photon_delta":
        return float(abs(result.field("delta_gamma")[index]) ** 2)
    if field == "photon_velocity":
        return float(sum(
            abs(result.field(name)[index] / velocity_unit) ** 2
            for name in ("v_gamma_x", "v_gamma_y", "v_gamma_z")
        ))
    if field == "photon_baryon_slip":
        return float(sum(
            abs(result.field(name)[index] / velocity_unit) ** 2
            for name in ("slip_x", "slip_y", "slip_z")
        ))
    raise KeyError(field)


def spectrum_rows(rows, field: str, kbin: int) -> list[dict[str, str]]:
    return sorted(
        (
            row for row in rows
            if row.get("field") == field
            and int(row.get("kbin", "-1")) == kbin
            and int(row.get("mubin", "1")) == 1
            and int(row.get("Nmodes", "0")) > 0
            and math.isfinite(float(row.get("P", "nan")))
        ),
        key=lambda row: float(row["z"]), reverse=True,
    )


def nearest_kbin(rows, field: str, zstart: float, target_k: float) -> int:
    selected = [
        row for row in rows
        if row.get("field") == field
        and abs(float(row["z"]) - zstart) <= 2.0e-8 * max(zstart, 1.0)
        and int(row.get("Nmodes", "0")) > 0
        and math.isfinite(float(row.get("k", "nan")))
    ]
    if not selected:
        raise ValueError(f"missing initial {field} spectrum")
    return int(min(selected, key=lambda row: abs(float(row["k"]) - target_k))["kbin"])


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison(path: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    colors = plt.get_cmap("tab10")
    with PdfPages(path) as pdf:
        figure, axes = plt.subplots(3, 2, figsize=(9.0, 10.0), sharex=True)
        for axis, field in zip(axes.flat, FIELDS):
            for color_index, mode in enumerate(sorted({int(row["kmode"]) for row in rows})):
                selected = sorted(
                    (row for row in rows if row["field"] == field and int(row["kmode"]) == mode),
                    key=lambda row: float(row["z"]), reverse=True,
                )
                if not selected:
                    continue
                redshift = [float(row["z"]) for row in selected]
                color = colors(color_index % 10)
                axis.loglog(
                    redshift,
                    [max(float(row["measured_transfer_power"]), 1.0e-30) for row in selected],
                    color=color, linewidth=1.2, label=f"m={mode} Vespa",
                )
                axis.loglog(
                    redshift,
                    [max(float(row["oracle_transfer_power"]), 1.0e-30) for row in selected],
                    color=color, linewidth=1.0, linestyle="--", label=f"m={mode} P1",
                )
            axis.invert_xaxis()
            axis.set_title(field)
            axis.grid(alpha=0.2)
        axes.flat[0].legend(fontsize=7, ncol=2)
        for axis in axes[-1, :]:
            axis.set_xlabel("redshift")
        figure.suptitle("Weak broad-band transfer: Vespa spectra vs independent P1")
        figure.tight_layout()
        pdf.savefig(figure)
        figure.savefig(path.with_suffix(".png"), dpi=180)
        plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("handoff_metadata", type=Path)
    parser.add_argument("response_table", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tolerance", type=float, default=0.15)
    parser.add_argument("--max-dlna", type=float, default=1.0e-4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((args.run / "manifest.json").read_text())
    metadata = json.loads(args.handoff_metadata.read_text())
    spectra = read_tsv(args.run / "spectra.tsv")
    summary = read_tsv(args.run / "summary.tsv")[-1]
    redshifts = sorted({float(row["z"]) for row in spectra}, reverse=True)
    zstart = float(manifest["zstart"])

    cosmology = lmhd.Cosmology(
        h=float(summary["hubble_h"]),
        omega_m=float(summary["omega_m"]),
        omega_b=float(summary.get("omega_b", 0.05366655)),
        omega_lambda=float(summary["omega_l"]),
        omega_r=float(summary["omega_r"]),
        hydrogen_mass_fraction=0.7546,
    )
    ionization = lmhd.PeeblesIonization(
        cosmology, zmax=3600.0, zmin=min(400.0, float(manifest["zend"])),
        samples=12000, recfast_v2=True,
    )
    temperature = lmhd.ComptonTemperature(
        cosmology, ionization, zmax=3600.0,
        zmin=min(400.0, float(manifest["zend"])), samples=12000,
    )
    table = lya_rt.ResponseTable.load(args.response_table)
    nonlocal_scale = float(manifest.get("nonlocal_scale", 1.0))

    def rates(redshift: float, k_mpc: float) -> np.ndarray:
        local = lmhd.recfast_v2_local_response(
            cosmology, redshift, ionization(redshift), temperature(redshift)
        )
        if not table.redshift[0] <= redshift <= table.redshift[-1]:
            return local
        differential = table.rates(redshift, k_mpc, differential_only=True)
        differential[2] = table.rates(redshift, k_mpc)[2]
        return local + nonlocal_scale * differential

    comparisons: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    initial_velocity_unit = float(manifest["velocity_unit_cm_s_at_start"])
    astart = 1.0 / (1.0 + zstart)
    for mode_metadata in metadata["modes"]:
        kmode = int(mode_metadata["kmode"])
        k_mpc = float(mode_metadata["k_comoving_mpc_inv"])
        mode = lmhd.ReferenceMode(
            k_mpc=k_mpc,
            theta=float(metadata["theta_rad"]),
            b0_gauss=float(metadata["field_pG_comoving"]) * 1.0e-12,
            perturbation_fraction=float(mode_metadata["epsilon_amplitude"]),
            polarization="magnetosonic",
        )
        oracle = lmhd.evolve_photon_p1(
            mode, cosmology, zstart, float(manifest["zend"]), redshifts,
            ionization, max_dln_a=args.max_dlna, include_gravity=True,
            include_dark_matter=True, include_photon_stress=True,
            include_helium_electrons=True,
            recombination_rates=lambda z, k=k_mpc: rates(z, k),
            baryon_temperature=temperature, initial_state=p1_state(mode_metadata),
        )
        oracle_index = {float(z): index for index, z in enumerate(oracle.redshift)}
        target_k = 2.0 * math.pi * kmode / float(metadata["box_ckpc_h"])
        for field in FIELDS:
            kbin = nearest_kbin(spectra, field, zstart, target_k)
            measured = spectrum_rows(spectra, field, kbin)
            initial_row = min(measured, key=lambda row: abs(float(row["z"]) - zstart))
            initial_power = float(initial_row["P"])
            initial_expected = expected_power(
                oracle, field, oracle_index[float(initial_row["z"])],
                initial_velocity_unit,
            )
            if initial_power <= 0.0 or initial_expected <= 0.0:
                continue
            for row in measured:
                z = float(row["z"])
                index = oracle_index[z]
                a = 1.0 / (1.0 + z)
                velocity_unit = initial_velocity_unit * math.sqrt(astart / a)
                expected_ratio = expected_power(
                    oracle, field, index, velocity_unit
                ) / initial_expected
                measured_ratio = float(row["P"]) / initial_power
                scale = max(abs(expected_ratio), abs(measured_ratio), 1.0e-30)
                error = abs(measured_ratio - expected_ratio) / scale
                status = "PASS" if error <= args.tolerance else "FAIL"
                record = {
                    "kmode": kmode,
                    "k_comoving_mpc_inv": k_mpc,
                    "kbin": kbin,
                    "field": field,
                    "z": z,
                    "measured_transfer_power": measured_ratio,
                    "oracle_transfer_power": expected_ratio,
                    "symmetric_relative_error": error,
                    "tolerance": args.tolerance,
                    "status": status,
                }
                comparisons.append(record)
                if status == "FAIL":
                    failures.append(record)

    write_tsv(args.output / "comparison.tsv", comparisons)
    if comparisons:
        plot_comparison(args.output / "comparison.pdf", comparisons)
    payload = {
        "schema": 1,
        "status": "PASS" if comparisons and not failures else "FAIL",
        "scope": (
            "weak broad-band superposition transfer; one guide-field angle, "
            "power-only device spectra"
        ),
        "solver_source_fingerprint_sha256": manifest.get(
            "solver_source_fingerprint_sha256"
        ),
        "reference_source_fingerprint_sha256": metadata.get(
            "reference_source_fingerprint_sha256"
        ),
        "spectral_epsilon": metadata["spectral_epsilon"],
        "modes": [int(mode["kmode"]) for mode in metadata["modes"]],
        "tolerance": args.tolerance,
        "checks": len(comparisons),
        "failures": failures,
    }
    (args.output / "gate.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
