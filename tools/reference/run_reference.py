#!/usr/bin/env python3
"""Generate a deterministic, compact CPU reference example (not 3D validation)."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if __package__:
    from . import lmhd, lya_rt
else:
    import lmhd
    import lya_rt


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def save_figure(figure, stem: Path) -> None:
    for extension in ("png", "pdf"):
        figure.savefig(stem.with_suffix("." + extension), dpi=160, bbox_inches="tight")
    plt.close(figure)


def generate(output: Path, *, k_mpc: float = 1000.0, zstart: float = 3200.0,
             zend: float = 3100.0, points: int = 9, max_dlna: float = 0.001,
             with_lya: bool = True) -> dict:
    if not zstart > zend >= 0 or points < 2 or max_dlna <= 0:
        raise ValueError("require zstart > zend >= 0, points >= 2, max_dlna > 0")
    output.mkdir(parents=True, exist_ok=True)
    cosmology = lmhd.Cosmology()
    mode = lmhd.ReferenceMode(k_mpc, math.pi / 4, 30e-12)
    redshifts = np.linspace(zstart, zend, points)
    options = dict(ionization=lmhd.FullyIonized(), max_dln_a=max_dlna,
                   include_gravity=False, include_helium_electrons=False)
    sv = lmhd.evolve_sv_tcr_fsr(mode, cosmology, zstart, zend, redshifts, **options)
    p1 = lmhd.evolve_photon_p1(mode, cosmology, zstart, zend, redshifts,
                             include_dark_matter=False, **options)
    payload = {
        "schema": 1,
        "scope": "Independent linear reference models; not nonlinear 3D validation",
        "reference_revision": lmhd.REFERENCE_REVISION,
        "units": {"k": "comoving Mpc^-1 (no h)", "B0": "comoving gauss",
                  "velocity": "physical peculiar cm/s", "delta": "dimensionless"},
        "mode": asdict(mode), "cosmology": asdict(cosmology),
        "background": "fully ionized hydrogen; no helium electrons or gravity",
        "closure_note": "SV abrupt TCR/FSR and continuous photon P1 are distinct models",
        "models": {result.model: {"metadata": result.metadata,
                                  "samples": lmhd.result_rows(result)}
                   for result in (sv, p1)},
    }
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.2), layout="constrained")
    for result, label in ((sv, "Schiff TCR/FSR"), (p1, "Photon P1")):
        for axis, field in zip(axes, ("b_mode_over_initial", "delta_b", "v_b_z")):
            axis.plot(result.redshift, np.abs(result.field(field)), label=label)
    for axis, ylabel in zip(axes, ("|b / b(initial)|", "|delta_b|", "|v_b parallel| [cm/s]")):
        axis.set(xlabel="Redshift z", ylabel=ylabel)
        axis.invert_xaxis()
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    save_figure(figure, output / "lmhd_p1")

    if with_lya:
        ionization = lmhd.PeeblesIonization(cosmology, zmax=3500, zmin=500, samples=1200)
        response = lya_rt.first_order_response(
            cosmology, ionization, 1100.0, 1e5,
            xmax=20, bins_per_width=1, lmax=4, method="sparse")
        rates = lya_rt.recombination_response(response, cosmology)
        payload["lya"] = {
            "scope": "small-domain scalar hierarchy smoke example; refine before science use",
            "k_comoving_mpc_inv": 1e5, "redshift": 1100.0,
            "xmax_doppler_widths": 20, "bins_per_width": 1, "lmax": 4,
            "line_averages": {key: float(response[key]) for key in ("Abar", "Bbar", "Cbar")},
            "rates": asdict(rates),
            "residual_max": max(b.residual_max for b in response["bases"].values()),
        }
        figure, axes = plt.subplots(1, 2, figsize=(8, 3.2), layout="constrained")
        line = response["homogeneous"]
        axes[0].plot(line.x, line.occupation / line.background.f_equilibrium)
        axes[0].set(xlabel="Doppler frequency x", ylabel="f / f_eq", title="Homogeneous line")
        for name, basis in response["bases"].items():
            scale = max(np.max(np.abs(basis.multipoles[0])), np.finfo(float).tiny)
            axes[1].plot(basis.x, basis.multipoles[0] / scale, label=name)
        axes[1].set(xlabel="Doppler frequency x", ylabel="Monopole / max |monopole|")
        axes[1].legend()
        save_figure(figure, output / "lya_hierarchy")
    write_json(output / "reference.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--k-mpc", type=float, default=1000)
    parser.add_argument("--zstart", type=float, default=3200)
    parser.add_argument("--zend", type=float, default=3100)
    parser.add_argument("--points", type=int, default=9)
    parser.add_argument("--max-dlna", type=float, default=0.001)
    parser.add_argument("--no-lya", action="store_true")
    args = parser.parse_args()
    generate(args.output, k_mpc=args.k_mpc, zstart=args.zstart, zend=args.zend,
             points=args.points, max_dlna=args.max_dlna, with_lya=not args.no_lya)
    print(f"Reference JSON and PNG/PDF figures: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
