#!/usr/bin/env python3
"""Compare matched Vespa Fourier-mode histories with the independent P1 oracle."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Mapping

import numpy as np

if __package__:
    from . import lmhd
else:
    import lmhd


def read_modes(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        for row in reader:
            parsed: dict[str, object] = {"field": row["field"]}
            for key, value in row.items():
                if key == "field":
                    continue
                parsed[key] = int(value) if key in ("cycle", "measured", "kmode") else float(value)
            rows.append(parsed)
    if not rows:
        raise ValueError(f"no mode records in {path}")
    return rows


def complex_value(row: Mapping[str, object]) -> complex:
    return complex(float(row["real"]), float(row["imag"]))


def select(rows: Iterable[Mapping[str, object]], field: str, redshift: float) -> Mapping[str, object]:
    candidates = [row for row in rows if row["field"] == field]
    if not candidates:
        raise KeyError(f"missing field {field}")
    return min(candidates, key=lambda row: abs(float(row["z"]) - redshift))


def physical_velocity(value: complex, manifest: Mapping[str, object], redshift: float) -> complex:
    a_ref = 1.0 / (1.0 + float(manifest["zstart"]))
    a = 1.0 / (1.0 + redshift)
    return value * float(manifest["velocity_unit_cm_s_at_start"]) * math.sqrt(a_ref / a)


def discrete_symbol(resolution: int, kmode: int) -> float:
    argument = 2.0 * math.pi * kmode / resolution
    return math.sin(argument) / argument


def oracle_for(manifest: Mapping[str, object], redshifts: list[float]) -> lmhd.EvolutionResult:
    cosmology = lmhd.Cosmology(h=float(manifest.get("hubble", 0.674)))
    mode = lmhd.ReferenceMode(
        k_mpc=float(manifest["k_comoving_mpc_inv"]),
        theta=float(manifest["theta_rad"]),
        b0_gauss=float(manifest["field_pG_comoving"]) * 1.0e-12,
        perturbation_fraction=float(manifest["epsilon"]),
        polarization=str(manifest["polarization"]),
    )
    if manifest["physics"] != "linear":
        raise ValueError("full-physics comparison needs the certified recombination oracle")
    return lmhd.evolve_photon_p1(
        mode,
        cosmology,
        float(manifest["zstart"]),
        min(redshifts),
        redshifts,
        lmhd.FullyIonized(),
        max_dln_a=1.0e-6,
        include_gravity=False,
        include_dark_matter=False,
        include_photon_stress=True,
        include_helium_electrons=False,
    )


FIELD_MAP = {
    "gas_delta": ("delta_b", False),
    "gas_v_x": ("v_b_z", True),
    "gas_v_y": ("v_b_y", True),
    "photon_delta": ("delta_gamma", False),
    "photon_v_x": ("v_gamma_z", True),
    "photon_v_y": ("v_gamma_y", True),
    "photon_baryon_slip_x": ("slip_z", True),
    "photon_baryon_slip_y": ("slip_y", True),
}


def relative_error(candidate: complex, reference: complex) -> float:
    return abs(candidate - reference) / max(abs(reference), np.finfo(float).tiny)


def compare_run(run_dir: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest.get("status") != "PASS":
        raise ValueError(f"run is not complete: {run_dir}")
    if manifest.get("coupling", "radiation") != "radiation":
        raise ValueError(f"run is not a photon-moment case: {run_dir}")
    rows = read_modes(run_dir / "modes.tsv")
    redshifts = sorted(
        {float(row["z"]) for row in rows if int(row["measured"]) > 0},
        reverse=True,
    )
    oracle = oracle_for(manifest, redshifts)
    oracle_index = {float(z): i for i, z in enumerate(oracle.redshift)}
    resolution = int(manifest["resolution"])
    kmode = int(manifest["kmode"])
    symbol = discrete_symbol(resolution, kmode)
    comparisons: list[dict[str, object]] = []
    for redshift in redshifts:
        index = oracle_index[min(oracle_index, key=lambda z: abs(z - redshift))]
        for vespa_field, (oracle_field, velocity) in FIELD_MAP.items():
            candidate = complex_value(select(rows, vespa_field, redshift))
            if velocity:
                candidate = physical_velocity(candidate, manifest, redshift)
            reference = complex(oracle.field(oracle_field)[index])
            discrete_reference = symbol * reference
            comparisons.append(
                {
                    "run": run_dir.name,
                    "backend": manifest["backend"],
                    "resolution": resolution,
                    "z": redshift,
                    "field": vespa_field,
                    "candidate_real": candidate.real,
                    "candidate_imag": candidate.imag,
                    "reference_real": reference.real,
                    "reference_imag": reference.imag,
                    "centered_reference_real": discrete_reference.real,
                    "centered_reference_imag": discrete_reference.imag,
                    "continuum_relative_error": relative_error(candidate, reference),
                    "centered_relative_error": relative_error(candidate, discrete_reference),
                }
            )

    final_rows = [row for row in comparisons if row["z"] == min(redshifts)]
    gate_fields = ("gas_v_x", "gas_v_y", "photon_v_x", "photon_v_y")
    max_centered = max(
        float(row["centered_relative_error"])
        for row in final_rows if row["field"] in gate_fields
    )
    gate = {
        "run": run_dir.name,
        "resolution": resolution,
        "centered_symbol": symbol,
        "max_centered_velocity_relative_error": max_centered,
        "threshold": 5.0e-3,
        "status": "PASS" if max_centered <= 5.0e-3 else "FAIL",
    }
    return comparisons, gate


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    gates: list[dict[str, object]] = []
    for run_dir in args.run_dirs:
        rows, gate = compare_run(run_dir.resolve())
        all_rows.extend(rows)
        gates.append(gate)
        print(
            f"{gate['status']} {gate['run']} N={gate['resolution']} "
            f"centered velocity error={gate['max_centered_velocity_relative_error']:.3e}"
        )
    write_tsv(args.output_dir / "comparison.tsv", all_rows)
    (args.output_dir / "gates.json").write_text(
        json.dumps({"gates": gates}, indent=2, sort_keys=True) + "\n"
    )
    return 0 if all(gate["status"] == "PASS" for gate in gates) else 1


if __name__ == "__main__":
    raise SystemExit(main())
