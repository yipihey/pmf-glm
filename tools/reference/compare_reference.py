#!/usr/bin/env python3
"""Compare exact-epoch complex mode tables in matching physical units."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__:
    from .run_reference import write_json, save_figure
else:
    from run_reference import write_json, save_figure


def read_modes(path: Path) -> dict[tuple[str, float], complex]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if not {"field", "z", "real", "imag"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: require field, z, real, imag columns")
        result = {}
        for row in reader:
            key = (row["field"], float(row["z"]))
            value = complex(float(row["real"]), float(row["imag"]))
            if key in result:
                raise ValueError(f"duplicate field/epoch in {path}: {key}")
            if not all(math.isfinite(x) for x in (key[1], value.real, value.imag)):
                raise ValueError(f"nonfinite mode in {path}: {key}")
            result[key] = value
    if not result:
        raise ValueError(f"no modes in {path}")
    return result


def compare(reference: dict, candidate: dict, *, rtol: float = 0.005,
            atol: float = 0.0, floor_fraction: float = 1e-8) -> tuple[list, dict]:
    if not all(math.isfinite(x) for x in (rtol, atol, floor_fraction)) or min(rtol, atol) < 0 or floor_fraction <= 0:
        raise ValueError("tolerances must be finite and nonnegative; floor must be positive")
    if not reference or set(reference) != set(candidate):
        raise ValueError("field/epoch sets must match exactly; interpolation and nearest-epoch selection are not used")
    scales = {}
    for (field, _), value in reference.items():
        scales[field] = max(scales.get(field, 0), abs(value))
    rows = []
    for (field, z), expected in sorted(reference.items()):
        actual = candidate[field, z]
        floor = max(scales[field] * floor_fraction, 1e-300)
        denominator = max(abs(expected), floor)
        error = abs(actual - expected)
        phase = (math.atan2((actual / expected).imag, (actual / expected).real)
                 if abs(expected) > floor and abs(actual) > floor else None)
        rows.append(dict(field=field, z=z, absolute_complex_error=error,
                         relative_complex_error=error / denominator,
                         phase_error_rad=phase, passed=error <= atol + rtol * denominator))
    summary = {
        "scope": "Complex mode-table comparison only; not native nonlinear hierarchy validation",
        "units_contract": "Both inputs must use the same field definitions; velocities are physical peculiar cm/s",
        "status": "PASS" if all(row["passed"] for row in rows) else "FAIL",
        "rtol": rtol, "atol": atol, "zero_signal_floor_fraction": floor_fraction,
        "records": len(rows), "failed_records": sum(not row["passed"] for row in rows),
        "max_relative_complex_error": max(row["relative_complex_error"] for row in rows),
    }
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rtol", type=float, default=0.005)
    parser.add_argument("--atol", type=float, default=0)
    args = parser.parse_args()
    rows, summary = compare(read_modes(args.reference), read_modes(args.candidate), rtol=args.rtol, atol=args.atol)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "comparison.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    write_json(args.output / "comparison.json", summary)
    figure, axis = plt.subplots(figsize=(8, 4), layout="constrained")
    for field in sorted({row["field"] for row in rows}):
        selected = sorted((row for row in rows if row["field"] == field), key=lambda row: row["z"], reverse=True)
        axis.plot([row["z"] for row in selected], [row["relative_complex_error"] for row in selected], label=field)
    axis.set(xlabel="Redshift z", ylabel="Relative complex error (signal floor)")
    axis.invert_xaxis()
    axis.legend(fontsize=6, ncol=2)
    save_figure(figure, args.output / "comparison")
    print(f"{summary['status']}: {summary['failed_records']}/{len(rows)} failed mode records")
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
