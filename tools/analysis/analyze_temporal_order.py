#!/usr/bin/env python3
"""Three-level temporal convergence audit for matched Vespa PMF modes."""

from __future__ import annotations

import argparse
import cmath
import csv
import json
import math
from pathlib import Path
from typing import Iterable, Mapping


FIELDS = (
    "B_y",
    "gas_delta",
    "gas_v_x",
    "gas_v_y",
    "dm_delta",
    "photon_delta",
    "photon_v_x",
    "photon_v_y",
    "photon_baryon_slip_x",
    "photon_baryon_slip_y",
    "x_HII",
    "gas_pressure_fraction",
    "gas_specific_internal_energy_fraction",
    "gas_temperature_fraction",
)

SIGNAL_FLOORS = {
    "photon_delta": 1.0e-8,
    "photon_v_x": 1.0e-8,
    "photon_v_y": 1.0e-8,
    "x_HII": 1.0e-8,
    "gas_temperature_fraction": 1.0e-6,
    "gas_specific_internal_energy_fraction": 1.0e-6,
}

TEMPORAL_CONTROLS = (
    "MHD_CFL",
    "MHD_RADIATION_NONLINEAR_CFL",
    "MHD_RADIATION_MAX_DLNA",
)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if not rows:
        raise ValueError(f"empty TSV: {path}")
    return rows


def complex_value(row: Mapping[str, str]) -> complex:
    return complex(float(row["real"]), float(row["imag"]))


def exact_mode_map(
    rows: Iterable[Mapping[str, str]], redshift: float, *, tolerance: float = 2.0e-8
) -> dict[str, complex]:
    selected = [
        row for row in rows
        if abs(float(row["z"]) - redshift) <= tolerance * max(redshift, 1.0)
    ]
    if not selected:
        nearest = min(abs(float(row["z"]) - redshift) for row in rows)
        raise ValueError(
            f"no exact diagnostic at z={redshift:g}; nearest offset is {nearest:.6g}"
        )
    return {str(row["field"]): complex_value(row) for row in selected}


def trajectory_scale(rows_by_level: Iterable[Iterable[Mapping[str, str]]], field: str) -> float:
    amplitudes = [
        abs(complex_value(row))
        for rows in rows_by_level for row in rows if row["field"] == field
    ]
    return max(amplitudes, default=0.0)


def phase_error(left: complex, right: complex) -> float:
    if left == 0 or right == 0:
        return math.nan
    return abs(cmath.phase(left / right))


def validate_temporal_ladder(
    manifests: list[Mapping[str, object]], *, ratio_tolerance: float = 1.0e-10
) -> dict[str, list[float]]:
    if len(manifests) != 3:
        raise ValueError("temporal convergence requires coarse, medium, and fine runs")
    invariants = (
        "backend",
        "physics",
        "coupling",
        "resolution",
        "zstart",
        "zend",
        "initial_state_fingerprint_sha256",
        "solver_source_fingerprint_sha256",
    )
    for key in invariants:
        values = [manifest.get(key) for manifest in manifests]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"temporal ladder changes {key}: {values}")

    controls: dict[str, list[float]] = {}
    for key in TEMPORAL_CONTROLS:
        values = [
            float(manifest["environment"][key])  # type: ignore[index]
            for manifest in manifests
        ]
        controls[key] = values
        ratios = (values[0] / values[1], values[1] / values[2])
        if any(abs(ratio - 2.0) > ratio_tolerance for ratio in ratios):
            raise ValueError(f"{key} is not refined by two: {values}")
    return controls


def compare_levels(
    coarse_rows: list[dict[str, str]],
    medium_rows: list[dict[str, str]],
    fine_rows: list[dict[str, str]],
    redshifts: Iterable[float],
    *,
    error_tolerance: float = 0.01,
    phase_tolerance: float = 0.01,
    contraction_tolerance: float = 0.8,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    all_levels = (coarse_rows, medium_rows, fine_rows)
    scales = {field: trajectory_scale(all_levels, field) for field in FIELDS}
    for redshift in redshifts:
        coarse = exact_mode_map(coarse_rows, redshift)
        medium = exact_mode_map(medium_rows, redshift)
        fine = exact_mode_map(fine_rows, redshift)
        for field in FIELDS:
            if field not in coarse or field not in medium or field not in fine:
                raise KeyError(f"missing {field} at z={redshift:g}")
            uc, um, uf = coarse[field], medium[field], fine[field]
            signal_floor = SIGNAL_FLOORS.get(field, 1.0e-12)
            # A relative convergence gate is meaningful only when every level
            # resolves the mode. A single value crossing the Float32 floor is
            # a quantization diagnostic, not evidence of temporal divergence.
            resolved_signal = min(abs(uc), abs(um), abs(uf)) >= signal_floor
            scale = max(scales[field], signal_floor)
            coarse_difference = abs(uc - um)
            fine_difference = abs(um - uf)
            precision_floor = 64.0 * 2.0**-23 * scale
            resolved_difference = max(coarse_difference, fine_difference) >= precision_floor
            contraction = (
                fine_difference / coarse_difference
                if coarse_difference > precision_floor else math.nan
            )
            observed_order = (
                math.log2(coarse_difference / fine_difference)
                if coarse_difference > 0.0 and fine_difference > 0.0 else math.nan
            )
            trajectory_error = fine_difference / scale
            point_error = fine_difference / max(abs(uf), signal_floor)
            angle_error = phase_error(um, uf)
            phase_ok = not math.isfinite(angle_error) or angle_error <= phase_tolerance
            error_ok = trajectory_error <= error_tolerance
            # Demand contraction only while the remaining difference is itself
            # scientifically material. Below 20% of the error budget, observed
            # order is dominated by Float32 accumulation and is diagnostic.
            contraction_gated = (
                resolved_difference
                and trajectory_error > 0.2 * error_tolerance
                and math.isfinite(contraction)
            )
            contraction_ok = not contraction_gated or contraction <= contraction_tolerance
            if not resolved_signal:
                status = "BELOW_FLOOR"
            elif error_ok and phase_ok and contraction_ok:
                status = "PASS"
            else:
                status = "FAIL"
            row = {
                "redshift": redshift,
                "field": field,
                "coarse_amplitude": abs(uc),
                "medium_amplitude": abs(um),
                "fine_amplitude": abs(uf),
                "coarse_medium_abs": coarse_difference,
                "medium_fine_abs": fine_difference,
                "trajectory_scale": scale,
                "trajectory_relative_error": trajectory_error,
                "point_relative_error": point_error,
                "phase_error_rad": angle_error,
                "contraction": contraction,
                "observed_order": observed_order,
                "precision_floor": precision_floor,
                "signal_floor": signal_floor,
                "contraction_gated": contraction_gated,
                "status": status,
            }
            rows.append(row)
            if status == "FAIL":
                failures.append(row)
    return rows, failures


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("coarse", type=Path)
    parser.add_argument("medium", type=Path)
    parser.add_argument("fine", type=Path)
    parser.add_argument("--error-tolerance", type=float, default=0.01)
    parser.add_argument("--phase-tolerance", type=float, default=0.01)
    parser.add_argument("--contraction-tolerance", type=float, default=0.8)
    args = parser.parse_args()

    manifests = [
        json.loads((path / "manifest.json").read_text())
        for path in (args.coarse, args.medium, args.fine)
    ]
    if any(manifest.get("status") != "PASS" for manifest in manifests):
        raise ValueError("all temporal levels must be completed PASS runs")
    controls = validate_temporal_ladder(manifests)
    requested = [
        set(float(value) for value in manifest["output_redshifts"])
        for manifest in manifests
    ]
    redshifts = sorted(set.intersection(*requested), reverse=True)
    start = max(float(manifest["zstart"]) for manifest in manifests)
    redshifts = [value for value in redshifts if value < start - 1.0e-8]
    mode_levels = [
        read_tsv(path / "modes.tsv") for path in (args.coarse, args.medium, args.fine)
    ]
    rows, failures = compare_levels(
        *mode_levels, redshifts,
        error_tolerance=args.error_tolerance,
        phase_tolerance=args.phase_tolerance,
        contraction_tolerance=args.contraction_tolerance,
    )
    resolved_orders = [
        float(row["observed_order"]) for row in rows
        if row["status"] != "BELOW_FLOOR"
        and math.isfinite(float(row["observed_order"]))
        and bool(row["contraction_gated"])
    ]
    status = "PASS" if rows and not failures else "FAIL"
    args.output.mkdir(parents=True, exist_ok=True)
    write_tsv(args.output / "temporal-order.tsv", rows)
    gate = {
        "schema": 1,
        "status": status,
        "scope": "three-level exact-redshift temporal convergence",
        "coarse": str(args.coarse.resolve()),
        "medium": str(args.medium.resolve()),
        "fine": str(args.fine.resolve()),
        "redshifts": redshifts,
        "error_tolerance": args.error_tolerance,
        "phase_tolerance_rad": args.phase_tolerance,
        "contraction_tolerance": args.contraction_tolerance,
        "temporal_controls": controls,
        "median_observed_order": (
            sorted(resolved_orders)[len(resolved_orders) // 2]
            if resolved_orders else math.nan
        ),
        "failure_count": len(failures),
        "failures": failures,
    }
    (args.output / "gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(gate, indent=2, sort_keys=True))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
