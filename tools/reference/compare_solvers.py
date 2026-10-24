"""Compare two completed ``pmf-compare-nonlinear`` directories (GLM versus CT).

Writes ``solver_differences.tsv`` (relative amplitude and phase difference of every
gated complex observable at every output redshift, CT relative to GLM) and
``solver_summary.json`` (per-family maxima plus each run's own gate result).
Nothing is fitted or loosened; the tolerances are those recorded in the GLM gate.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

FAMILY = {"B_y": "magnetic", "gas_delta": "baryon_density", "dm_delta": "dark_matter",
          "dm_v_x": "dark_matter", "x_HII": "ionization"}


def family(field):
    return FAMILY.get(field, "velocity")


def load(directory):
    directory = Path(directory)
    with (directory/"comparison.tsv").open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    return rows, json.loads((directory/"gate.json").read_text())


def differences(glm_rows, ct_rows, z_tolerance=1e-3):
    """Pair rows by field and output index (the first post-handoff epoch differs slightly
    in z between solvers because the first step does); epochs must agree within z_tolerance."""
    def by_field(rows):
        out = {}
        for r in rows:
            out.setdefault(r["field"], []).append(r)
        return out
    glm, ct = by_field(glm_rows), by_field(ct_rows)
    if glm.keys() != ct.keys() or any(len(glm[f]) != len(ct[f]) for f in glm):
        raise ValueError("GLM and CT comparison tables have different fields or epochs")
    out = []
    for field, rows in glm.items():
        for r, c in zip(rows, ct[field]):
            za, zb = float(r["z"]), float(c["z"])
            if abs(za-zb) > z_tolerance:
                raise ValueError(f"CT run epoch z={zb} does not match GLM z={za} for {field}")
            a = complex(float(r["real"]), float(r["imag"]))
            b = complex(float(c["real"]), float(c["imag"]))
            resolved = abs(a) > float(r["signal_floor"])
            out.append(dict(z=za, z_ct=zb, field=field, family=family(field),
                glm_status=r["status"], ct_status=c["status"],
                glm_abs=abs(a), ct_abs=abs(b),
                relative_amplitude_difference=abs(b)/abs(a)-1 if resolved else math.nan,
                phase_difference_rad=float(np.angle(b/a)) if resolved else math.nan))
    return sorted(out, key=lambda r: (-r["z"], r["field"]))


def summarize(rows, glm_gate, ct_gate, tolerances=None):
    tolerances = tolerances or (glm_gate["amplitude_tolerance"], glm_gate["phase_tolerance_rad"])
    summary = {}
    for fam in sorted({r["family"] for r in rows}):
        sel = [r for r in rows if r["family"] == fam and r["glm_status"] in ("PASS", "FAIL")
               and np.isfinite(r["relative_amplitude_difference"])]
        if not sel:
            continue
        summary[fam] = dict(n=len(sel),
            max_abs_relative_amplitude_difference=max(abs(r["relative_amplitude_difference"]) for r in sel),
            max_abs_phase_difference_rad=max(abs(r["phase_difference_rad"]) for r in sel))
    active = [r for r in rows if r["glm_status"] in ("PASS", "FAIL")
              and np.isfinite(r["relative_amplitude_difference"])]
    amp = max(abs(r["relative_amplitude_difference"]) for r in active)
    phase = max(abs(r["phase_difference_rad"]) for r in active)
    keep = ("status", "mhd_solver", "max_amplitude_error", "max_phase_error_rad", "active_observables")
    return dict(
        glm_gate={k: glm_gate.get(k) for k in keep}, ct_gate={k: ct_gate.get(k) for k in keep},
        amplitude_tolerance=tolerances[0], phase_tolerance_rad=tolerances[1],
        max_abs_relative_amplitude_difference=amp, max_abs_phase_difference_rad=phase,
        within_gate_tolerances=bool(amp <= tolerances[0] and phase <= tolerances[1]),
        families=summary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("glm", type=Path)
    parser.add_argument("ct", type=Path)
    parser.add_argument("--output", type=Path, help="default: <ct>/solver_comparison")
    args = parser.parse_args(argv)
    glm_rows, glm_gate = load(args.glm)
    ct_rows, ct_gate = load(args.ct)
    rows = differences(glm_rows, ct_rows)
    out = args.output or args.ct/"solver_comparison"
    out.mkdir(parents=True, exist_ok=False)
    with (out/"solver_differences.tsv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    summary = summarize(rows, glm_gate, ct_gate)
    summary["glm_directory"], summary["ct_directory"] = str(args.glm.resolve()), str(args.ct.resolve())
    (out/"solver_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
