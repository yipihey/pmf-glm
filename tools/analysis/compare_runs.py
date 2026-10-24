#!/usr/bin/env python3
"""Compare two production runs (for example GLM versus CT) from their TSV outputs.

``pmf-compare-runs RUN_A RUN_B`` pairs ``history.tsv`` rows and ``spectra.tsv`` epochs at
common redshifts (``|z_A - z_B| <= --z-tol``; never by cycle number) and reports the
relative difference of B relative to A: ``(B - A) / A``. Writes ``history_differences.tsv``,
``spectra_differences.tsv`` and ``run_comparison.json`` into ``--output``
(default ``RUN_B/run_comparison``, which must not exist). Nothing is fitted.

Values whose reference magnitude is below ``--floor`` times the largest magnitude of
that column (history) or of that field and epoch (spectra) are skipped, not reported as
zero (default 1e-8, i.e. well above Float32 round-off power; lower it deliberately). The divergence diagnostics are different quantities for the two solvers (centred
``divBdx_over_brms`` versus exact face ``divb_face_dx_over_brms``) and are listed
separately, never differenced.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import tomllib

DIVERGENCE_COLUMNS = ("divBdx_over_brms", "divb_face_dx_over_brms")
SKIP_COLUMNS = ("cycle", "measured", "z", "a") + DIVERGENCE_COLUMNS
SPECTRUM_FAMILY = {"B_vector": "magnetic", "gas_delta": "baryon_density", "dm_delta": "dark_matter"}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader((l for l in stream if l.strip() and not l.startswith("#")),
                                   delimiter="\t"))
    if not rows:
        raise ValueError(f"empty TSV: {path}")
    return rows


def _num(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return math.nan


def rel_diff(a: float, b: float, scale: float, floor: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b)) or abs(a) <= floor * scale or abs(a) == 0:
        return math.nan
    return (b - a) / a


def pair_redshifts(za: list[float], zb: list[float], z_tol: float) -> list[tuple[int, int]]:
    """Index pairs (i, j), each A epoch matched to the nearest unused B epoch within z_tol."""
    pairs, used = [], set()
    for i, z in enumerate(za):
        best = min((j for j in range(len(zb)) if j not in used), key=lambda j: abs(zb[j] - z),
                   default=None)
        if best is not None and abs(zb[best] - z) <= z_tol:
            pairs.append((i, best))
            used.add(best)
    return pairs


def compare_history(a_rows, b_rows, z_tol, floor):
    pairs = pair_redshifts([_num(r["z"]) for r in a_rows], [_num(r["z"]) for r in b_rows], z_tol)
    columns = [c for c in a_rows[0] if c in b_rows[0] and c not in SKIP_COLUMNS]
    scale = {c: max((abs(v) for v in (_num(r[c]) for r in a_rows) if math.isfinite(v)), default=0.0)
             for c in columns}
    out = []
    for i, j in pairs:
        row = {"z_a": _num(a_rows[i]["z"]), "z_b": _num(b_rows[j]["z"]),
               "cycle_a": a_rows[i].get("cycle"), "cycle_b": b_rows[j].get("cycle")}
        for c in columns:
            row[c] = rel_diff(_num(a_rows[i][c]), _num(b_rows[j][c]), scale[c], floor)
        out.append(row)
    return out, columns


def compare_spectra(a_rows, b_rows, z_tol, floor):
    def epochs(rows):
        seen = {}
        for r in rows:
            seen.setdefault((r["cycle"], r["z"]), _num(r["z"]))
        return seen
    ea, eb = epochs(a_rows), epochs(b_rows)
    ka, kb = list(ea), list(eb)
    pairs = pair_redshifts([ea[k] for k in ka], [eb[k] for k in kb], z_tol)
    a_by = {}
    for r in a_rows:
        a_by.setdefault((r["cycle"], r["z"], r["field"]), {})[(r["kbin"], r["mubin"])] = r
    b_by = {}
    for r in b_rows:
        b_by.setdefault((r["cycle"], r["z"], r["field"]), {})[(r["kbin"], r["mubin"])] = r
    out = []
    for i, j in pairs:
        for field in sorted({k[2] for k in a_by if k[:2] == ka[i]}):
            ra, rb = a_by[ka[i] + (field,)], b_by.get(kb[j] + (field,), {})
            scale = max((abs(_num(r["P"])) for r in ra.values() if math.isfinite(_num(r["P"]))),
                        default=0.0)
            for key, r in ra.items():
                if key not in rb or int(r["Nmodes"]) == 0:
                    continue
                d = rel_diff(_num(r["P"]), _num(rb[key]["P"]), scale, floor)
                if math.isfinite(d):
                    out.append({"field": field, "z_a": ea[ka[i]], "z_b": eb[kb[j]], "kbin": key[0],
                                "mubin": key[1], "k": _num(r["k"]), "P_a": _num(r["P"]),
                                "P_b": _num(rb[key]["P"]), "relative_difference": d})
    return out


def _stats(values):
    v = [abs(x) for x in values if math.isfinite(x)]
    return {"n": len(v), "max_abs": max(v) if v else None,
            "rms": math.sqrt(sum(x * x for x in v) / len(v)) if v else None}


def run_label(run: Path) -> dict:
    meta = {}
    if (run / "metadata.toml").is_file():
        with (run / "metadata.toml").open("rb") as stream:
            meta = tomllib.load(stream)
    return {"directory": str(run.resolve()), "mhd_solver": meta.get("mhd_solver", "unknown"),
            "status": meta.get("status", "unknown")}


def divergence_report(rows):
    last = rows[-1]
    return {c: {"first": _num(rows[0][c]), "last": _num(last[c]),
                "max": max((v for v in (_num(r[c]) for r in rows) if math.isfinite(v)), default=math.nan)}
            for c in DIVERGENCE_COLUMNS if c in rows[0]}


def compare_runs(a: Path, b: Path, z_tol=1e-3, floor=1e-8):
    summary = {"a": run_label(a), "b": run_label(b), "z_tolerance": z_tol, "floor": floor,
               "definition": "relative difference (B - A) / A at common redshifts"}
    hist_rows = spec_rows = None
    if (a / "history.tsv").is_file() and (b / "history.tsv").is_file():
        ha, hb = read_tsv(a / "history.tsv"), read_tsv(b / "history.tsv")
        hist_rows, columns = compare_history(ha, hb, z_tol, floor)
        if not hist_rows:
            raise ValueError("no common history redshifts within --z-tol")
        summary["history"] = {
            "common_epochs": len(hist_rows),
            "max_abs_z_mismatch": max(abs(r["z_a"] - r["z_b"]) for r in hist_rows),
            "columns": {c: _stats([r[c] for r in hist_rows]) for c in columns},
            "divergence_a": divergence_report(ha), "divergence_b": divergence_report(hb)}
    if (a / "spectra.tsv").is_file() and (b / "spectra.tsv").is_file():
        spec_rows = compare_spectra(read_tsv(a / "spectra.tsv"), read_tsv(b / "spectra.tsv"), z_tol, floor)
        fields = {}
        for f in sorted({r["field"] for r in spec_rows}):
            sel = [r for r in spec_rows if r["field"] == f]
            fields[f] = dict(_stats([r["relative_difference"] for r in sel]),
                             epochs=len({r["z_a"] for r in sel}),
                             family=SPECTRUM_FAMILY.get(f, "other"))
        summary["spectra"] = {"fields": fields}
    if hist_rows is None and spec_rows is None:
        raise ValueError("both runs need history.tsv and/or spectra.tsv")
    return summary, hist_rows, spec_rows


def _write(path: Path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_a", type=Path)
    p.add_argument("run_b", type=Path)
    p.add_argument("--output", type=Path, help="default: RUN_B/run_comparison (must not exist)")
    p.add_argument("--z-tol", type=float, default=1e-3, help="absolute redshift pairing tolerance")
    p.add_argument("--floor", type=float, default=1e-8,
                   help="skip reference values below this fraction of the column/field maximum")
    args = p.parse_args(argv)
    summary, hist, spec = compare_runs(args.run_a, args.run_b, args.z_tol, args.floor)
    out = args.output or args.run_b / "run_comparison"
    out.mkdir(parents=True, exist_ok=False)
    if hist:
        _write(out / "history_differences.tsv", hist)
    if spec:
        _write(out / "spectra_differences.tsv", spec)
    (out / "run_comparison.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    a, b = summary["a"], summary["b"]
    print(f"A: {a['mhd_solver']} {a['status']} {a['directory']}")
    print(f"B: {b['mhd_solver']} {b['status']} {b['directory']}")
    if "history" in summary:
        h = summary["history"]
        print(f"history: {h['common_epochs']} common epochs, max |dz|={h['max_abs_z_mismatch']:.3g}")
        for c, s in h["columns"].items():
            if s["n"]:
                print(f"  {c:24s} max|rel diff|={s['max_abs']:.3e} rms={s['rms']:.3e} (n={s['n']})")
        print("  divergence A:", json.dumps(h["divergence_a"]))
        print("  divergence B:", json.dumps(h["divergence_b"]))
    for f, s in summary.get("spectra", {}).get("fields", {}).items():
        print(f"spectra {f:10s} max|rel diff|={s['max_abs']:.3e} rms={s['rms']:.3e} "
              f"(n={s['n']} bins, {s['epochs']} epochs)")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
