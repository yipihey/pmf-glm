#!/usr/bin/env python3
"""Record byte hashes for this extraction; source checkouts are audit-only inputs."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


COPIED_REFERENCE = (
    "lmhd.py", "lya_rt.py", "sv25_paper.py", "build_p1_meanfield_handoff.py",
    "build_p1_multimode_handoff.py", "build_lya_response_table.py",
    "compare_matched_modes.py", "compare_multimode_to_oracle.py",
)
COPIED_TESTS = (
    "test_lmhd.py", "test_lya_rt.py", "test_p1_handoff.py", "test_multimode.py",
    "test_temporal_order.py", "test_sv25_paper.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def revision(root: Path) -> str | None:
    result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                            text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="sv25_reference source directory")
    parser.add_argument("--chemistry", type=Path, required=True, help="ChemistryKernels source root")
    parser.add_argument("--solver-schema", type=Path, required=True, help="pmf_dm_lattice_bench.jl inspected for schemas")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    source = args.source.resolve()
    pairs = [(source / name, root / "tools/reference" / name, "imports only") for name in COPIED_REFERENCE]
    pairs.append((source / "analyze_temporal_order.py", root / "tools/analysis/analyze_temporal_order.py", "unchanged"))
    pairs.extend((source / name, root / "test/python" / name,
                  "standalone imports; local RECFAST fixture; raster-only tests excluded") for name in COPIED_TESTS)
    pairs.extend([
        (args.chemistry / "test/fixtures/recfast_v2_xe.csv", root / "test/python/fixtures/recfast_v2_xe.csv", "unchanged"),
        (args.chemistry / "LICENSE", root / "test/python/fixtures/licenses/ChemistryKernels.LICENSE", "unchanged"),
        (source / "run_matched_vespa.py", root / "tools/reference/matched_units.py",
         "constants KPC_CM, HYDROGEN_MASS_FRACTION, OMEGA_M, OMEGA_B and two unit helpers only; positive-box validation added"),
    ])
    records = [dict(source_path=str(original.resolve()), destination_path=str(target.relative_to(root)),
                    source_sha256=sha256(original), extracted_sha256=sha256(target), adaptation=adaptation)
               for original, target, adaptation in pairs]
    for entry in records:
        if entry["destination_path"] == "tools/reference/build_p1_multimode_handoff.py":
            entry["adaptation"] = "standalone import; component arguments explicitly set amplitude_sign=1 to match current single-mode generator"
        elif entry["destination_path"] == "test/python/test_multimode.py":
            entry["adaptation"] = "standalone imports; regression test generates and combines two small modes"
    payload = {
        "schema": 1, "source_tree": str(source), "source_revision": revision(source.parents[1]),
        "content_basis": "working-tree bytes, including dirty and untracked content; not a git archive",
        "records": records,
        "inspected_not_copied": [dict(source_path=str(path.resolve()), sha256=sha256(path))
                                  for path in (args.solver_schema, args.chemistry / "src/ChemistryKernels.jl",
                                               args.chemistry / "test/fixtures/gen_recfast_v2.py")],
        "license_notes": {
            "vespa_runs": "No LICENSE or COPYING file found at source root or sv25_reference. No new license is inferred or granted; confirm redistribution rights before release.",
            "chemistry_fixture": "Original ChemistryKernels Enzo Public License retained verbatim beside the fixture.",
            "third_party_runtime": "NumPy, SciPy, Matplotlib and their dependencies are installed separately, not vendored.",
        },
        "excluded": ["campaigns", "production outputs", ".vendor", "PDFs and raster comparisons",
                     "run_paper_reproduction.py and other campaign launchers", "Julia/Metal runners",
                     "native nonlinear hierarchy validation"],
        "new_files": ["pyproject.toml", "tools/README.md", "tools/reference/__init__.py",
                      "tools/analysis/__init__.py", "tools/reference/run_reference.py",
                      "tools/reference/matched_scenario.py", "tools/reference/compare_reference.py",
                      "tools/reference/record_provenance.py", "tools/analysis/analyze_production.py",
                      "tools/analysis/render_density_frames.py",
                      "test/python/test_portable_tools.py", "test/python/fixtures/README.md",
                      "test/python/fixtures/metadata.toml",
                      "test/python/fixtures/history.tsv", "test/python/fixtures/spectra.tsv",
                      "test/python/fixtures/density_pdf.tsv"],
    }
    target = root / "tools/reference/provenance.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Recorded {len(records)} extracted source records in {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
