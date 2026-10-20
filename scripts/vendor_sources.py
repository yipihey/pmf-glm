#!/usr/bin/env python3
"""Maintainer-only, explicit extraction of the PMF kernel sources and provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT.parent)
    args = parser.parse_args()
    sources = {
        "MHDKernels": ("Vespa.jl", "lib/MHDKernels"),
        "PoissonKernels": ("Vespa.jl", "lib/PoissonKernels"),
        "ChemistryKernels": ("ChemistryKernels.jl", "."),
        "EmissionKernels": ("EmissionKernels.jl", "."),
    }
    records = {}
    for name, (repository, subdir) in sources.items():
        repo = args.source_root / repository
        source = repo / subdir
        dest = ROOT / "lib" / name
        if dest.exists():
            raise SystemExit(f"refusing to overwrite {dest}; review upstream updates explicitly")
        dest.mkdir(parents=True)
        for folder in ("src", "ext", "data", "docs"):
            if (source / folder).is_dir():
                shutil.copytree(source / folder, dest / folder)
        shutil.copy2(source / "Project.toml", dest / "Project.toml")
        if (repo / "LICENSE").is_file():
            shutil.copy2(repo / "LICENSE", dest / "LICENSE")
        records[name] = {
            "repository": git(repo, "remote", "get-url", "origin"),
            "commit": git(repo, "rev-parse", "HEAD"),
            "source_subdirectory": subdir,
            "working_tree_modified": bool(git(repo, "status", "--porcelain", "--", subdir)),
            "files": {
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(dest.rglob("*")) if p.is_file()
            },
        }
    tests = ROOT / "test" / "inherited"
    tests.mkdir(parents=True, exist_ok=True)
    for source in (args.source_root / "Vespa.jl/lib/MHDKernels/test").glob("*.jl"):
        if source.name in {"runtests.jl", "test_terminal_ct.jl", "test_drag_godunov.jl",
                           "test_radiation_coupling.jl", "test_recombination_transport.jl",
                           "test_passive_transport.jl", "test_terminal_handoff_overlap.jl"}:
            shutil.copy2(source, tests / source.name)
    runner = args.source_root / "Vespa.jl/lib/MHDKernels/test/pmf_dm_lattice_bench.jl"
    target = ROOT / "src" / "production" / "runner.jl"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(runner, target)
    records["production_runner"] = {
        "source": "Vespa.jl/lib/MHDKernels/test/pmf_dm_lattice_bench.jl",
        "sha256": hashlib.sha256(runner.read_bytes()).hexdigest(),
    }
    (ROOT / "provenance").mkdir(exist_ok=True)
    (ROOT / "provenance" / "extraction.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n")
    print("extracted four source packages, portable MHD tests, and production driver")


if __name__ == "__main__":
    main()
