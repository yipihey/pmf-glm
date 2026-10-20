#!/usr/bin/env python3
"""Check that the distribution does not depend on a developer's checkout layout."""
from __future__ import annotations
from pathlib import Path
import re
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
errors = []
for project in (ROOT / "Project.toml", *sorted((ROOT / "environments").glob("*/Project.toml"))):
    config = tomllib.loads(project.read_text())
    for name, source in config.get("sources",{}).items():
        path = (project.parent/source["path"]).resolve()
        if not path.is_relative_to(ROOT) or not (path/"Project.toml").is_file():
            errors.append(f"nonlocal or missing source {name}: {project.relative_to(ROOT)}")
        if name not in config.get("deps",{}):
            errors.append(f"source {name} is not a direct active-root dependency")
for folder in ("src","scripts","tools","environments"):
    for path in (ROOT/folder).rglob("*"):
        if path.resolve() == Path(__file__).resolve():
            continue
        if path.suffix not in (".jl",".py",".toml"):
            continue
        text = path.read_text()
        if re.search(r"(?:/Users/[^/\s]+/Projects|/home/[^/\s]+/|\.\./Vespa\.jl|\.\./vespa-runs)",text):
            errors.append(f"developer-specific runtime path: {path.relative_to(ROOT)}")
if errors:
    print("\n".join(errors),file=sys.stderr)
    raise SystemExit(1)
print("PASS: active-root sources are internal; runtime source has no developer checkout paths")
