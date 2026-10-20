#!/usr/bin/env python3
"""Build revision metadata, measured-evidence includes, and offline provenance."""
from __future__ import annotations

import datetime as dt
import hashlib
import html
import json
import pathlib
import re
import subprocess
import tomllib

DOCS = pathlib.Path(__file__).resolve().parents[1]
ROOT = DOCS.parent
SHORTCODE = re.compile(r'\{\{<\s*(?:src|test)\s+"([^"]+)"(?:\s+"([^"]+)")?\s*>\}\}')


def git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else ""


def write_source(relative: str) -> None:
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError(f"Invalid provenance source: {relative}")
    body = path.read_text()
    target = DOCS / "sources" / (relative + ".html")
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = "\n".join(
        f'<span id="L{i}"><a href="#L{i}" aria-label="Line {i}">{i:5}</a> {html.escape(line)}</span>'
        for i, line in enumerate(body.splitlines(), 1)
    )
    target.write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<title>{html.escape(relative)} | PMFGLM source</title>'
        '<style>body{font:16px/1.5 system-ui;margin:1rem;color:#1f2328;background:#fff}'
        'pre{overflow:auto;font-size:13px}span:target{background:#fff1b8}'
        'a{color:#0072b2}pre a{text-decoration:none}h1{font-size:1.2rem;overflow-wrap:anywhere}'
        '@media(prefers-color-scheme:dark){body{color:#eee;background:#17191c}'
        'a{color:#56b4e9}span:target{background:#414035}}</style></head><body>'
        f'<h1>{html.escape(relative)}</h1><p>Snapshot from this documentation build. '
        f'SHA-256: <code>{hashlib.sha256(path.read_bytes()).hexdigest()}</code></p>'
        f'<pre><code>{lines}</code></pre></body></html>\n'
    )


def source_fingerprint() -> str:
    paths = [ROOT / "Project.toml"]
    for folder in ("src", "lib", "scripts"):
        paths.extend(p for p in (ROOT / folder).rglob("*") if p.is_file() and (p.suffix == ".jl" or p.name == "Project.toml"))
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: p.relative_to(ROOT).as_posix()):
        if not path.is_file():
            return "unavailable"
        row = f"{path.relative_to(ROOT).as_posix()}\0{hashlib.sha256(path.read_bytes()).hexdigest()}\n"
        digest.update(row.encode())
    return digest.hexdigest()


def table_cell(value) -> str:
    return html.escape(str(value)).replace("|", "&#124;").replace("\n", " ")


def evidence_summary(data, current_fingerprint: str | None) -> str:
    if not isinstance(data, dict):
        return ""
    records = data.get("backends")
    if not isinstance(records, dict):
        records = {data.get("backend", "record"): data}
    rows = []
    for backend, record in records.items():
        if not isinstance(record, dict):
            continue
        status = record.get("status", "not reported")
        if status in ("NOT_RUN", "SKIPPED"):
            status = f"not yet measured ({status})"
        fingerprint = record.get("source_fingerprint_sha256")
        match = "not recorded"
        if fingerprint and current_fingerprint:
            match = "matches source at render" if fingerprint == current_fingerprint else "STALE: source differs"
        wall = record.get("wall_seconds", "not yet measured")
        rows.append("| " + " | ".join(table_cell(x) for x in (
            backend, status, record.get("assertions_passed", "not reported"), wall, match
        )) + " |")
    if not rows:
        return ""
    return "| Backend | Recorded status | Assertions passed | Wall seconds | Source identity |\n|---|---|---|---|---|\n" + "\n".join(rows) + "\n\n"


def evidence_section(path: pathlib.Path, current_fingerprint: str | None = None) -> str:
    data = json.loads(path.read_text()) if path.suffix == ".json" else tomllib.loads(path.read_text())
    if not isinstance(data, (dict, list)):
        raise ValueError(f"Evidence must be an object or list: {path}")
    encoded = html.escape(json.dumps(data, indent=2, ensure_ascii=True, default=str))
    logs = []
    records = data.get("backends", {"record": data}) if isinstance(data, dict) else {}
    for record in records.values():
        name = record.get("log") if isinstance(record, dict) else None
        if isinstance(name, str) and pathlib.Path(name).name == name and (path.parent / name).is_file():
            logs.append(f"[Execution log: {name}](data/{name})")
    return "".join([
        f"### {path.name}\n\n",
        evidence_summary(data, current_fingerprint) if path.name.startswith("validation") else "",
        f"[Download the original record](data/{path.name}). "
        "This is recorded evidence, not a fresh measurement by the documentation build.\n\n",
        " · ".join(logs) + "\n\n" if logs else "",
        '<details open><summary>Recorded metadata and measurements</summary>\n',
        f'<pre class="evidence-record"><code>{encoded}</code></pre>\n</details>\n',
    ])


def prepare() -> None:
    includes = DOCS / "_includes"
    includes.mkdir(exist_ok=True)
    commit = git("rev-parse", "HEAD")
    sha = commit[:9] if commit else "uncommitted extraction"
    if commit and git("status", "--porcelain", "--untracked-files=normal"):
        sha += " (working tree modified)"
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    variables = {"build": {"sha": sha, "commit": commit or "unavailable", "rendered": now}}
    # JSON scalar quoting is also valid YAML and prevents revision text changing structure.
    (DOCS / "_variables.yml").write_text(
        "build:\n" + "".join(f"  {key}: {json.dumps(value)}\n" for key, value in variables["build"].items())
    )
    referenced = set()
    for qmd in DOCS.glob("*.qmd"):
        for relative, symbol in SHORTCODE.findall(qmd.read_text()):
            path = ROOT / relative
            if symbol and (not path.is_file() or symbol not in path.read_text()):
                raise ValueError(f"Missing provenance symbol: {relative}: {symbol}")
            referenced.add(relative)
    for relative in sorted(referenced):
        write_source(relative)
    fingerprint = source_fingerprint()
    for category in ("validation", "benchmark", "production"):
        paths = sorted(p for p in (DOCS / "data").glob(category + "*") if p.suffix in (".json", ".toml"))
        sections = [evidence_section(path, fingerprint) for path in paths]
        text = "\n\n".join(sections) if sections else (
            f"**{category.capitalize()} results: not yet measured.** "
            "No generated evidence file is present in this build. "
            "This is not a passing result.\n"
        )
        (includes / f"{category}.qmd").write_text(text)
    runner = ROOT / "src" / "production" / "runner.jl"
    entries = []
    if runner.exists():
        body = runner.read_text()
        names = sorted(set(re.findall(r'get\(\s*ENV\s*,\s*"(MHD_[A-Z0-9_]+)"', body)))
        for name in names:
            line = body[:body.index('"' + name + '"')].count("\n") + 1
            entries.append(f"| `{name}` | [runner source](sources/src/production/runner.jl.html#L{line}) |")
        write_source("src/production/runner.jl")
    (includes / "environment.qmd").write_text(
        "| Environment variable | Parsing and defaults |\n|---|---|\n" + "\n".join(entries) + "\n"
        if entries else "Runner option index: **not yet available in this checkout**.\n"
    )
    print(f"Documentation prepared: {sha}; {len(referenced)} checked provenance files")


if __name__ == "__main__":
    prepare()
