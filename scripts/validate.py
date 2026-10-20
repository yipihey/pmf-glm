#!/usr/bin/env python3
"""Run validation and publish measured, fingerprinted evidence for the Quadrant site."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def fingerprint() -> str:
    digest = hashlib.sha256()
    paths = [ROOT / "Project.toml"]
    for folder in ("src", "lib", "scripts"):
        paths += [p for p in (ROOT / folder).rglob("*")
                  if p.is_file() and (p.suffix == ".jl" or p.name == "Project.toml")]
    for path in sorted(paths, key=lambda p: p.relative_to(ROOT).as_posix()):
        line = f"{path.relative_to(ROOT).as_posix()}\0{hashlib.sha256(path.read_bytes()).hexdigest()}\n"
        digest.update(line.encode())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--julia", default="julia")
    parser.add_argument("--project", default=str(ROOT))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--backend", choices=("cpu","metal","cuda"), default="cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "data")
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.backend == "cpu":
        script = ROOT / "test" / "runtests.jl"
        extra = []
    else:
        script = ROOT / "scripts" / "test_gpu.jl"
        extra = [f"--backend={args.backend}"]
    command = [args.julia, f"--project={args.project}", f"--threads={args.threads}", str(script), *extra]
    log = args.output / f"validation-{args.backend}.log"
    print("Running:"," ".join(command),flush=True)
    start = time.perf_counter()
    with log.open("w") as stream:
        result = subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    elapsed = time.perf_counter()-start
    text = log.read_text()
    counts = [(int(m[1]),int(m[2])) for m in re.finditer(r"\|\s+(\d+)\s+(\d+)\s+[\d.]+s",text)]
    passes = sum(p for p,total in counts if p == total)
    evidence = {
        "schema_version":1,"backend":args.backend,
        "status":"PASS" if result.returncode==0 and passes>0 else "FAIL",
        "assertions_passed":passes,"wall_seconds":elapsed,"returncode":result.returncode,
        "created_utc":datetime.now(timezone.utc).isoformat(),"machine":platform.node(),
        "platform":platform.platform(),"threads":args.threads,
        "source_fingerprint_sha256":fingerprint(),"command":command,
        "log":log.name,"scope":"current standalone extraction; automated numerical/operator tests",
    }
    destination = args.output / f"validation-{args.backend}.json"
    destination.write_text(json.dumps(evidence,indent=2)+"\n")
    combined = {"schema_version":1,"backends":{}}
    for name in ("cpu","metal","cuda"):
        path = args.output / f"validation-{name}.json"
        combined["backends"][name] = json.loads(path.read_text()) if path.exists() else {
            "status":"NOT_RUN","scope":"hardware validation has not been performed in this extraction"}
    (args.output / "validation.json").write_text(json.dumps(combined,indent=2)+"\n")
    print(json.dumps(evidence,indent=2))
    if result.returncode:
        print(text[-12000:])
    return 0 if evidence["status"]=="PASS" else 1


if __name__=="__main__":
    raise SystemExit(main())
