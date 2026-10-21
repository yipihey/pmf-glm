#!/usr/bin/env python3
"""Render original runner frames.tsv slice/max-projection Float32 payloads."""

from __future__ import annotations

import argparse
import json
import hashlib
import math

from PIL import Image, ImageOps
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if __package__:
    from .analyze_production import read_tsv, save_figure
else:
    from analyze_production import read_tsv, save_figure


def load_frame(path: Path, resolution: int) -> np.ndarray:
    if resolution < 1 or path.stat().st_size != 4 * resolution**2:
        raise ValueError(f"expected exactly N*N Float32 values in {path}")
    values = np.fromfile(path, dtype="<f4").reshape((resolution, resolution), order="F")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError(f"density frame must be finite and nonnegative: {path}")
    return values


def render(frames: Path, output: Path, *, box_ckpc_h: float | None = None,
           hubble: float = 0.674, log_min: float = -2, log_max: float = 2) -> list[dict]:
    if not np.isfinite([hubble, log_min, log_max]).all() or hubble <= 0 or log_min >= log_max:
        raise ValueError("require finite limits log_min < log_max and h > 0")
    if box_ckpc_h is not None and (not np.isfinite(box_ckpc_h) or box_ckpc_h <= 0):
        raise ValueError("box size must be positive and finite")
    index = frames / "frames.tsv" if frames.is_dir() else frames
    rows = read_tsv(index, ("cycle", "z", "N", "slice_f32", "maxproj_f32", "layout"))
    rows.sort(key=lambda row: int(row["cycle"]))
    if len({int(row["cycle"]) for row in rows}) != len(rows):
        raise ValueError("duplicate frame cycles; resolve repeated restart epochs explicitly")
    if not rows or any(not math.isfinite(float(row["z"])) for row in rows):
        raise ValueError("require nonempty frames with finite redshifts")
    output.mkdir(parents=True, exist_ok=True)
    extent = 1.0 if box_ckpc_h is None else box_ckpc_h / hubble
    unit = "/ L" if box_ckpc_h is None else "[comoving kpc, no h]"
    records = []
    for row in rows:
        if row["layout"] != "julia-column-major":
            raise ValueError("unsupported frame layout")
        n = int(row["N"])
        figure, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
        for axis, column, title in zip(axes, ("slice_f32", "maxproj_f32"),
                                       ("Central density slice", "Line-of-sight maximum density")):
            path = index.parent / row[column]
            if not path.resolve().is_relative_to(index.parent.resolve()):
                raise ValueError("frame payload must be inside the frame directory")
            values = load_frame(path, n)
            plotted = np.log10(np.maximum(values, 10.0**log_min))
            image = axis.imshow(plotted.T, origin="lower", extent=(0, extent, 0, extent),
                                vmin=log_min, vmax=log_max, cmap="viridis", interpolation="nearest")
            axis.set(xlabel=f"x {unit}", ylabel=f"y {unit}", title=title)
        figure.colorbar(image, ax=axes, label="log10(rho / reference density)", shrink=0.8)
        figure.suptitle(f"z={float(row['z']):.8g}, cycle={int(row['cycle'])}")
        stem = f"density_cycle{int(row['cycle']):07d}_z{float(row['z']):.12g}"
        save_figure(figure, output / stem)
        records.append(dict(cycle=int(row["cycle"]), z=float(row["z"]), resolution=n, figure_stem=stem,
                            payload_sha256={column: hashlib.sha256((index.parent / row[column]).read_bytes()).hexdigest()
                                            for column in ("slice_f32", "maxproj_f32")}))
    (output / "frames.json").write_text(json.dumps(dict(scope="Density frames, not a column-density projection",
        log10_limits=[log_min, log_max], box_ckpc_h=box_ckpc_h, hubble=hubble, frames=records),
        indent=2, sort_keys=True, allow_nan=False) + "\n")
    return records


def animate(output: Path, records: list[dict], destination: Path, *, fps: float = 8) -> None:
    """Encode only this manifest's cycle-ordered frames with fixed color limits.

    Frames are equally spaced in playback time; captions retain actual epochs.
    PNG padding accommodates differing caption widths without distorting maps.
    """
    if not math.isfinite(fps) or fps <= 0 or not records:
        raise ValueError("require finite positive fps and at least one frame")
    if destination.suffix.lower() != ".gif":
        raise ValueError("animation destination must end in .gif")
    images = []
    for record in sorted(records, key=lambda r: r["cycle"]):
        with Image.open(output / (record["figure_stem"] + ".png")) as source:
            images.append(source.convert("RGB"))
    width = max(im.width for im in images)
    height = max(im.height for im in images)
    padded = [ImageOps.pad(im, (width, height), color="white", method=Image.Resampling.NEAREST)
              if im.size == (width, height) else
              ImageOps.expand(im, (0, 0, width-im.width, height-im.height), fill="white")
              for im in images]
    destination.parent.mkdir(parents=True, exist_ok=True)
    padded[0].save(destination, save_all=True, append_images=padded[1:],
                   duration=max(10, round(1000/fps)), loop=0, disposal=2)
    manifest = json.loads((output / "frames.json").read_text())
    manifest["animation"] = dict(path=str(destination.resolve()), fps=fps,
                                 spacing="equal playback intervals, actual redshifts in captions")
    (output / "frames.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames", type=Path, help="frames directory or frames.tsv")
    parser.add_argument("output", type=Path)
    parser.add_argument("--box-ckpc-h", type=float)
    parser.add_argument("--hubble", type=float, default=0.674)
    parser.add_argument("--log-min", type=float, default=-2)
    parser.add_argument("--log-max", type=float, default=2)
    parser.add_argument("--gif", type=Path, help="encode a GIF directly (no FFmpeg required)")
    parser.add_argument("--fps", type=float, default=8)
    args = parser.parse_args()
    options = vars(args).copy()
    gif = options.pop("gif")
    fps = options.pop("fps")
    records = render(**options)
    if gif is not None:
        animate(args.output, records, gif, fps=fps)
    print(f"Rendered {len(records)} density slice/projection pairs: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
