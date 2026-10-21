#!/usr/bin/env python3
"""Analyze pmf_dm_lattice_bench.jl TSV histories, spectra, and density PDFs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import tomllib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_tsv(path: Path, required: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        reader = csv.DictReader((line for line in stream if line.strip() and not line.startswith("#")),
                                delimiter="\t")
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError(f"missing or duplicate TSV header in {path}")
        missing = set(required) - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        rows = list(reader)
    if not rows or any(None in row or None in row.values() for row in rows):
        raise ValueError(f"empty or malformed TSV: {path}")
    return rows


def length_in_mpc(unit: str, hubble: float, box_ckpc_h: float | None,
                  spectral_boxsize: float) -> float:
    """Comoving Mpc per coordinate unit passed to power_spectrum_aniso_gpu."""
    if not math.isfinite(hubble) or hubble <= 0:
        raise ValueError("hubble must be finite and positive")
    if unit == "cMpc":
        return 1.0
    if unit == "ckpc":
        return 0.001
    if unit == "ckpc/h":
        return 0.001 / hubble
    if unit != "box" or box_ckpc_h is None or not math.isfinite(box_ckpc_h) or box_ckpc_h <= 0:
        raise ValueError("box coordinates require a positive --box-ckpc-h")
    if not math.isfinite(spectral_boxsize) or spectral_boxsize <= 0:
        raise ValueError("spectral boxsize must be finite and positive")
    return box_ckpc_h * 0.001 / hubble / spectral_boxsize


def isotropic_spectra(rows: list[dict], length_mpc: float) -> list[dict]:
    """Mode-count weighted angular average; retain cycle identity at repeated z."""
    if not math.isfinite(length_mpc) or length_mpc <= 0:
        raise ValueError("coordinate length must be finite and positive")
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        key = (row["field"], int(row["cycle"]), int(row["measured"]),
               float(row["z"]), int(row["kbin"]))
        angular_key = (*key, int(row["mubin"]))
        if angular_key in seen:
            raise ValueError(f"duplicate angular spectrum record: {angular_key}")
        seen.add(angular_key)
        count = float(row["Nmodes"])
        if not math.isfinite(count) or count < 0 or not count.is_integer():
            raise ValueError("Nmodes must be a nonnegative integer")
        if count == 0:
            continue  # Production writes NaN k/P for unpopulated bins.
        k, power = float(row["k"]), float(row["P"])
        if not all(math.isfinite(value) for value in (k, power, key[3])) or k <= 0 or power < 0:
            raise ValueError("populated spectral bins require finite k > 0, P >= 0, z")
        groups[key].append((k, power, count))
    result = []
    for (field, cycle, measured, z, kbin), values in sorted(groups.items()):
        total = sum(value[2] for value in values)
        k = sum(value[0] * value[2] for value in values) / total / length_mpc
        power = sum(value[1] * value[2] for value in values) / total * length_mpc**3
        result.append(dict(field=field, cycle=cycle, measured=measured, z=z, kbin=kbin,
                           k_comoving_mpc_inv=k, P_field_code2_mpc3=power,
                           Delta2_field_code2=k**3 * power / (2 * math.pi**2), Nmodes=int(total)))
    if not result:
        raise ValueError("no populated spectral bins")
    return result


def pdf_integrals(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        key = (row["field"], int(row["cycle"]), float(row["z"]))
        identity = (*key, int(row["bin"]))
        if identity in seen:
            raise ValueError(f"duplicate PDF bin: {identity}")
        seen.add(identity)
        lo, hi, value, count = (float(row[name]) for name in ("logrho_lo", "logrho_hi", "pdf", "count"))
        if not all(math.isfinite(x) for x in (lo, hi, value, count, key[2])) or hi <= lo or min(value, count) < 0:
            raise ValueError("PDF bins require finite increasing edges and nonnegative values/counts")
        groups[key].append((value * (hi - lo), count))
    return [dict(field=field, cycle=cycle, z=z,
                 integral_per_log10_rho=sum(v[0] for v in values),
                 included_cell_count=sum(v[1] for v in values))
            for (field, cycle, z), values in sorted(groups.items())]


def save_figure(figure, stem: Path) -> None:
    for extension in ("png", "pdf"):
        figure.savefig(Path(str(stem) + "." + extension), dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_history(rows: list[dict], output: Path) -> None:
    rows = sorted(rows, key=lambda row: (int(row["cycle"]), int(row["measured"])))
    z = np.asarray([float(row["z"]) for row in rows])
    if not np.all(np.isfinite(z)):
        raise ValueError("history redshifts must be finite")
    use_cycles = np.unique(z).size == 1
    x = np.asarray([int(row["cycle"]) for row in rows]) if use_cycles else z
    figure, axes = plt.subplots(2, 3, figsize=(11, 6), layout="constrained")
    groups = (
        (("delta_b_rms", "delta_dm_rms"), "Density contrast RMS"),
        (("brms", "btrans_rms"), "Magnetic RMS [code units]"),
        (("xHII_mean", "xe_total_mean"), "Abundance per H nucleus"),
        (("fH2_mean", "xHeII_mean"), "Abundance (production convention)"),
        (("temperature_mean",), "Temperature [K]"),
        (("divBdx_over_brms",), "divB dx / B RMS"),
    )
    for axis, (fields, ylabel) in zip(axes.flat, groups):
        for field in fields:
            values = np.asarray([float(row.get(field, "nan")) for row in rows])
            if not np.any(np.isfinite(values)):
                continue
            line, = axis.plot(x, values, marker=".", label=field)
            if field.endswith("_mean"):
                prefix = field[:-5]
                lower = [float(row.get(prefix + "_min", "nan")) for row in rows]
                upper = [float(row.get(prefix + "_max", "nan")) for row in rows]
                axis.fill_between(x, lower, upper, color=line.get_color(), alpha=0.12)
        axis.set(xlabel="Cycle" if use_cycles else "Redshift z", ylabel=ylabel)
        axis.ticklabel_format(axis="x", style="plain", useOffset=False)
        if not use_cycles:
            axis.invert_xaxis()
        axis.grid(alpha=0.2)
        if axis.lines:
            axis.legend(fontsize=7)
    save_figure(figure, output / "history")


def plot_streaming(rows: list[dict], output: Path) -> None:
    rows = sorted(rows, key=lambda r: int(r["cycle"]))
    columns = ("z", "vbc_x_cms", "vbc_y_cms", "vbc_z_cms", "vbc_rms_cms", "vbc_dispersion_cms")
    values = np.array([[float(r[c]) for c in columns] for r in rows])
    if not np.isfinite(values).all():
        raise ValueError("streaming diagnostics require finite epochs and velocities")
    figure, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    for axis, indices in zip(axes, ((1, 2, 3), (4, 5))):
        for i in indices:
            axis.plot(values[:, 0], values[:, i]/1e5, marker=".", label=columns[i].replace("_cms", ""))
        axis.set(xlabel="Redshift z", ylabel="Peculiar velocity [km/s]")
        axis.invert_xaxis()
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle("Baryon minus DM velocity at DM positions (DM mass weighted)")
    save_figure(figure, output / "streaming")


def sample_epochs(rows: list[dict], maximum: int = 3) -> list[tuple[int, float]]:
    epochs = sorted({(int(row["cycle"]), float(row["z"])) for row in rows})
    indices = np.unique(np.linspace(0, len(epochs) - 1, min(maximum, len(epochs)), dtype=int))
    return [epochs[index] for index in indices]


def plot_spectra(rows: list[dict], output: Path, epochs=None, stem: str = "spectra") -> None:
    categories = [("gas_delta", "dm_delta"), ("B_vector",),
                  ("B_pressure_force", "B_tension_force", "B_lorentz_force"),
                  ("photon_delta", "photon_velocity", "photon_baryon_slip")]
    known = {field for fields in categories for field in fields}
    categories.extend((field,) for field in sorted({row["field"] for row in rows} - known))
    nrows = math.ceil(len(categories) / 2)
    figure, axes = plt.subplots(nrows, 2, figsize=(10, 3.5 * nrows), layout="constrained")
    epochs = sample_epochs(rows) if epochs is None else epochs
    styles = ("-", "--", ":")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for axis, fields in zip(axes.flat, categories):
        for field_index, field in enumerate(fields):
            for style, (cycle, z) in zip(styles, epochs):
                selected = sorted((r for r in rows if r["field"] == field and r["cycle"] == cycle and r["z"] == z),
                                  key=lambda r: r["k_comoving_mpc_inv"])
                selected = [r for r in selected if r["P_field_code2_mpc3"] > 0]
                if selected:
                    axis.loglog([r["k_comoving_mpc_inv"] for r in selected],
                                [r["P_field_code2_mpc3"] for r in selected],
                                linestyle=style, marker=".", color=colors[field_index % len(colors)],
                                label=f"{field}, z={z:.8g}")
        axis.set(xlabel="Comoving k [Mpc^-1, no h]", ylabel="P [field-code^2 Mpc^3]")
        axis.set_title(", ".join(fields), fontsize=8)
        axis.grid(alpha=0.2)
        if axis.lines:
            axis.legend(fontsize=6)
    for axis in list(axes.flat)[len(categories):]:
        axis.set_visible(False)
    save_figure(figure, output / stem)


def plot_pdfs(rows: list[dict], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 4), layout="constrained")
    for field in sorted({row["field"] for row in rows}):
        for cycle, z in sample_epochs(rows):
            selected = sorted((r for r in rows if r["field"] == field and int(r["cycle"]) == cycle and float(r["z"]) == z),
                              key=lambda r: float(r["logrho_lo"]))
            if selected:
                x = [0.5 * (float(r["logrho_lo"]) + float(r["logrho_hi"])) for r in selected]
                axis.plot(x, [float(r["pdf"]) for r in selected], label=f"{field}, z={z:g}")
    axis.set(xlabel="log10(rho / reference density)", ylabel="Volume PDF per log10(rho)")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    save_figure(figure, output / "density_pdf")


def discover_run(run: Path) -> tuple[dict, dict]:
    files = {name: run / filename for name, filename in
             (("history", "history.tsv"), ("spectra", "spectra.tsv"), ("pdf", "pdfs.tsv"), ("streaming", "streaming.tsv"))
             if (run / filename).is_file()}
    if not files:
        raise ValueError(f"no history.tsv, spectra.tsv, or pdfs.tsv in {run}")
    metadata_path = run / "metadata.toml"
    metadata = {}
    if metadata_path.is_file():
        with metadata_path.open("rb") as stream:
            metadata = tomllib.load(stream)
    return files, metadata


def analyze(output: Path, *, history: Path | None = None, spectra: Path | None = None,
            pdf: Path | None = None, streaming: Path | None = None, length_unit: str | None = None, hubble: float | None = None,
            box_ckpc_h: float | None = None, spectral_boxsize: float | None = None,
            run: Path | None = None) -> dict:
    run_metadata = {}
    if run is not None:
        files, run_metadata = discover_run(run)
        history, spectra, pdf = history or files.get("history"), spectra or files.get("spectra"), pdf or files.get("pdf")
        streaming = streaming or files.get("streaming")
        if output.resolve() == run.resolve():
            raise ValueError("analysis output must not be the input run directory")
    environment = run_metadata.get("environment", {})
    cosmology = run_metadata.get("configuration", {}).get("cosmology", {})
    if spectral_boxsize is None:
        spectral_boxsize = float(environment.get("MHD_BOXSIZE", 1.0))
    if hubble is None:
        hubble = float(environment.get("MHD_H", cosmology.get("h", 0.674)))
    unit_origin = "explicit command arguments"
    if spectra and length_unit is None and run_metadata:
        # The numerical FFT box and physical comoving box are distinct inputs.
        if box_ckpc_h is None:
            if "MHD_TERMINAL_LBOX_CM" in environment:
                box_ckpc_h = float(environment["MHD_TERMINAL_LBOX_CM"]) / 3.0856775814913673e21 * hubble
            elif "MHD_TERMINAL_LBOX_CKPC" in environment:
                box_ckpc_h = float(environment["MHD_TERMINAL_LBOX_CKPC"]) * hubble
            else:
                box_ckpc_h = cosmology.get("box_ckpc_h")
        if box_ckpc_h is not None:
            length_unit = "box"
            unit_origin = "run metadata physical box and MHD_BOXSIZE (default 1)"
    if not any((history, spectra, pdf, streaming)):
        raise ValueError("supply at least one history, spectra, or PDF TSV")
    if spectra and length_unit is None:
        raise ValueError("spectra require an explicit --length-unit; units are not in the TSV")
    output.mkdir(parents=True, exist_ok=True)
    summary = {"schema": 1, "scope": "Production-output diagnostics, not a solver certification",
               "inputs": {}, "notes": [
                   "P(k) is Nmodes-weighted across mu bins; zero-count NaN bins are omitted.",
                   "Only coordinate units are converted; magnetic/force/velocity amplitudes retain code units.",
                   "Pressure and tension powers cannot be added to infer Lorentz power without the cross term.",
                   "PDF is volume-weighted per log10(rho), not per rho; finite range may omit probability.",
                   "Initial measured=0 records are retained."]}
    if run is not None:
        summary["run"] = dict(path=str(run.resolve()), status=run_metadata.get("status", "UNKNOWN"))
        if (run / "metadata.toml").is_file():
            summary["inputs"]["metadata"] = dict(path=str((run / "metadata.toml").resolve()),
                sha256=hashlib.sha256((run / "metadata.toml").read_bytes()).hexdigest())
    for label, path in (("history", history), ("spectra", spectra), ("pdf", pdf), ("streaming", streaming)):
        if path:
            summary["inputs"][label] = dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if history:
        rows = read_tsv(history, ("cycle", "measured", "z", "brms", "delta_b_rms"))
        plot_history(rows, output)
        summary["history_rows"] = len(rows)
        summary["history_finite_ranges"] = {}
        for field in rows[0]:
            values = np.asarray([float(row[field]) for row in rows])
            finite = values[np.isfinite(values)]
            summary["history_finite_ranges"][field] = ([float(finite.min()), float(finite.max())]
                                                       if finite.size else None)
    if streaming:
        rows = read_tsv(streaming, ("cycle", "z", "vbc_x_cms", "vbc_y_cms", "vbc_z_cms", "vbc_rms_cms", "vbc_dispersion_cms"))
        plot_streaming(rows, output)
        summary["streaming_rows"] = len(rows)
        summary["notes"].append("Streaming is v_b-v_c sampled at actual DM positions, weighted by equal particle mass; not a volume-weighted velocity spectrum.")
    if spectra:
        rows = read_tsv(spectra, ("field", "cycle", "measured", "z", "kbin", "mubin", "k", "P", "Nmodes"))
        scale = length_in_mpc(length_unit, hubble, box_ckpc_h, spectral_boxsize)
        isotropic = isotropic_spectra(rows, scale)
        with (output / "isotropic_spectra.tsv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(isotropic[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(isotropic)
        plot_spectra(isotropic, output)
        epoch_stems = []
        for cycle, z in sorted({(row["cycle"], row["z"]) for row in isotropic}):
            stem = f"spectra_cycle{cycle:07d}_z{z:.12g}"
            plot_spectra(isotropic, output, epochs=[(cycle, z)], stem=stem)
            epoch_stems.append(stem)
        summary["per_epoch_spectra"] = epoch_stems
        summary["isotropic_rows"] = len(isotropic)
        summary["spectral_fields"] = sorted({row["field"] for row in isotropic})
        summary["spectral_units"] = dict(input_length_unit=length_unit, mpc_per_coordinate_unit=scale,
                                         k="comoving Mpc^-1 (no h)", P="field-code^2 comoving Mpc^3",
                                         Delta2="field-code^2", hubble=hubble,
                                         box_ckpc_h=box_ckpc_h, spectral_boxsize=spectral_boxsize,
                                         declaration_source=unit_origin)
    if pdf:
        rows = read_tsv(pdf, ("field", "cycle", "z", "bin", "logrho_lo", "logrho_hi", "count", "pdf"))
        summary["pdf_integrals"] = pdf_integrals(rows)
        plot_pdfs(rows, output)
    (output / "analysis.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="input run directory, or output directory with explicit input files")
    parser.add_argument("--output", type=Path, help="plot directory (default RUN_DIR/analysis in run-directory mode)")
    parser.add_argument("--run", type=Path, help="explicit input run directory; positional path becomes output")
    parser.add_argument("--history", type=Path)
    parser.add_argument("--streaming", type=Path)
    parser.add_argument("--spectra", type=Path)
    parser.add_argument("--pdf", "--pdfs", dest="pdf", type=Path)
    parser.add_argument("--length-unit", choices=("box", "cMpc", "ckpc", "ckpc/h"),
                        help="coordinate unit supplied to the production spectrum routine")
    parser.add_argument("--box-ckpc-h", type=float)
    parser.add_argument("--spectral-boxsize", type=float,
                        help="production MHD_BOXSIZE value when --length-unit=box")
    parser.add_argument("--hubble", type=float, help="h (default from run metadata, otherwise 0.674)")
    args = parser.parse_args()
    options = vars(args)
    path, output = options.pop("path"), options.pop("output")
    if options["run"] is None and not any(options[name] for name in ("history", "spectra", "pdf", "streaming")):
        options["run"] = path
        output = output or path / "analysis"
    else:
        output = output or path
    try:
        analyze(output, **options)
    except ValueError as error:
        parser.error(str(error))
    print(f"Analysis JSON, isotropic TSV, and PNG/PDF figures: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
