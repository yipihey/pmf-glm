#!/usr/bin/env python3
"""Build a weak broad-band P1 handoff by superposing resolved linear modes."""

from __future__ import annotations

import argparse
from argparse import Namespace
import json
import math
from pathlib import Path
import shutil

import numpy as np

if __package__:
    from . import build_p1_meanfield_handoff as single
else:
    import build_p1_meanfield_handoff as single


GAMMA_GAS = 5.0 / 3.0
REAL_FIELDS = (
    *(f"mhd_U_{index}" for index in range(1, 10)),
    "particle_px", "particle_py", "particle_pz",
    "particle_vx", "particle_vy", "particle_vz",
    "chem_eint", "chem_HII", "chem_H2I", "chem_HeII", "gas_delta",
)


def parse_modes(value: str) -> tuple[int, ...]:
    modes = tuple(sorted({int(item) for item in value.split(",") if item.strip()}))
    if not modes or any(mode <= 0 for mode in modes):
        raise argparse.ArgumentTypeError("modes must be positive comma-separated integers")
    return modes


def read_real(path: Path, name: str, n: int) -> np.ndarray:
    value = np.fromfile(path / f"{name}.f32", dtype=np.float32)
    if value.size != n**3:
        raise ValueError(f"{name}: got {value.size} elements, expected {n**3}")
    return value.reshape((n, n, n), order="F").astype(np.float64)


def read_photons(path: Path, n: int) -> np.ndarray:
    shape = (n // 2 + 1, n, n, 4)
    value = np.fromfile(path / "photon_hat_batch.c32", dtype=np.complex64)
    if value.size != int(np.prod(shape)):
        raise ValueError("unexpected photon batch size")
    return value.reshape(shape, order="F").astype(np.complex128)


def component_args(args: argparse.Namespace, output: Path, mode: int, epsilon: float) -> Namespace:
    return Namespace(
        output=output,
        resolution=args.resolution,
        zstart=args.zstart,
        zhandoff=args.zhandoff,
        box_ckpc_h=args.box_ckpc_h,
        hubble=args.hubble,
        field_pg=args.field_pg,
        epsilon=epsilon,
        amplitude_sign=1,
        theta=args.theta,
        kmode=mode,
        velocity_unit=args.velocity_unit,
        max_dlna=args.max_dlna,
        response_table=args.response_table,
        nonlocal_scale=args.nonlocal_scale,
        hydrogen_storage="centered",
    )


def build_components(
    args: argparse.Namespace, amplitudes: dict[int, float], label: str
) -> list[tuple[int, Path, dict[str, object]]]:
    result = []
    for mode, epsilon in amplitudes.items():
        path = args.output / "components" / label / f"kmode-{mode}"
        metadata = single.build_handoff(component_args(args, path, mode, epsilon))
        result.append((mode, path, metadata))
    return result


def composite_density_rms(
    components: list[tuple[int, Path, dict[str, object]]], n: int
) -> float:
    density = np.zeros((n, n, n), dtype=np.float64)
    for _, path, _ in components:
        density += read_real(path, "gas_delta", n)
    return float(np.sqrt(np.mean(density * density)))


def combine(args: argparse.Namespace, components) -> dict[str, object]:
    n = args.resolution
    loaded = [
        (
            mode,
            {name: read_real(path, name, n) for name in REAL_FIELDS},
            read_photons(path, n),
            metadata,
        )
        for mode, path, metadata in components
    ]
    arrays: dict[str, np.ndarray] = {}

    rho = 1.0 + sum(item[1]["gas_delta"] for item in loaded)
    arrays["mhd_U_1"] = rho
    for name in ("mhd_U_2", "mhd_U_3", "mhd_U_4"):
        velocity = sum(
            item[1][name] / item[1]["mhd_U_1"] for item in loaded
        )
        arrays[name] = rho * velocity

    guide_x = float(loaded[0][3]["guide_field_code"])
    guide_bx = guide_x * math.cos(args.theta)
    guide_by = guide_x * math.sin(args.theta)
    arrays["mhd_U_6"] = guide_bx + sum(
        item[1]["mhd_U_6"] - guide_bx for item in loaded
    )
    arrays["mhd_U_7"] = guide_by + sum(
        item[1]["mhd_U_7"] - guide_by for item in loaded
    )
    arrays["mhd_U_8"] = sum(item[1]["mhd_U_8"] for item in loaded)
    arrays["mhd_U_9"] = np.zeros_like(rho)

    pressure_components = []
    for _, fields, _, _ in loaded:
        kinetic = 0.5 * (
            fields["mhd_U_2"] ** 2
            + fields["mhd_U_3"] ** 2
            + fields["mhd_U_4"] ** 2
        ) / fields["mhd_U_1"]
        magnetic = 0.5 * sum(fields[name] ** 2 for name in (
            "mhd_U_6", "mhd_U_7", "mhd_U_8"
        ))
        pressure_components.append(
            (GAMMA_GAS - 1.0) * (fields["mhd_U_5"] - kinetic - magnetic)
        )
    pressure0 = float(np.mean(pressure_components[0]))
    pressure = pressure0 + sum(
        value - float(np.mean(value)) for value in pressure_components
    )
    kinetic = 0.5 * (
        arrays["mhd_U_2"] ** 2
        + arrays["mhd_U_3"] ** 2
        + arrays["mhd_U_4"] ** 2
    ) / rho
    magnetic = 0.5 * sum(arrays[name] ** 2 for name in (
        "mhd_U_6", "mhd_U_7", "mhd_U_8"
    ))
    arrays["mhd_U_5"] = pressure / (GAMMA_GAS - 1.0) + kinetic + magnetic

    for name in (
        "particle_px", "particle_py", "particle_pz",
        "particle_vx", "particle_vy", "particle_vz",
        "chem_HII", "chem_H2I",
    ):
        arrays[name] = sum(item[1][name] for item in loaded)
    helium0 = float(np.mean(loaded[0][1]["chem_HeII"]))
    arrays["chem_HeII"] = helium0 + sum(
        item[1]["chem_HeII"] - float(np.mean(item[1]["chem_HeII"]))
        for item in loaded
    )
    arrays["chem_eint"] = pressure / ((GAMMA_GAS - 1.0) * rho)
    arrays["gas_delta"] = rho - 1.0
    arrays["photon_hat_batch"] = sum(item[2] for item in loaded)

    if np.min(rho) <= 0.0 or np.min(pressure) <= 0.0:
        raise ValueError("composite handoff violates density or pressure positivity")

    args.output.mkdir(parents=True, exist_ok=True)
    single.write_arrays(args.output, arrays)
    first_component = components[0][1]
    for name in (
        "chem_hydrogen_storage.txt", "chem_hydrogen_reference.txt", "runtime.tsv"
    ):
        shutil.copy2(first_component / name, args.output / name)
    metadata = {
        "schema": 1,
        "status": "PASS",
        "scope": "weak multimode superposition of independent photon-P1 modes",
        "reference_source_fingerprint_sha256": loaded[0][3][
            "reference_source_fingerprint_sha256"
        ],
        "resolution": n,
        "zstart": args.zstart,
        "zhandoff": args.zhandoff,
        "box_ckpc_h": args.box_ckpc_h,
        "field_pG_comoving": args.field_pg,
        "theta_rad": args.theta,
        "spectral_epsilon": args.spectral_epsilon,
        "target_gas_delta_rms": args.target_gas_delta_rms,
        "measured_gas_delta_rms": float(np.sqrt(np.mean((rho - 1.0) ** 2))),
        "rho_min": float(np.min(rho)),
        "rho_max": float(np.max(rho)),
        "pressure_min": float(np.min(pressure)),
        "guide_field_code": guide_x,
        "modes": [
            {
                "kmode": mode,
                "epsilon_amplitude": float(component_metadata["epsilon"]),
                "k_comoving_mpc_inv": float(component_metadata["k_comoving_mpc_inv"]),
                "mode_state": component_metadata["mode_state"],
            }
            for mode, _, component_metadata in components
        ],
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--modes", type=parse_modes, default=(1, 2, 4))
    parser.add_argument("--spectral-epsilon", type=float, default=-0.1)
    parser.add_argument("--target-gas-delta-rms", type=float, default=1.0e-2)
    parser.add_argument("--zstart", type=float, default=1.0e6)
    parser.add_argument("--zhandoff", type=float, default=3200.0)
    parser.add_argument("--box-ckpc-h", type=float, default=1.0)
    parser.add_argument("--hubble", type=float, default=0.674)
    parser.add_argument("--field-pg", type=float, default=315.0)
    parser.add_argument("--theta", type=float, default=math.pi / 4.0)
    parser.add_argument("--velocity-unit", type=float, default=8.0e5)
    parser.add_argument("--max-dlna", type=float, default=1.0e-4)
    parser.add_argument("--response-table", type=Path, required=True)
    parser.add_argument("--nonlocal-scale", type=float, default=1.0)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if max(args.modes) >= args.resolution // 2:
        parser.error("all modes must lie below Nyquist")

    exponent = 0.5 * (args.spectral_epsilon - 3.0)
    trial = {
        mode: 0.01 * (mode / min(args.modes)) ** exponent for mode in args.modes
    }
    trial_components = build_components(args, trial, "trial")
    trial_rms = composite_density_rms(trial_components, args.resolution)
    if trial_rms <= 0.0 or not math.isfinite(trial_rms):
        raise ValueError("trial multimode density response is invalid")
    scale = args.target_gas_delta_rms / trial_rms
    amplitudes = {mode: value * scale for mode, value in trial.items()}
    if max(abs(value) for value in amplitudes.values()) > 0.1:
        raise ValueError("required perturbation amplitude exceeds weak-mode limit 0.1")
    components = build_components(args, amplitudes, "scaled")
    metadata = combine(args, components)
    metadata["trial_gas_delta_rms"] = trial_rms
    metadata["amplitude_scale"] = scale
    metadata["spectral_power_law"] = "P_deltaB(k) proportional to k^(spectral_epsilon-3)"
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
