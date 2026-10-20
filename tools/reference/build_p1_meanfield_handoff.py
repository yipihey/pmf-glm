#!/usr/bin/env python3
"""Build a complete Vespa handoff from the independent photon-P1 oracle.

This generator is deliberately restricted to the controlled mean-field linear
problem.  It is not a Batchelor-field pre-evolver: the latter has a quadratic
Lorentz-force convolution and needs a separately certified magnetic damping
model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Mapping

import numpy as np

if __package__:
    from . import lmhd, lya_rt, matched_units as runner
else:
    import lmhd
    import lya_rt
    import matched_units as runner


GAMMA_GAS = 5.0 / 3.0


def complex_json(value: complex) -> Mapping[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}


def mode_line(amplitude: complex, n: int, kmode: int) -> np.ndarray:
    phase = 2.0 * math.pi * kmode * (np.arange(n) + 0.5) / n
    return np.real(complex(amplitude) * np.exp(1j * phase))


def mode_field(amplitude: complex, n: int, kmode: int) -> np.ndarray:
    line = mode_line(amplitude, n, kmode)
    return np.broadcast_to(line[:, None, None], (n, n, n))


def particle_supercomoving_velocity(
    peculiar_velocity_cm_s: complex,
    redshift: float,
    hubble: float,
    box_ckpc_h: float,
) -> complex:
    """Convert physical peculiar velocity to Vespa's particle ``dx/dtau``.

    The PM particle integrator uses unit-box coordinates and supercomoving time
    ``d tau = H0 dt / a^2``.  Therefore ``dx/dtau = a v_pec / (H0 L_com)``.
    This is distinct from the evolving MHD velocity normalization.
    """

    if redshift <= -1.0 or hubble <= 0.0 or box_ckpc_h <= 0.0:
        raise ValueError("particle velocity conversion requires physical inputs")
    scale_factor = 1.0 / (1.0 + redshift)
    h0_s = 100.0 * hubble * 1.0e5 / lmhd.MPC_CM
    box_comoving_cm = runner.physical_box_cm(box_ckpc_h, hubble)
    return scale_factor * peculiar_velocity_cm_s / (h0_s * box_comoving_cm)


def helium_saha_line(
    cosmology: lmhd.Cosmology,
    redshift: float,
    hydrogen_xe: np.ndarray,
    density_factor: np.ndarray,
    temperature: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return total electrons, HeII/H, and HeIII/H on one mode line."""

    f_he = cosmology.helium_per_hydrogen
    n_h = cosmology.hydrogen_density(redshift) * density_factor
    t = np.maximum(temperature, 1.0)
    quantum = (
        2.0 * math.pi * lmhd.MELECTRON * lmhd.KBOLTZ * t
        / lmhd.HPLANCK**2
    ) ** 1.5
    saha1 = 4.0 * quantum * np.exp(-lmhd.CHI_HEI_OVER_K / t)
    saha2 = quantum * np.exp(-lmhd.CHI_HEII_OVER_K / t)
    electrons = np.maximum(hydrogen_xe, 0.0) + 2.0 * f_he
    for _ in range(32):
        ne = np.maximum(electrons * n_h, np.finfo(float).tiny)
        ratio1 = saha1 / ne
        ratio2 = saha2 / ne
        denominator = 1.0 + ratio1 + ratio1 * ratio2
        heii = f_he * ratio1 / denominator
        heiii = f_he * ratio1 * ratio2 / denominator
        updated = hydrogen_xe + heii + 2.0 * heiii
        if np.max(np.abs(updated - electrons)) <= 2.0e-14:
            electrons = updated
            break
        electrons = updated
    return electrons, heii, heiii


def atomic_array(path: Path, array: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    np.asarray(array).ravel(order="F").tofile(temporary)
    temporary.replace(path)


def write_arrays(output: Path, arrays: Mapping[str, np.ndarray]) -> None:
    rows = []
    for name, value in arrays.items():
        if np.iscomplexobj(value):
            cast = np.asarray(value, dtype=np.complex64)
            suffix = "c32"
            element = "ComplexF32"
        else:
            cast = np.asarray(value, dtype=np.float32)
            suffix = "f32"
            element = "Float32"
        filename = f"{name}.{suffix}"
        atomic_array(output / filename, cast)
        rows.append(
            {
                "name": name,
                "file": filename,
                "eltype": element,
                "dims": "x".join(str(value) for value in cast.shape),
                "bytes": cast.nbytes,
            }
        )
    with (output / "arrays.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def hydrogen_storage_field(
    storage: str,
    hydrogen_xe: np.ndarray,
    density_factor: np.ndarray,
    hydrogen_mass_fraction: float,
    reference: float,
) -> np.ndarray:
    """Encode the hydrogen ionization state for a Vespa handoff.

    Centered storage is a fraction residual, not a mass density.  Keeping that
    residual near zero is required to retain recombination perturbations when
    the homogeneous ionized fraction rounds to one in Float32.
    """

    if storage == "centered":
        return np.asarray(hydrogen_xe, dtype=float) - float(reference)
    if storage == "ionized":
        return (
            float(hydrogen_mass_fraction)
            * np.asarray(density_factor, dtype=float)
            * np.asarray(hydrogen_xe, dtype=float)
        )
    raise ValueError(f"unsupported hydrogen storage {storage!r}")


def source_fingerprint(response_table: Path | None) -> str:
    digest = hashlib.sha256()
    paths = [Path(__file__).resolve(), Path(lmhd.__file__).resolve()]
    if response_table is not None:
        paths.extend((Path(lya_rt.__file__).resolve(), response_table.resolve()))
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_handoff(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    if not args.zstart > args.zhandoff > 0.0:
        raise ValueError("require zstart > zhandoff > 0")
    if args.resolution < 4 or not 0 < args.kmode < args.resolution // 2:
        raise ValueError("kmode must be resolved below Nyquist")

    omega_r = 4.15e-5 / args.hubble**2
    cosmology = lmhd.Cosmology(
        h=args.hubble,
        omega_m=runner.OMEGA_M,
        omega_b=runner.OMEGA_B,
        omega_lambda=1.0 - runner.OMEGA_M - omega_r,
        omega_r=omega_r,
        hydrogen_mass_fraction=runner.HYDROGEN_MASS_FRACTION,
    )
    ionization = lmhd.PeeblesIonization(
        cosmology, zmax=3600.0, zmin=min(400.0, args.zhandoff),
        samples=12000, recfast_v2=True,
    )
    temperature_history = lmhd.ComptonTemperature(
        cosmology, ionization, zmax=3600.0,
        zmin=min(400.0, args.zhandoff), samples=12000,
    )
    table = (
        lya_rt.ResponseTable.load(args.response_table)
        if args.response_table is not None else None
    )
    k_mpc = runner.comoving_k_mpc(
        args.kmode, args.box_ckpc_h, args.hubble
    )

    def recombination_rates(redshift: float) -> np.ndarray:
        local = lmhd.recfast_v2_local_response(
            cosmology, redshift, ionization(redshift),
            temperature_history(redshift),
        )
        if table is None or not table.redshift[0] <= redshift <= table.redshift[-1]:
            return local
        differential = table.rates(redshift, k_mpc, differential_only=True)
        differential[2] = table.rates(redshift, k_mpc)[2]
        return local + args.nonlocal_scale * differential

    mode = lmhd.ReferenceMode(
        k_mpc=k_mpc,
        theta=args.theta,
        b0_gauss=args.field_pg * 1.0e-12,
        perturbation_fraction=args.epsilon,
        polarization="magnetosonic",
    )
    result = lmhd.evolve_photon_p1(
        mode, cosmology, args.zstart, args.zhandoff, [args.zhandoff],
        ionization, max_dln_a=args.max_dlna, include_gravity=True,
        include_dark_matter=True, include_photon_stress=True,
        include_helium_electrons=True,
        recombination_rates=recombination_rates,
        baryon_temperature=temperature_history,
    )
    index = int(np.argmin(np.abs(result.redshift - args.zhandoff)))
    state = {
        name: args.amplitude_sign * complex(values[index])
        for name, values in result.fields.items()
    }
    n = args.resolution
    kmode = args.kmode
    xh = cosmology.hydrogen_mass_fraction
    a_reference = 1.0 / (1.0 + args.zstart)
    a_handoff = 1.0 / (1.0 + args.zhandoff)
    velocity_unit = args.velocity_unit * math.sqrt(a_reference / a_handoff)

    rho_line = 1.0 + mode_line(state["delta_b"], n, kmode)
    temperature_line = (
        temperature_history(args.zhandoff)
        + mode_line(state["delta_temperature"], n, kmode)
    )
    hydrogen_xe_line = (
        ionization(args.zhandoff) + mode_line(state["delta_xe"], n, kmode)
    )
    total_xe_line, heii_line, heiii_line = helium_saha_line(
        cosmology, args.zhandoff, hydrogen_xe_line, rho_line,
        temperature_line,
    )
    particles_per_mass = xh * (1.0 + total_xe_line) + (1.0 - xh) / 4.0
    pressure_line = (
        rho_line * particles_per_mass * lmhd.KBOLTZ * temperature_line
        / (lmhd.MHYDROGEN * velocity_unit**2)
    )
    if np.min(rho_line) <= 0.0 or np.min(pressure_line) <= 0.0:
        raise ValueError("oracle handoff violates density or pressure positivity")

    rho = np.broadcast_to(rho_line[:, None, None], (n, n, n))
    pressure = np.broadcast_to(pressure_line[:, None, None], (n, n, n))
    vx = mode_field(state["v_b_z"] / velocity_unit, n, kmode)
    vy = mode_field(state["v_b_y"] / velocity_unit, n, kmode)
    vz = mode_field(state["v_b_x"] / velocity_unit, n, kmode)

    rho_reference = cosmology.baryon_density(args.zstart)
    physical_b_reference = mode.b0_gauss / a_reference**2
    bunit_reference = args.velocity_unit * math.sqrt(4.0 * math.pi * rho_reference)
    guide_code = physical_b_reference / bunit_reference
    bx = guide_code * math.cos(args.theta) + mode_field(
        guide_code * state["b_x_over_B0"], n, kmode
    )
    by = guide_code * math.sin(args.theta) + mode_field(
        guide_code * state["b_y_over_B0"], n, kmode
    )
    bz = np.zeros((n, n, n), dtype=float)
    kinetic = 0.5 * rho * (vx * vx + vy * vy + vz * vz)
    magnetic = 0.5 * (bx * bx + by * by + bz * bz)
    energy = pressure / (GAMMA_GAS - 1.0) + kinetic + magnetic

    hxe = np.broadcast_to(hydrogen_xe_line[:, None, None], (n, n, n))
    heii = np.broadcast_to(heii_line[:, None, None], (n, n, n))
    specific_eint = pressure / ((GAMMA_GAS - 1.0) * rho)

    # Lattice-displacement particles: delta_dm = -div(psi), with box
    # coordinates in [0,1).  The sole mode is longitudinal along runtime x.
    displacement_amplitude = 1j * state["delta_dm"] / (2.0 * math.pi * kmode)
    particle_dx = mode_field(displacement_amplitude, n, kmode)
    particle_velocity_amplitude = particle_supercomoving_velocity(
        state["v_dm_z"], args.zhandoff, args.hubble, args.box_ckpc_h
    )
    particle_vx = mode_field(particle_velocity_amplitude, n, kmode)
    zeros = np.zeros((n, n, n), dtype=float)

    photon_real = (
        mode_field(state["delta_gamma"], n, kmode),
        mode_field(state["v_gamma_z"] / velocity_unit, n, kmode),
        mode_field(state["v_gamma_y"] / velocity_unit, n, kmode),
        mode_field(state["v_gamma_x"] / velocity_unit, n, kmode),
    )
    photon_hat = np.stack(
        [np.fft.rfftn(value, axes=(2, 1, 0)) for value in photon_real],
        axis=-1,
    )

    hydrogen_reference = float(ionization(args.zhandoff))
    hydrogen_payload = hydrogen_storage_field(
        args.hydrogen_storage, hxe, rho, xh,
        hydrogen_reference,
    )
    arrays = {
        "mhd_U_1": rho,
        "mhd_U_2": rho * vx,
        "mhd_U_3": rho * vy,
        "mhd_U_4": rho * vz,
        "mhd_U_5": energy,
        "mhd_U_6": bx,
        "mhd_U_7": by,
        "mhd_U_8": bz,
        "mhd_U_9": zeros,
        "particle_px": particle_dx,
        "particle_py": zeros,
        "particle_pz": zeros,
        "particle_vx": particle_vx,
        "particle_vy": zeros,
        "particle_vz": zeros,
        "chem_eint": specific_eint,
        "chem_HII": hydrogen_payload,
        "chem_H2I": zeros,
        "chem_HeII": 4.0 * xh * rho * heii,
        "photon_hat_batch": photon_hat,
        "gas_delta": rho - 1.0,
    }
    payload_bytes = sum(
        np.asarray(value, dtype=(np.complex64 if np.iscomplexobj(value) else np.float32)).nbytes
        for value in arrays.values()
    )
    args.output.mkdir(parents=True, exist_ok=True)
    write_arrays(args.output, arrays)
    (args.output / "chem_hydrogen_storage.txt").write_text(
        args.hydrogen_storage + "\n"
    )
    if args.hydrogen_storage == "centered":
        (args.output / "chem_hydrogen_reference.txt").write_text(
            f"{hydrogen_reference:.17g}\n"
        )
    (args.output / "runtime.tsv").write_text(
        "cycle\tmeasured\ta\tz\telapsed_time\n"
        f"0\t0\t{a_handoff:.17g}\t{args.zhandoff:.17g}\t0\n"
    )
    metadata = {
        "schema": 1,
        "status": "PASS",
        "scope": "linear mean-field photon-P1 handoff; not Batchelor certified",
        "reference_source_fingerprint_sha256": source_fingerprint(
            args.response_table
        ),
        "resolution": n,
        "zstart": args.zstart,
        "zhandoff": args.zhandoff,
        "box_ckpc_h": args.box_ckpc_h,
        "k_comoving_mpc_inv": k_mpc,
        "kmode": kmode,
        "theta_rad": args.theta,
        "field_pG_comoving": args.field_pg,
        "epsilon": args.epsilon,
        "amplitude_sign": args.amplitude_sign,
        "signed_epsilon": args.amplitude_sign * args.epsilon,
        "max_dln_a": args.max_dlna,
        "generation_wall_seconds": time.perf_counter() - started,
        "payload_bytes": payload_bytes,
        "payload_gib": payload_bytes / 2.0**30,
        "velocity_unit_cm_s_at_start": args.velocity_unit,
        "velocity_unit_cm_s_at_handoff": velocity_unit,
        "chem_hydrogen_storage": args.hydrogen_storage,
        "chem_hydrogen_reference": (
            hydrogen_reference if args.hydrogen_storage == "centered" else None
        ),
        "particle_velocity_convention": "dx/dtau, dtau=H0*dt/a^2",
        "particle_vx_mode_code": complex_json(particle_velocity_amplitude),
        "guide_field_code": guide_code,
        "background": {
            "xHII": ionization(args.zhandoff),
            "temperature_K": temperature_history(args.zhandoff),
            "xe_total_mean": float(np.mean(total_xe_line)),
            "xHeII_mean": float(np.mean(heii_line)),
            "xHeIII_mean": float(np.mean(heiii_line)),
            "pressure_code_mean": float(np.mean(pressure_line)),
        },
        "mode_state": {name: complex_json(value) for name, value in state.items()},
        "integrity": {
            "rho_min": float(np.min(rho_line)),
            "pressure_min": float(np.min(pressure_line)),
            "brms_code": float(np.sqrt(np.mean(bx * bx + by * by + bz * bz))),
            "gas_delta_rms": float(np.sqrt(np.mean((rho_line - 1.0) ** 2))),
            "dm_delta_rms": abs(state["delta_dm"]) / math.sqrt(2.0),
            "chem_HII_min": float(np.min(hydrogen_payload)),
            "chem_HII_max": float(np.max(hydrogen_payload)),
        },
    }
    temporary = args.output / "metadata.json.tmp"
    temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output / "metadata.json")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--zstart", type=float, default=1.0e6)
    parser.add_argument("--zhandoff", type=float, default=3200.0)
    parser.add_argument("--box-ckpc-h", type=float, default=1.0)
    parser.add_argument("--hubble", type=float, default=0.674)
    parser.add_argument("--field-pg", type=float, default=30.0)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument(
        "--amplitude-sign", type=int, choices=(-1, 1), default=1,
        help="multiply every evolved first-order mode before constructing the 3D state",
    )
    parser.add_argument("--theta", type=float, default=math.pi / 4.0)
    parser.add_argument("--kmode", type=int, default=1)
    parser.add_argument("--velocity-unit", type=float, default=8.0e5)
    parser.add_argument("--max-dlna", type=float, default=1.0e-4)
    parser.add_argument("--response-table", type=Path)
    parser.add_argument("--nonlocal-scale", type=float, default=1.0)
    parser.add_argument(
        "--hydrogen-storage", choices=("centered", "ionized"),
        default="centered",
        help="Vespa chemistry representation written into chem_HII.f32",
    )
    args = parser.parse_args()
    metadata = build_handoff(args)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
