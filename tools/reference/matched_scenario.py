#!/usr/bin/env python3
"""Generate cell-centered matched-mode inputs and an independent P1 trajectory."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import math
from pathlib import Path

import numpy as np

if __package__:
    from . import lmhd, matched_units
    from .build_p1_meanfield_handoff import mode_line
    from .run_reference import write_json
else:
    import lmhd
    import matched_units
    from build_p1_meanfield_handoff import mode_line
    from run_reference import write_json


def generate(output: Path, *, resolution: int = 32, kmode: int = 1,
             box_ckpc_h: float = 1.0, hubble: float = 0.674,
             zstart: float = 3200.0, zend: float = 3100.0,
             max_dlna: float = 0.001, polarization: str = "magnetosonic") -> dict:
    if resolution < 4 or not 0 < kmode < resolution / 2:
        raise ValueError("mode must be strictly below Nyquist and N >= 4")
    if not zstart > zend >= 0 or max_dlna <= 0:
        raise ValueError("require zstart > zend >= 0 and max_dlna > 0")
    output.mkdir(parents=True, exist_ok=True)
    cosmology = lmhd.Cosmology(h=hubble)
    mode = lmhd.ReferenceMode(matched_units.comoving_k_mpc(kmode, box_ckpc_h, hubble),
                              math.pi / 4, 30e-12, polarization=polarization)
    result = lmhd.evolve_photon_p1(
        mode, cosmology, zstart, zend, np.linspace(zstart, zend, 9),
        max_dln_a=max_dlna, include_gravity=False, include_dark_matter=False,
        include_photon_stress=True, include_helium_electrons=False)
    payload = {
        "schema": 1, "status": "REFERENCE_ONLY", "physics": "linear",
        "scope": "Prescribed mean-field linear test; no production solver was run",
        "cosmology": asdict(cosmology), "reference_mode": asdict(mode),
        "resolution": resolution, "kmode": kmode, "box_ckpc_h": box_ckpc_h,
        "k_comoving_mpc_inv": mode.k_mpc, "hubble": hubble,
        "theta_rad": mode.theta, "field_pG_comoving": mode.b0_gauss * 1e12,
        "epsilon": mode.perturbation_fraction, "polarization": polarization,
        "zstart": zstart, "zend": zend, "max_dln_a": max_dlna,
        "fourier_convention": "f(x) = Re[A exp(2 pi i m (j+1/2)/N)]",
        "geometry": "wavevector along reference z, guide field in y-z plane",
        "units": {"B": "comoving gauss", "velocity": "physical peculiar cm/s",
                  "coordinate": "box fraction", "k": "comoving Mpc^-1 (no h)"},
    }
    arrays = {name: mode_line(complex(values[0]), resolution, kmode)
              for name, values in result.fields.items()
              if name not in ("b_mode_over_initial",)}
    arrays["coordinate"] = (np.arange(resolution) + 0.5) / resolution
    arrays["density_over_mean"] = 1 + arrays["delta_b"]
    arrays["B_x_comoving_gauss"] = mode.b0_gauss * arrays["b_x_over_B0"]
    arrays["B_y_comoving_gauss"] = mode.b0_gauss * (math.sin(mode.theta) + arrays["b_y_over_B0"])
    arrays["B_z_comoving_gauss"] = np.full(resolution, mode.b0_gauss * math.cos(mode.theta))
    np.savez_compressed(output / "initial_fields.npz", **arrays)
    with (output / "reference_modes.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("field", "z", "real", "imag"), delimiter="\t")
        writer.writeheader()
        for index, z in enumerate(result.redshift):
            for name, values in result.fields.items():
                value = complex(values[index])
                writer.writerow(dict(field=name, z=float(z), real=value.real, imag=value.imag))
    write_json(output / "scenario.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--kmode", type=int, default=1)
    parser.add_argument("--box-ckpc-h", type=float, default=1)
    parser.add_argument("--hubble", type=float, default=0.674)
    parser.add_argument("--zstart", type=float, default=3200)
    parser.add_argument("--zend", type=float, default=3100)
    parser.add_argument("--max-dlna", type=float, default=0.001)
    parser.add_argument("--polarization", choices=("magnetosonic", "alfven"), default="magnetosonic")
    args = parser.parse_args()
    generate(**vars(args))
    print(f"Matched-mode input and oracle: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
