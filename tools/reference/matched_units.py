"""Unit helpers extracted from run_matched_vespa.py, without solver imports."""

import math

KPC_CM = 3.0856775814913673e21
HYDROGEN_MASS_FRACTION = 0.7546
OMEGA_M = 0.315
OMEGA_B = 0.05366655


def physical_box_cm(box_ckpc_h: float, hubble: float) -> float:
    """Convert a comoving ckpc/h box to comoving centimeters."""
    if box_ckpc_h <= 0 or hubble <= 0:
        raise ValueError("box length and h must be positive")
    return box_ckpc_h * KPC_CM / hubble


def comoving_k_mpc(kmode: int, box_ckpc_h: float, hubble: float) -> float:
    if kmode <= 0:
        raise ValueError("kmode must be positive")
    physical_box_cm(box_ckpc_h, hubble)
    return 2.0 * math.pi * kmode * 1000.0 * hubble / box_ckpc_h
