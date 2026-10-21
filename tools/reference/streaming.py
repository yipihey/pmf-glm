"""Controlled matter-era moving-background growth oracle (REFERENCE_ONLY).

Tseliakhovich & Hirata 2010, arXiv:1005.2416, Sec. II.3, transformed to
DM rest frame, exp(+ik.x), vbc=vb-vc. This deliberately uses Einstein-de Sitter
expansion and prescribed adiabatic sound speed, without photons, PMFs or
chemistry. It is not a cosmological transfer-function or halo prediction.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


def growth(*, k_mpc=(100., 0., 0.), vbc_kms=(0., 0., 0.), zstart=200., zend=20.,
           baryon_fraction=0.16, h=0.674, sound_speed_kms=0., points=64):
    """Return complex delta_c, theta_c/H, delta_b, theta_b/H versus a.

    Initial perturbations are unit growing modes for both fluids. Velocities
    and sound speed are physical at zstart and decay as a^-1. k is comoving
    Mpc^-1 without h. Amplitudes can be rescaled to the linear regime.
    """
    k, v = np.asarray(k_mpc, dtype=float), np.asarray(vbc_kms, dtype=float)
    if k.shape != (3,) or v.shape != (3,) or not np.isfinite([k, v]).all():
        raise ValueError("k and vbc require three finite components")
    if not np.isfinite([zstart, zend, baryon_fraction, h, sound_speed_kms]).all():
        raise ValueError("require finite model parameters")
    if not (zstart > zend >= 0 and 0 < baryon_fraction < 1 and h > 0 and sound_speed_kms >= 0):
        raise ValueError("invalid expansion, baryon fraction, or sound speed")
    if not isinstance(points, int) or points < 2:
        raise ValueError("points must be an integer >= 2")
    a0, a1 = 1/(1+zstart), 1/(1+zend)
    loga = np.linspace(np.log(a0), np.log(a1), points)

    def rhs(log_a, y):
        a = np.exp(log_a)
        H = 100*h*a**(-1.5)  # km/s/Mpc, exactly matter dominated
        adv = 1j*np.dot(k, v)*(a0/a)/(a*H)
        pressure = np.dot(k, k)*(sound_speed_kms*a0/a)**2/(a*H)**2
        dc, uc, db, ub = y
        grav = 1.5*((1-baryon_fraction)*dc+baryon_fraction*db)
        return [-uc, -0.5*uc-grav, -adv*db-ub, -(0.5+adv)*ub-grav+pressure*db]

    solution = solve_ivp(rhs, (loga[0], loga[-1]), np.array([1, -1, 1, -1], complex),
                         t_eval=loga, rtol=1e-10, atol=1e-12, method="DOP853")
    if not solution.success:
        raise RuntimeError(solution.message)
    return np.exp(loga), solution.y
