"""Independent physical-velocity oracle for homogeneous baryons, photons and CDM.

No Julia solver, code-unit clocks, or production photon closure are imported.
Ionization is held fixed to isolate velocity transport, not model recombination.
"""
import argparse
import json

import numpy as np
from scipy.integrate import solve_ivp


def evolve(zstart=1200.0, zend=600.0, initial=(1e5, 1e5, 1e5), xe=1.0):
    """Return physical cm/s in baryon, photon, CDM order with driver default cosmology."""
    if not zstart > zend >= 0 or not np.isfinite(xe) or xe < 0:
        raise ValueError("require zstart>zend>=0 and finite xe>=0")
    if np.shape(initial) != (3,) or not np.isfinite(initial).all():
        raise ValueError("initial velocities must be three finite values")
    h, om, ol = 0.674, 0.315, 0.685
    ob, orad = 0.17037 * om, 4.15e-5 / h**2
    ok = 1 - om - ol - orad
    h0 = 100 * h * 1e5 / 3.0857e24

    def rhs(lna, u):
        a = np.exp(lna)
        hubble = h0 * np.sqrt(om/a**3 + ol + orad/a**4 + ok/a**2)
        rho_b = ob * 1.8788e-29 * h**2 / a**3
        photon_energy = 7.5657e-15 * (2.725/a)**4
        gamma = (4/3) * photon_energy * xe * 0.76 * 6.6524e-25 / (2.99792458e10 * 1.67262171e-24)
        inertia = 3 * rho_b * 2.99792458e10**2 / (4 * photon_energy)
        baryon, photon, cdm = u
        exchange = gamma / hubble * (baryon - photon)
        return [-baryon - exchange, inertia * exchange, -cdm]

    result = solve_ivp(rhs, (np.log(1/(1+zstart)), np.log(1/(1+zend))),
                       initial, method="Radau", rtol=2e-11, atol=1e-7)
    if not result.success:
        raise RuntimeError(result.message)
    return result.y[:, -1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zstart", type=float, default=1200)
    parser.add_argument("--zend", type=float, default=600)
    parser.add_argument("--xe", type=float, default=1)
    args = parser.parse_args()
    values = evolve(args.zstart, args.zend, xe=args.xe)
    print(json.dumps(dict(zip(("baryon_cm_s", "photon_cm_s", "cdm_cm_s"), values)), indent=2))


if __name__ == "__main__":
    main()
