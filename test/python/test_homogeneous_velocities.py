import numpy as np
import pytest

from tools.reference.homogeneous_velocities import evolve


def test_unscattered_photon_dipole_does_not_share_matter_hubble_drag():
    initial = np.array([3e5, -2e5, 1e5])
    result = evolve(initial=initial, xe=0)
    assert result == pytest.approx(initial * [601/1201, 1, 601/1201], rel=1e-9)


def test_homogeneous_expanding_exchange_conserves_weighted_momentum():
    result = evolve()
    def inertia(z):
        rb = .17037 * .315 * 1.8788e-29 * .674**2 * (1+z)**3
        eg = 7.5657e-15 * (2.725 * (1+z))**4
        return 3 * rb * 2.99792458e10**2 / (4 * eg)
    # Since R grows as a, R*u_b+u_gamma is constant for the homogeneous
    # physical equations even during expansion and at nonzero scattering.
    assert inertia(600)*result[0]+result[1] == pytest.approx((inertia(1200)+1)*1e5, rel=1e-9)
    assert result[2] == pytest.approx(1e5 * 601/1201, rel=1e-9)
