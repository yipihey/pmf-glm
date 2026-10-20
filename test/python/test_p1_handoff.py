import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.reference import build_p1_meanfield_handoff as handoff, lmhd


class P1HandoffTests(unittest.TestCase):
    def test_cell_centered_mode_recovers_complex_amplitude(self) -> None:
        n = 32
        kmode = 3
        amplitude = 0.02 - 0.03j
        line = handoff.mode_line(amplitude, n, kmode)
        phase = 2.0 * math.pi * kmode * (np.arange(n) + 0.5) / n
        recovered = 2.0 * np.mean(line * np.exp(-1j * phase))
        self.assertAlmostEqual(recovered.real, amplitude.real, places=14)
        self.assertAlmostEqual(recovered.imag, amplitude.imag, places=14)

    def test_dm_displacement_satisfies_linear_continuity(self) -> None:
        n = 128
        kmode = 2
        delta = -1.2e-4 + 3.0e-5j
        displacement = handoff.mode_line(
            1j * delta / (2.0 * math.pi * kmode), n, kmode
        )
        phase = 2.0 * math.pi * kmode * (np.arange(n) + 0.5) / n
        recovered_psi = 2.0 * np.mean(displacement * np.exp(-1j * phase))
        recovered_delta = -1j * 2.0 * math.pi * kmode * recovered_psi
        self.assertAlmostEqual(recovered_delta.real, delta.real, places=14)
        self.assertAlmostEqual(recovered_delta.imag, delta.imag, places=14)

    def test_dm_velocity_uses_supercomoving_particle_units(self) -> None:
        redshift = 3200.0
        hubble = 0.674
        box_ckpc_h = 1.0
        peculiar_velocity = -0.3364j
        converted = handoff.particle_supercomoving_velocity(
            peculiar_velocity, redshift, hubble, box_ckpc_h
        )
        a = 1.0 / (1.0 + redshift)
        h0 = 100.0 * hubble * 1.0e5 / lmhd.MPC_CM
        box_cm = handoff.runner.physical_box_cm(box_ckpc_h, hubble)
        expected = a * peculiar_velocity / (h0 * box_cm)
        self.assertAlmostEqual(converted.real, expected.real, places=18)
        self.assertAlmostEqual(converted.imag, expected.imag, places=18)
        self.assertAlmostEqual(abs(converted), 1.051e-8, delta=2.0e-11)

    def test_dm_velocity_and_displacement_obey_supercomoving_continuity(self) -> None:
        cosmology = lmhd.Cosmology(h=0.674)
        redshift = 3200.0
        kmode = 1
        box_ckpc_h = 1.0
        delta_prime_lna = 2.5e-6 - 1.0e-6j
        # d(delta)/dln(a) = -i k_com vpec / (a H).
        k_com = (
            2.0 * math.pi * kmode * 1000.0 * cosmology.h / lmhd.MPC_CM
        )
        a = 1.0 / (1.0 + redshift)
        vpec = 1j * a * cosmology.hubble(redshift) * delta_prime_lna / k_com
        particle_v = handoff.particle_supercomoving_velocity(
            vpec, redshift, cosmology.h, box_ckpc_h
        )
        recovered = -1j * 2.0 * math.pi * kmode * particle_v / (
            cosmology.hubble(redshift) * a * a / cosmology.h0
        )
        self.assertAlmostEqual(recovered.real, delta_prime_lna.real, places=14)
        self.assertAlmostEqual(recovered.imag, delta_prime_lna.imag, places=14)

    def test_helium_saha_line_has_physical_limits(self) -> None:
        cosmology = lmhd.Cosmology()
        hydrogen = np.ones(8)
        density = np.ones(8)
        temperature = np.full(8, cosmology.tcmb0 * (1.0 + 1.0e6))
        electrons, heii, heiii = handoff.helium_saha_line(
            cosmology, 1.0e6, hydrogen, density, temperature
        )
        expected = 1.0 + 2.0 * cosmology.helium_per_hydrogen
        np.testing.assert_allclose(electrons, expected, rtol=0.0, atol=1.0e-12)
        self.assertTrue(np.all(heii >= 0.0))
        self.assertTrue(np.all(heiii >= 0.0))

    def test_raw_arrays_roundtrip_in_julia_column_major_order(self) -> None:
        original = np.arange(24, dtype=np.float32).reshape((2, 3, 4), order="F")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "array.f32"
            handoff.atomic_array(path, original)
            restored = np.fromfile(path, dtype=np.float32).reshape(
                original.shape, order="F"
            )
        np.testing.assert_array_equal(restored, original)

    def test_centered_hydrogen_storage_preserves_sub_ulp_residual(self) -> None:
        reference = 1.0 - 2.5e-10
        hydrogen = reference + np.asarray([-6.0e-14, 6.0e-14])
        density = np.asarray([0.99, 1.01])
        encoded = handoff.hydrogen_storage_field(
            "centered", hydrogen, density, 0.7546, reference
        ).astype(np.float32)
        self.assertTrue(np.all(encoded != 0.0))
        np.testing.assert_allclose(
            encoded.astype(np.float64), hydrogen - reference,
            rtol=2.0e-5, atol=0.0,
        )

    def test_ionized_hydrogen_storage_remains_mass_density(self) -> None:
        hydrogen = np.asarray([0.2, 0.8])
        density = np.asarray([0.5, 1.5])
        encoded = handoff.hydrogen_storage_field(
            "ionized", hydrogen, density, 0.75, 0.5
        )
        np.testing.assert_allclose(encoded, 0.75 * density * hydrogen)

    def test_unknown_hydrogen_storage_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            handoff.hydrogen_storage_field(
                "neutral", np.ones(2), np.ones(2), 0.75, 1.0
            )


if __name__ == "__main__":
    unittest.main()
