#!/usr/bin/env python3

import math
from pathlib import Path
import unittest

import numpy as np

from tools.reference import lmhd, matched_units as run_matched_vespa


class BackgroundTest(unittest.TestCase):
    def setUp(self):
        self.cosmology = lmhd.Cosmology()

    def test_hubble_and_density_are_physical(self):
        self.assertGreater(self.cosmology.hubble(1000), self.cosmology.hubble(0))
        ratio = self.cosmology.baryon_density(1999) / self.cosmology.baryon_density(999)
        self.assertAlmostEqual(ratio, 8.0, places=12)

    def test_q_wavenumber_is_inverse_mean_free_path(self):
        ionization = lmhd.FullyIonized()
        z = 1200.0
        k = lmhd.wavenumber_for_q(self.cosmology, z, 3.0, ionization)
        coeff = lmhd.background_coefficients(
            self.cosmology, z, ionization, 1.0e-12
        )
        self.assertAlmostEqual(
            k / lmhd.MPC_CM * coeff["lambda_comoving"], 3.0, places=12
        )

    def test_tcr_eta_prime_matches_paper_definition(self):
        coeff = lmhd.background_coefficients(
            self.cosmology, 1200.0, lmhd.FullyIonized(), 1.0e-12
        )
        expected = lmhd.CLIGHT * coeff["lambda_physical"] / 15.0
        self.assertAlmostEqual(coeff["eta_prime"], expected, places=12)

    def test_paper_start_is_horizon_entry_capped_at_one_mev(self):
        neutrino = lmhd.neutrino_decoupling_redshift(self.cosmology)
        small_k = lmhd.sv_paper_initial_redshift(self.cosmology, 0.03)
        large_k = lmhd.sv_paper_initial_redshift(self.cosmology, 2.0e5)
        self.assertLess(small_k, neutrino)
        self.assertAlmostEqual(large_k, neutrino, places=6)
        horizon = lmhd.horizon_entry_redshift(self.cosmology, 0.03)
        a = 1.0 / (1.0 + horizon)
        self.assertAlmostEqual(
            lmhd.CLIGHT * 0.03 / lmhd.MPC_CM,
            a * self.cosmology.hubble(horizon),
            delta=1.0e-12 * a * self.cosmology.hubble(horizon),
        )

    def test_ckpc_h_box_conversion_and_fundamental_mode(self):
        h = 0.674
        box = 1.0
        self.assertAlmostEqual(
            run_matched_vespa.physical_box_cm(box, h),
            run_matched_vespa.KPC_CM / h,
            places=6,
        )
        self.assertAlmostEqual(
            run_matched_vespa.comoving_k_mpc(1, box, h),
            2.0 * math.pi * 1000.0 * h,
            places=12,
        )

    def test_saha_limits(self):
        self.assertGreater(lmhd.saha_ionization(self.cosmology, 3000), 0.999)
        self.assertLess(lmhd.saha_ionization(self.cosmology, 800), 1.0e-3)

    def test_helium_saha_electrons_recombine(self):
        neutral = lmhd.helium_saha_electrons(self.cosmology, 1200.0, 0.3)
        fully_ionized = lmhd.helium_saha_electrons(
            self.cosmology, 1.0e6, 1.0
        )
        self.assertAlmostEqual(neutral, 0.3, places=10)
        self.assertAlmostEqual(
            fully_ionized,
            1.0 + 2.0 * self.cosmology.helium_per_hydrogen,
            places=10,
        )

    def test_helium_electrons_enter_high_redshift_eos(self):
        z = 3200.0
        hydrogen_only = lmhd.background_coefficients(
            self.cosmology, z, lmhd.FullyIonized(), 30.0e-12,
            include_helium_electrons=False,
        )
        with_helium = lmhd.background_coefficients(
            self.cosmology, z, lmhd.FullyIonized(), 30.0e-12,
            include_helium_electrons=True,
        )
        expected_particles = (
            1.0 + self.cosmology.helium_per_hydrogen
            + with_helium["electrons_per_h"]
        )
        self.assertAlmostEqual(
            with_helium["particles_per_h"], expected_particles, places=13
        )
        self.assertGreater(
            with_helium["cs2_baryon"], hydrogen_only["cs2_baryon"]
        )

    def test_custom_background_electron_history_overrides_saha_helium(self):
        total_electrons = 1.037
        coeff = lmhd.background_coefficients(
            self.cosmology,
            1900.0,
            lmhd.FullyIonized(),
            30.0e-12,
            include_helium_electrons=True,
            background_electron_fraction=lambda _: total_electrons,
        )
        self.assertAlmostEqual(coeff["electrons_per_h"], total_electrons)
        self.assertAlmostEqual(
            coeff["particles_per_h"],
            1.0 + self.cosmology.helium_per_hydrogen + total_electrons,
        )

    def test_peebles_history_is_finite_and_recombines(self):
        history = lmhd.PeeblesIonization(
            self.cosmology, zmax=3500, zmin=500, samples=1200
        )
        values = np.asarray([history(z) for z in (3000, 1600, 1200, 900, 500)])
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertTrue(np.all(np.diff(values) <= 1.0e-10))
        self.assertGreater(values[0], 0.99)
        self.assertLess(values[-1], 0.02)

    def test_tabulated_thermochemistry_accepts_csv_and_tsv(self):
        import tempfile

        for separator, header in (
            (",", "z,xe,Tb_K"),
            ("\t", "redshift\txHII\tTb_K"),
            (" ", "redshift xHII Tb_K"),
        ):
            with self.subTest(separator=repr(separator)), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "history"
                path.write_text(
                    "# fixture\n" + header + "\n" +
                    separator.join(("900", "0.1", "2400")) + "\n" +
                    separator.join(("1100", "0.3", "3000")) + "\n"
                )
                history = lmhd.TabulatedThermochemistry(path)
                self.assertAlmostEqual(history.ionization(1000), 0.2)
                self.assertAlmostEqual(history.temperature(1000), 2700.0)
                self.assertAlmostEqual(history.clamped_ionization(800), 0.1)
                self.assertAlmostEqual(history.clamped_ionization(1200), 0.3)

    def test_temperature_response_includes_particle_number_and_cooling(self):
        z = 1100.0
        xe = 0.14
        temperature = self.cosmology.tcmb0 * (1.0 + z) * 0.999
        recombination = lmhd.recfast_v2_local_response(
            self.cosmology, z, xe, temperature
        )
        coarse = lmhd.recfast_v2_temperature_response(
            self.cosmology, z, xe, temperature, recombination,
            relative_step=2.0e-5,
        )
        fine = lmhd.recfast_v2_temperature_response(
            self.cosmology, z, xe, temperature, recombination,
            relative_step=1.0e-5,
        )
        self.assertTrue(np.all(np.isfinite(fine)))
        self.assertTrue(np.allclose(coarse, fine, rtol=2.0e-7, atol=1.0e-12))
        self.assertLess(fine[3], 0.0)
        self.assertLess(fine[2], 0.0)

    def test_recfast_v2_ode_matches_external_fixture(self):
        h = 0.71
        fixture_cosmology = lmhd.Cosmology(
            h=h,
            omega_m=0.27,
            omega_b=0.044,
            omega_lambda=0.73,
            omega_r=4.15e-5 / h**2,
            hydrogen_mass_fraction=0.76,
            tcmb0=2.725,
        )
        history = lmhd.PeeblesIonization(
            fixture_cosmology,
            zmax=5000,
            zmin=600,
            samples=8000,
            recfast_v2=True,
        )
        fixture = Path(__file__).parent / "fixtures" / "recfast_v2_xe.csv"
        rows = np.genfromtxt(fixture, delimiter=",", comments="#")
        for redshift in (700.0, 800.0, 900.0, 1000.0, 1100.0, 1400.0, 1600.0):
            reference = np.interp(redshift, rows[:, 0], rows[:, 1])
            self.assertLess(abs(history(redshift) / reference - 1.0), 0.003)

    def test_compton_temperature_locks_then_decouples(self):
        ionization = lmhd.PeeblesIonization(
            self.cosmology, zmax=3500.0, zmin=100.0, samples=2000,
            recfast_v2=True,
        )
        temperature = lmhd.ComptonTemperature(
            self.cosmology, ionization, zmax=3500.0, zmin=100.0,
            samples=2000,
        )
        self.assertLess(
            abs(temperature(1200.0) /
                (self.cosmology.tcmb0 * 1201.0) - 1.0),
            0.01,
        )
        self.assertLess(temperature(100.0), self.cosmology.tcmb0 * 101.0)
        self.assertLess(lmhd.case_b_recombination_log_slope(3000.0), -0.6)

    def test_recfast_local_response_has_physical_signs(self):
        ionization = lmhd.PeeblesIonization(
            self.cosmology, zmax=3500.0, zmin=500.0, samples=2000,
            recfast_v2=True,
        )
        redshift = 1100.0
        temperature = self.cosmology.tcmb0 * (1.0 + redshift)
        response = lmhd.recfast_v2_local_response(
            self.cosmology, redshift, ionization(redshift), temperature
        )
        refined = lmhd.recfast_v2_local_response(
            self.cosmology, redshift, ionization(redshift), temperature,
            relative_step=5.0e-6,
        )
        self.assertLess(response[0], 0.0)
        self.assertLess(response[1], 0.0)
        self.assertEqual(response[2], 0.0)
        self.assertGreater(response[3], 0.0)
        np.testing.assert_allclose(response, refined, rtol=2.0e-7, atol=1.0e-22)


class AngularClosureTest(unittest.TestCase):
    def test_diffusion_limit(self):
        a0, along, transverse = lmhd._angular_transport(1.0e-10)
        self.assertAlmostEqual(a0, 1.0, places=9)
        self.assertAlmostEqual(along, 1.0, places=9)
        self.assertAlmostEqual(transverse, 1.0, places=9)

    def test_free_streaming_response_decays(self):
        small = lmhd._angular_transport(1.0)[2]
        large = lmhd._angular_transport(1.0e8)[2]
        self.assertGreater(small, large)
        self.assertGreater(large, 0.0)


class LinearModeTest(unittest.TestCase):
    def setUp(self):
        self.cosmology = lmhd.Cosmology()
        self.ionization = lmhd.FullyIonized()
        self.k = lmhd.wavenumber_for_q(
            self.cosmology, 1100.0, 0.3, self.ionization
        )

    def mode(self, perturbation=1.0e-4):
        return lmhd.ReferenceMode(
            k_mpc=self.k,
            theta=math.pi / 4,
            b0_gauss=315.0e-12,
            perturbation_fraction=perturbation,
        )

    def test_sv_initial_normalization(self):
        result = lmhd.evolve_sv_tcr_fsr(
            self.mode(), self.cosmology, 1.0e5, 9.0e4, [1.0e5],
            self.ionization, max_dln_a=2.0e-3, include_gravity=False
        )
        self.assertAlmostEqual(result.field("b_mode_over_initial")[0].real, 1.0)
        self.assertAlmostEqual(result.field("delta_b")[0].real, 0.0)

    def test_paper_saha_closure_reduces_pressure_stiffness(self):
        mode = self.mode()
        z = 1100.0
        partial_ionization = lambda redshift: 0.5
        fixed = lmhd._sv_closed_fsr_matrix(
            z, mode, self.cosmology, partial_ionization, "fixed", None,
            include_gravity=False, include_helium_electrons=False,
        )
        saha = lmhd._sv_closed_fsr_matrix(
            z, mode, self.cosmology, partial_ionization, "saha", None,
            include_gravity=False, include_helium_electrons=False,
        )
        self.assertGreater(saha[5, 2], fixed[5, 2])

    def test_paper_three_level_closure_evolves_delta_xe(self):
        mode = self.mode(1.0e-3)
        result = lmhd.evolve_sv_paper_mode(
            mode,
            self.cosmology,
            1100.0,
            [1100.0],
            self.ionization,
            initialization="redshift",
            start_redshift=1200.0,
            closure="three_level",
            recombination_rates=lambda z, k: (-1.0e-12, -2.0e-12, 3.0e-13),
            max_dln_a=2.0e-3,
            include_gravity=False,
        )
        self.assertTrue(np.isfinite(result.field("delta_xe")[-1]))
        self.assertNotEqual(abs(result.field("delta_xe")[-1]), 0.0)
        self.assertEqual(result.metadata["initialization"], "redshift")

    def test_paper_temperature_closure_couples_pressure_and_recombination(self):
        mode = self.mode(1.0e-3)
        temperature = lambda z: self.cosmology.tcmb0 * (1.0 + z)
        result = lmhd.evolve_sv_paper_mode(
            mode,
            self.cosmology,
            1100.0,
            [1200.0, 1100.0],
            self.ionization,
            initialization="redshift",
            start_redshift=1200.0,
            closure="three_level",
            recombination_rates=lambda z, k: (
                -1.0e-12, -2.0e-12, 3.0e-13, 4.0e-12
            ),
            max_dln_a=1.0e-3,
            include_gravity=False,
            baryon_temperature=temperature,
        )
        self.assertEqual(result.field("delta_temperature")[0], 0.0)
        self.assertNotEqual(abs(result.field("delta_temperature")[-1]), 0.0)
        self.assertNotEqual(abs(result.field("delta_xe")[-1]), 0.0)
        self.assertTrue(np.isfinite(result.field("pressure_fraction")[-1]))
        self.assertTrue(result.metadata["include_baryon_temperature"])

    def test_matrix_action_handles_stable_nonnormal_stiff_block(self):
        generator = np.zeros((7, 7))
        generator[0, 1] = 1.0e12
        generator[1, 1] = -1.0e8
        initial = np.zeros(7)
        initial[1] = 1.0
        result = lmhd._advance_matrix(
            initial,
            1000.0,
            math.expm1(math.log1p(1000.0) - 1.0e-3),
            lambda z: generator,
            max_dln_a=1.0e-3,
            method="midpoint2",
        )
        np.testing.assert_allclose(
            result, np.asarray([1.0e4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), atol=1.0e-8
        )

    def test_paper_density_seed_has_zero_initial_magnetic_mode(self):
        mode = self.mode(1.0)
        theta = mode.theta
        state = np.asarray(
            [0.0, -3.0 * math.tan(theta), -3.0, 0.0, 0.0, 0.0]
        )
        start = lmhd.sv_paper_initial_redshift(self.cosmology, mode.k_mpc)
        result = lmhd.evolve_sv_paper_mode(
            mode,
            self.cosmology,
            0.99 * start,
            [start],
            self.ionization,
            initialization="physical",
            initial_state=state,
            include_gravity=False,
        )
        self.assertAlmostEqual(result.field("delta_b")[0].real, 3.0)
        self.assertAlmostEqual(result.field("b_y_over_B0")[0].real, 0.0)
        self.assertTrue(result.metadata["custom_initial_state"])

    def test_public_free_streaming_epoch_satisfies_mean_free_path_crossing(self):
        mode = self.mode()
        zfs = lmhd.free_streaming_redshift(
            mode, self.cosmology, self.ionization, 500.0
        )
        self.assertIsNotNone(zfs)
        coeff = lmhd.background_coefficients(
            self.cosmology, zfs, self.ionization, mode.b0_gauss
        )
        self.assertAlmostEqual(
            mode.k_comoving_cgs * coeff["lambda_comoving"], 1.0, places=10
        )

    def test_p1_drag_conserves_weighted_momentum(self):
        mode = self.mode()
        z = 1.0e5
        coeff = lmhd.background_coefficients(
            self.cosmology, z, self.ionization, mode.b0_gauss
        )
        matrix = lmhd._p1_matrix(
            z, mode, self.cosmology, self.ionization,
            include_gravity=False, include_dark_matter=False,
            include_photon_stress=False, include_helium_electrons=False,
            recombination_rates=None,
        )
        # Remove expansion; the scattering subblock conserves vb + vg/R.
        h = coeff["hubble"]
        gamma = coeff["alpha_drag"]
        nu = coeff["nu_scatter"]
        self.assertAlmostEqual(nu / gamma, coeff["R"], places=11)
        drag = matrix[np.ix_([3, 7], [3, 7])] * h
        drag[0, 0] += h
        weight = np.asarray([1.0, 1.0 / coeff["R"]])
        np.testing.assert_allclose(weight.dot(drag), 0.0, rtol=0, atol=1.0e-18)

    def test_p1_is_exactly_linear_in_perturbation(self):
        kwargs = dict(
            cosmology=self.cosmology,
            z_start=1.0e5,
            z_end=9.0e4,
            output_redshifts=[9.0e4],
            ionization=self.ionization,
            max_dln_a=2.0e-3,
            include_gravity=False,
            include_photon_stress=True,
        )
        low = lmhd.evolve_photon_p1(self.mode(1.0e-5), **kwargs)
        high = lmhd.evolve_photon_p1(self.mode(1.0e-3), **kwargs)
        for field in ("b_mode_over_initial", "delta_b"):
            left = low.field(field)[-1]
            right = high.field(field)[-1]
            if field == "delta_b":
                left /= 1.0e-5
                right /= 1.0e-3
            np.testing.assert_allclose(left, right, rtol=2.0e-11, atol=2.0e-12)

    def test_p1_passively_evolves_ionization_perturbation(self):
        result = lmhd.evolve_photon_p1(
            self.mode(1.0e-3),
            self.cosmology,
            1200.0,
            1100.0,
            [1100.0],
            self.ionization,
            max_dln_a=1.0e-3,
            include_gravity=False,
            include_photon_stress=True,
            recombination_rates=lambda z: (-1.0e-12, -2.0e-12, 3.0e-13),
        )
        self.assertIn("delta_xe", result.fields)
        self.assertTrue(np.isfinite(result.field("delta_xe")[-1]))
        self.assertNotEqual(abs(result.field("delta_xe")[-1]), 0.0)

    def test_p1_temperature_feedback_is_finite(self):
        temperature = lambda z: self.cosmology.tcmb0 * (1.0 + z)
        result = lmhd.evolve_photon_p1(
            self.mode(1.0e-3),
            self.cosmology,
            1200.0,
            1100.0,
            [1100.0],
            self.ionization,
            max_dln_a=1.0e-3,
            include_gravity=False,
            include_photon_stress=True,
            recombination_rates=lambda z: (
                -1.0e-12, -2.0e-12, 3.0e-13, 4.0e-12
            ),
            baryon_temperature=temperature,
        )
        self.assertIn("delta_temperature_fraction", result.fields)
        self.assertTrue(np.isfinite(result.field("delta_temperature")[-1]))
        self.assertNotEqual(abs(result.field("delta_temperature")[-1]), 0.0)

    def test_p1_accepts_complete_complex_handoff_state(self):
        temperature = lambda z: self.cosmology.tcmb0 * (1.0 + z)
        state = np.asarray(
            [complex(index, -0.25 * index) for index in range(14)],
            dtype=np.complex128,
        )
        result = lmhd.evolve_photon_p1(
            self.mode(1.0e-3), self.cosmology, 1200.0, 1100.0, [1200.0],
            self.ionization, include_gravity=False, include_dark_matter=True,
            recombination_rates=lambda z: (0.0, 0.0, 0.0, 0.0),
            baryon_temperature=temperature, initial_state=state,
        )
        self.assertEqual(result.field("b_y_over_B0")[0], state[1])
        self.assertEqual(result.field("delta_b")[0], state[2])
        self.assertEqual(result.field("v_dm_z")[0], state[11])
        self.assertEqual(result.field("delta_xe")[0], state[12])
        self.assertEqual(result.field("delta_temperature")[0], state[13])
        self.assertTrue(result.metadata["custom_initial_state"])

        with self.assertRaisesRegex(ValueError, "expected"):
            lmhd.evolve_photon_p1(
                self.mode(1.0e-3), self.cosmology, 1200.0, 1100.0,
                [1100.0], self.ionization, include_dark_matter=True,
                recombination_rates=lambda z: (0.0, 0.0, 0.0, 0.0),
                baryon_temperature=temperature,
                initial_state=np.zeros(13, dtype=np.complex128),
            )

    def test_magnus_exponential_refines(self):
        mode = self.mode()
        common = dict(
            mode=mode,
            cosmology=self.cosmology,
            z_start=2.0e4,
            z_end=1.8e4,
            output_redshifts=[1.8e4],
            ionization=self.ionization,
            include_gravity=False,
        )
        coarse = lmhd.evolve_photon_p1(max_dln_a=4.0e-3, **common)
        medium = lmhd.evolve_photon_p1(max_dln_a=2.0e-3, **common)
        fine = lmhd.evolve_photon_p1(max_dln_a=1.0e-3, **common)
        yc = coarse.field("delta_b")[-1]
        ym = medium.field("delta_b")[-1]
        yf = fine.field("delta_b")[-1]
        err_coarse = abs(yc - yf)
        err_medium = abs(ym - yf)
        self.assertLess(err_medium, 0.2 * err_coarse + 1.0e-14)


if __name__ == "__main__":
    unittest.main()
