#!/usr/bin/env python3

import math
from pathlib import Path
import unittest

import numpy as np
from tools.reference import lmhd, sv25_paper


class SpectrumConventionTest(unittest.TestCase):
    def setUp(self):
        self.spectrum = sv25_paper.MagneticSpectrum(315.0e-12, -0.1)

    def test_power_and_dimensionless_power_are_consistent(self):
        k = np.geomspace(0.03, 2.0e5, 17)
        expected = k**3 * self.spectrum.power(k) / (2.0 * math.pi**2)
        np.testing.assert_allclose(
            self.spectrum.dimensionless_power(k), expected, rtol=2.0e-15
        )

    def test_ir_normalization(self):
        expected = 0.5 * abs(self.spectrum.epsilon) * self.spectrum.b0_gauss**2
        self.assertAlmostEqual(
            float(self.spectrum.dimensionless_power(self.spectrum.k_lambda_mpc)),
            expected,
        )

    def test_saha_equations_have_expected_sign(self):
        xe = np.asarray([0.1, 0.5, 0.9])
        self.assertTrue(np.all(sv25_paper.saha_linear_delta_xe(xe, 0.1) < 0.0))
        self.assertTrue(np.all(sv25_paper.saha_mean_ionization_shift(xe, 0.01) < 0.0))

    def test_paper_cosmology_uses_planck_physical_baryon_density(self):
        cosmology = sv25_paper.paper_cosmology()
        self.assertAlmostEqual(cosmology.omega_b * cosmology.h**2, 0.02242)
        self.assertAlmostEqual(
            cosmology.omega_m + cosmology.omega_lambda + cosmology.omega_r,
            1.0,
        )


class TransferConventionTest(unittest.TestCase):
    def setUp(self):
        theta, mu, weight = sv25_paper.angular_nodes(6)
        del theta
        self.spectrum = sv25_paper.MagneticSpectrum(315.0e-12, -0.1)
        k = np.asarray([1.0, 10.0])
        z = np.asarray([1100.0])
        shape = (k.size, mu.size, z.size)
        one = np.ones(shape, dtype=np.complex128)
        self.grid = sv25_paper.TransferGrid(
            k,
            mu,
            weight,
            z,
            {"b_mode_over_initial": one, "delta_b": 2.0 * one},
            {"b_mode_over_initial": one, "delta_b": np.zeros_like(one)},
            {"b0_gauss": self.spectrum.b0_gauss},
        )

    def test_angular_average_matches_paper_definition(self):
        np.testing.assert_allclose(
            self.grid.angular_cross("delta_b"), np.full((2, 1), 8.0)
        )

    def test_split_quadrature_covers_full_solid_angle(self):
        theta, mu, weight = sv25_paper.split_angular_nodes(2)
        self.assertTrue(np.all((theta > 0.0) & (theta < math.pi / 2.0)))
        self.assertTrue(np.all((mu > 0.0) & (mu < 1.0)))
        self.assertAlmostEqual(float(np.sum(weight)), 2.0, places=14)

    def test_uniform_theta_quadrature_exposes_endpoint_weight(self):
        theta, mu, weight = sv25_paper.uniform_theta_nodes(16)
        self.assertEqual(theta.size, 17)
        self.assertAlmostEqual(float(mu[-1]), 0.0, places=14)
        raw_endpoint = math.pi / 32.0
        raw_sum = 2.0 * (0.5 * math.pi / 16.0) * float(
            np.sum(np.sin(theta)[1:-1]) + 0.5
        )
        self.assertAlmostEqual(
            float(weight[-1]), raw_endpoint * 2.0 / raw_sum, places=14
        )
        self.assertAlmostEqual(float(np.sum(weight)), 2.0, places=14)

    def test_pristine_magnetic_transfer_preserves_primordial_power(self):
        np.testing.assert_allclose(
            sv25_paper.magnetic_power(self.grid, self.spectrum)[:, 0],
            self.spectrum.dimensionless_power(self.grid.k_mpc),
        )

    def test_density_power_and_log_integral(self):
        power = sv25_paper.density_power(self.grid, self.spectrum)
        expected = 8.0 * self.spectrum.fractional_mode_weight(self.grid.k_mpc)
        np.testing.assert_allclose(power[:, 0], expected)
        np.testing.assert_allclose(
            sv25_paper.clumping_factor(self.grid, self.spectrum),
            np.trapezoid(expected, np.log(self.grid.k_mpc)),
        )
        np.testing.assert_array_equal(
            sv25_paper.scalar_field_power(
                self.grid, self.spectrum, "delta_b"
            ),
            power,
        )

    def test_physical_transfer_grid_builder(self):
        cosmology = sv25_paper.paper_cosmology()
        ionization = lambda z: lmhd.saha_ionization(cosmology, z)
        grid = sv25_paper.build_transfer_grid(
            cosmology,
            ionization,
            np.asarray([1.0e5]),
            np.asarray([2000.0, 1000.0]),
            315.0e-12,
            angle_count=2,
            closure="saha",
            max_dln_a=1.0e-2,
            include_alfven=False,
        )
        self.assertEqual(grid.magnetosonic["delta_b"].shape, (1, 2, 2))
        self.assertTrue(np.all(np.isfinite(grid.magnetosonic["delta_b"][:, :, -1])))
        self.assertTrue(grid.metadata["physical_mode_initialization"])

    def test_transfer_grid_accepts_custom_background_electron_history(self):
        cosmology = sv25_paper.paper_cosmology()
        grid = sv25_paper.build_transfer_grid(
            cosmology,
            lambda z: lmhd.saha_ionization(cosmology, z),
            np.asarray([1.0e5]),
            np.asarray([2000.0, 1000.0]),
            315.0e-12,
            angle_count=2,
            closure="saha",
            max_dln_a=1.0e-2,
            include_alfven=False,
            background_electron_fraction=lambda _: 1.02,
        )
        self.assertTrue(grid.metadata["custom_background_electron_fraction"])
        self.assertTrue(np.all(np.isfinite(grid.magnetosonic["delta_b"][:, :, -1])))

    def test_temperature_grid_can_include_fixed_closure_alfven_modes(self):
        cosmology = sv25_paper.paper_cosmology()
        ionization = lmhd.PeeblesIonization(
            cosmology, zmax=3500.0, zmin=500.0, samples=800
        )
        grid = sv25_paper.build_transfer_grid(
            cosmology,
            ionization,
            np.asarray([1.0e4]),
            np.asarray([1900.0, 1000.0]),
            315.0e-12,
            angle_count=2,
            closure="three_level",
            recombination_rates=lambda z, k: (-1.0, -2.0, 0.0, 0.0),
            max_dln_a=1.0e-2,
            include_alfven=True,
            baryon_temperature=lambda z: cosmology.tcmb0 * (1.0 + z),
        )
        self.assertTrue(np.all(np.isfinite(grid.alfven["v_b_x"][:, :, -1])))

    def test_fixed_angle_grid_requires_an_interior_angle(self):
        cosmology = sv25_paper.paper_cosmology()
        with self.assertRaisesRegex(ValueError, "fixed angle"):
            sv25_paper.build_transfer_grid(
                cosmology,
                lambda z: 1.0,
                [1.0],
                [1000.0],
                1.0e-12,
                angle_scheme="fixed",
                fixed_theta=0.0,
            )


class VisibilityHistoryTest(unittest.TestCase):
    def setUp(self):
        self.cosmology = sv25_paper.paper_cosmology()
        self.ionization = lmhd.PeeblesIonization(
            self.cosmology, zmax=2000.0, zmin=0.0, samples=3000
        )
        self.spectrum = sv25_paper.MagneticSpectrum(315.0e-12, -0.1)

    def _grid(self, amplitude):
        k = np.asarray([1.0, math.e])
        mu = np.asarray([-0.5, 0.5])
        weight = np.ones(2)
        redshift = np.linspace(600.0, 1500.0, 91)
        shape = (k.size, mu.size, redshift.size)
        fields = {
            "delta_xe": np.full(shape, amplitude, dtype=np.complex128),
            "delta_b": np.zeros(shape, dtype=np.complex128),
        }
        return sv25_paper.TransferGrid(
            k,
            mu,
            weight,
            redshift,
            fields,
            fields,
            {"b0_gauss": self.spectrum.b0_gauss},
        )

    def test_zero_perturbations_reduce_to_homogeneous_visibility(self):
        history = sv25_paper.visibility_history(
            self._grid(0.0),
            self.spectrum,
            self.cosmology,
            self.ionization,
        )
        np.testing.assert_array_equal(
            history.optical_depth_variance,
            np.zeros_like(history.redshift),
        )
        np.testing.assert_allclose(
            history.visibility, history.homogeneous_visibility, rtol=2.0e-15
        )
        self.assertAlmostEqual(history.peak_shift, 0.0, places=12)

    def test_visibility_accepts_total_background_electron_history(self):
        history = sv25_paper.visibility_history(
            self._grid(0.0),
            self.spectrum,
            self.cosmology,
            self.ionization,
            background_electron_fraction=lambda z: self.ionization(z) + 0.02,
        )
        np.testing.assert_allclose(
            history.visibility, history.homogeneous_visibility, rtol=2.0e-15
        )
        self.assertAlmostEqual(history.peak_shift, 0.0, places=12)

    def test_mode_is_integrated_before_unequal_time_variance(self):
        amplitude = 1.0e-3
        grid = self._grid(amplitude)
        history = sv25_paper.visibility_history(
            grid, self.spectrum, self.cosmology, self.ionization
        )
        weight = np.asarray(
            [
                lmhd.CLIGHT
                * lmhd.SIGMA_T
                * self.cosmology.hydrogen_density(float(z))
                / ((1.0 + z) * self.cosmology.hubble(float(z)))
                for z in grid.redshift
            ]
        )
        integrated_mode = np.concatenate(
            ([0.0], np.cumsum(0.5 * np.diff(grid.redshift) * (weight[1:] + weight[:-1])))
        ) * amplitude
        k_factor = np.trapezoid(
            2.0 * self.spectrum.fractional_mode_weight(grid.k_mpc),
            np.log(grid.k_mpc),
        )
        np.testing.assert_allclose(
            history.optical_depth_variance,
            k_factor * integrated_mode**2,
            rtol=3.0e-13,
            atol=1.0e-18,
        )

    def test_zero_first_order_fields_give_zero_mean_temperature_shift(self):
        k = np.asarray([10.0, 100.0])
        mu = np.asarray([-0.5, 0.5])
        weight = np.ones(2)
        redshift = np.asarray([600.0, 1000.0, 1400.0, 1900.0])
        shape = (k.size, mu.size, redshift.size)
        fields = {
            name: np.zeros(shape, dtype=np.complex128)
            for name in (
                "delta_b",
                "delta_xe",
                "delta_temperature",
                "delta_temperature_fraction",
                "theta_b",
                "v_b_x",
                "v_b_y",
                "v_b_z",
            )
        }
        grid = sv25_paper.TransferGrid(
            k,
            mu,
            weight,
            redshift,
            fields,
            fields,
            {"b0_gauss": self.spectrum.b0_gauss},
        )
        result = sv25_paper.mean_temperature_shift(
            grid,
            self.spectrum,
            self.cosmology,
            lambda z: 0.5,
            lambda z: self.cosmology.tcmb0 * (1.0 + z),
            np.zeros_like(redshift),
        )
        np.testing.assert_array_equal(
            result.absolute_temperature_shift, np.zeros_like(redshift)
        )
        np.testing.assert_array_equal(
            result.drag_dissipation_source_k_s, np.zeros_like(redshift)
        )
        correction, cross = (
            sv25_paper.temperature_ionization_source_correction(
                grid,
                self.spectrum,
                self.cosmology,
                lambda z: 0.5,
                lambda z: self.cosmology.tcmb0 * (1.0 + z),
                np.zeros_like(redshift),
            )
        )
        np.testing.assert_array_equal(correction, np.zeros_like(redshift))
        np.testing.assert_array_equal(cross, np.zeros_like(redshift))
        snapshots = [
            sv25_paper.NonlocalSecondOrderSnapshot(
                redshift=float(z),
                mean_rate_linear_s=-1.0,
                mean_rate_source_s=0.0,
                delta_xe_density_cross=0.0,
                source_line_average=0.0,
                source_residual_max=0.0,
                peebles_second_order=0.0,
                a2_line_average=0.0,
                c2_line_average=0.0,
                covariance_delta_xe2=0.0,
                covariance_feq_delta_xe=0.0,
                covariance_fbar_delta_xe=0.0,
            )
            for z in redshift
        ]
        coupled = sv25_paper.integrate_coupled_mean_thermal_history(
            grid,
            self.spectrum,
            self.cosmology,
            lambda z: 0.5,
            lambda z: self.cosmology.tcmb0 * (1.0 + z),
            snapshots,
            background_electron_fraction=lambda z: 0.52,
        )
        np.testing.assert_array_equal(
            coupled.mean_ionization_shift, np.zeros_like(redshift)
        )
        np.testing.assert_array_equal(
            coupled.fractional_temperature_shift, np.zeros_like(redshift)
        )


class HomogeneousRecombinationTest(unittest.TestCase):
    def test_fokker_planck_history_is_finite_and_physical(self):
        history = sv25_paper.homogeneous_recombination_histories(
            sv25_paper.paper_cosmology(),
            zmin=1000.0,
            zmax=1200.0,
            output_count=11,
            response_count=3,
            xmax=1000.0,
            bins_per_width=2.0,
            iterations=1,
            rtol=1.0e-7,
        )
        self.assertTrue(np.all(np.diff(history.redshift) > 0.0))
        self.assertTrue(np.all((history.fokker_planck_xe > 0.0) & (history.fokker_planck_xe <= 1.0)))
        self.assertTrue(np.all(history.fokker_planck_escape_probability > 0.0))


class LocalThreeLevelTest(unittest.TestCase):
    def setUp(self):
        self.cosmology = sv25_paper.paper_cosmology()
        self.ionization = lmhd.PeeblesIonization(
            self.cosmology, zmax=1400.0, zmin=900.0, samples=600
        )

    def test_local_rate_derivatives_refine_and_are_symmetric(self):
        xe = self.ionization(1100.0)
        coarse = sv25_paper.local_peebles_rate_derivatives(
            self.cosmology, 1100.0, xe, relative_step=2.0e-4
        )
        fine = sv25_paper.local_peebles_rate_derivatives(
            self.cosmology, 1100.0, xe, relative_step=1.0e-4
        )
        self.assertTrue(np.all(np.isfinite(coarse[0])))
        self.assertTrue(np.all(np.isfinite(coarse[1])))
        np.testing.assert_allclose(coarse[1], coarse[1].T, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(coarse[0], fine[0], rtol=2.0e-6)
        np.testing.assert_allclose(coarse[1], fine[1], rtol=2.0e-3, atol=1.0e-20)

    def test_local_response_contains_expansion_rate(self):
        response = sv25_paper.LocalPeeblesResponse(
            self.cosmology, self.ionization
        )(1100.0, 1.0e3)
        self.assertEqual(response.shape, (3,))
        self.assertTrue(np.all(np.isfinite(response)))
        self.assertNotEqual(float(response[2]), 0.0)

    def test_local_response_table_matches_direct_rate(self):
        table = sv25_paper.LocalPeeblesResponseTable.build(
            self.cosmology,
            self.ionization,
            zmin=1000.0,
            zmax=1200.0,
            samples=101,
        )
        direct = sv25_paper.LocalPeeblesResponse(
            self.cosmology, self.ionization
        )(1101.0, 1.0e4)
        np.testing.assert_allclose(table(1101.0, 1.0e4), direct, rtol=2.0e-5)

    def test_zero_first_order_fields_have_zero_mean_shift(self):
        k = np.asarray([1.0, 10.0, 100.0])
        mu = np.asarray([-0.5, 0.5])
        weight = np.ones(2)
        redshift = np.asarray([1200.0, 1100.0, 1000.0])
        shape = (k.size, mu.size, redshift.size)
        fields = {
            name: np.zeros(shape, dtype=np.complex128)
            for name in ("delta_xe", "delta_b", "theta_b")
        }
        grid = sv25_paper.TransferGrid(
            k,
            mu,
            weight,
            redshift,
            fields,
            fields,
            {"b0_gauss": 315.0e-12},
        )
        total, second, cross = sv25_paper.local_three_level_mean_ionization_shift(
            grid,
            sv25_paper.MagneticSpectrum(315.0e-12, -0.1),
            self.cosmology,
            self.ionization,
        )
        np.testing.assert_array_equal(total, np.zeros_like(redshift))
        np.testing.assert_array_equal(second, np.zeros_like(redshift))
        np.testing.assert_array_equal(cross, np.zeros_like(redshift))

    def test_zero_fields_have_zero_nonlocal_quadratic_source(self):
        k = np.asarray([1.0, 10.0, 100.0])
        mu = np.asarray([-0.5, 0.5])
        weight = np.ones(2)
        shape = (k.size, mu.size, 1)
        fields = {
            name: np.zeros(shape, dtype=np.complex128)
            for name in ("delta_xe", "delta_b", "theta_b", "v_b_x", "v_b_y")
        }
        grid = sv25_paper.TransferGrid(
            k,
            mu,
            weight,
            np.asarray([1100.0]),
            fields,
            fields,
            {"b0_gauss": 315.0e-12, "include_alfven": True},
        )
        snapshot = sv25_paper.nonlocal_second_order_snapshot(
            grid,
            sv25_paper.MagneticSpectrum(315.0e-12, -0.1),
            self.cosmology,
            self.ionization,
            0,
            xmax=1000.0,
            bins_per_width=2.0,
            lmax=16,
        )
        self.assertEqual(snapshot.source_line_average, 0.0)
        self.assertEqual(snapshot.mean_rate_source_s, 0.0)
        self.assertEqual(snapshot.delta_xe_density_cross, 0.0)
        self.assertLess(snapshot.mean_rate_linear_s, 0.0)
        local_gradient, _ = sv25_paper.local_peebles_rate_derivatives(
            self.cosmology, 1100.0, self.ionization(1100.0)
        )
        self.assertAlmostEqual(
            snapshot.mean_rate_linear_s / local_gradient[0], 1.0, delta=0.04
        )

    def test_zero_nonlocal_snapshots_integrate_to_zero(self):
        rows = [
            sv25_paper.NonlocalSecondOrderSnapshot(
                redshift=z,
                mean_rate_linear_s=-1.0e-12,
                mean_rate_source_s=0.0,
                delta_xe_density_cross=0.0,
                source_line_average=0.0,
                source_residual_max=0.0,
                peebles_second_order=0.5,
                a2_line_average=0.0,
                c2_line_average=0.0,
                covariance_delta_xe2=0.0,
                covariance_feq_delta_xe=0.0,
                covariance_fbar_delta_xe=0.0,
            )
            for z in (900.0, 1100.0, 1300.0)
        ]
        redshift, total, second = (
            sv25_paper.integrate_nonlocal_mean_ionization_shift(
                rows, self.cosmology
            )
        )
        np.testing.assert_array_equal(redshift, np.asarray([900.0, 1100.0, 1300.0]))
        np.testing.assert_array_equal(total, np.zeros(3))
        np.testing.assert_array_equal(second, np.zeros(3))

    def test_nonlocal_source_recovers_local_quadratic_rate_at_low_k(self):
        z = 1100.0
        b0 = 315.0e-12
        k = np.geomspace(1.0e-2, 4.0e-2, 3)
        mu = np.asarray([-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)])
        weight = np.ones(2)
        shape = (k.size, mu.size, 1)
        a_hubble = self.cosmology.hubble(z) / (1.0 + z)

        def constant(value):
            return np.full(shape, value, dtype=np.complex128)

        fields = {
            "delta_b": constant(0.02),
            "delta_xe": constant(-0.01),
            "theta_b": constant(0.0 * a_hubble),
            "v_b_x": constant(0.0),
            "v_b_y": constant(0.0),
        }
        grid = sv25_paper.TransferGrid(
            k,
            mu,
            weight,
            np.asarray([z]),
            fields,
            fields,
            {"b0_gauss": b0, "include_alfven": True},
        )
        spectrum = sv25_paper.MagneticSpectrum(b0, -0.1)
        snapshot = sv25_paper.nonlocal_second_order_snapshot(
            grid,
            spectrum,
            self.cosmology,
            self.ionization,
            0,
            xmax=1000.0,
            bins_per_width=2.0,
            lmax=12,
        )
        variables = (fields["delta_xe"], fields["delta_b"], fields["theta_b"])
        covariance = np.asarray(
            [
                [
                    np.real(
                        sv25_paper._array_cross_correlation(
                            grid, spectrum, variables[i], variables[j]
                        )
                    )[0]
                    for j in range(3)
                ]
                for i in range(3)
            ]
        )
        gradient, hessian = sv25_paper.local_peebles_rate_derivatives(
            self.cosmology, z, self.ionization(z)
        )
        expected_source = 0.5 * float(np.sum(hessian * covariance))
        self.assertAlmostEqual(
            snapshot.mean_rate_linear_s / gradient[0], 1.0, delta=0.03
        )
        self.assertAlmostEqual(
            snapshot.mean_rate_source_s / expected_source, 1.0, delta=0.03
        )


class PublishedLimitTest(unittest.TestCase):
    def test_one_efold_scale_is_log_interpolated(self):
        wave = np.geomspace(1.0, 100.0, 9)
        scale = np.asarray([5.0, 10.0])
        transfer = (wave[:, None] / scale) ** -1.0
        wavelength = sv25_paper.one_efold_damping_wavelength(wave, transfer)
        np.testing.assert_allclose(
            wavelength,
            2.0 * math.pi / (math.e * scale),
            rtol=1.0e-12,
        )

    def test_acoustic_diffusion_scale_is_positive_and_grows(self):
        cosmology = sv25_paper.paper_cosmology()
        redshift = np.asarray([1000.0, 2000.0, 10000.0])
        wave = sv25_paper.acoustic_diffusion_wavenumber(
            cosmology, lambda z: 1.0, redshift, integration_points=256
        )
        self.assertTrue(np.all(np.isfinite(wave) & (wave > 0.0)))
        self.assertTrue(np.all(np.diff(wave) > 0.0))

    def test_conservation_scale_matches_figure_five_marker(self):
        cosmology = lmhd.Cosmology()
        ionization = lmhd.PeeblesIonization(
            cosmology, zmax=3500.0, zmin=500.0, samples=1600
        )
        kcons = sv25_paper.conservation_wavenumber(
            cosmology, ionization, 315.0e-12, 1900.0
        )
        self.assertGreater(kcons, 700.0)
        self.assertLess(kcons, 1000.0)

    def test_primordial_density_ratio_is_small_for_paper_example(self):
        ratio = sv25_paper.primordial_density_to_magnetic_ratio(
            np.asarray([1.0e2, 1.0e5]), -0.1, math.pi / 2.0
        )
        self.assertTrue(np.all(ratio < 0.1))
        self.assertGreater(ratio[-1], ratio[0])

    def test_density_seed_transfer_reaches_paper_tcr_attractor(self):
        cosmology = sv25_paper.paper_cosmology()
        ionization = lambda z: lmhd.saha_ionization(cosmology, z)
        b0 = 315.0e-12
        theta = math.pi / 4.0
        transfer = sv25_paper.density_seeded_magnetic_transfer(
            cosmology,
            ionization,
            np.asarray([1.0e3]),
            b0,
            theta,
            max_dln_a=4.0e-3,
        )
        expected = 3.0 * b0 * math.sin(theta)
        np.testing.assert_allclose(transfer.ratio_gauss, expected, rtol=2.0e-5)

    def test_red_spectrum_lorentz_integrand_peaks_in_squeezed_limit(self):
        near = abs(float(sv25_paper.lorentz_variance_integrand(1.0, 1.0e-3, -4.0)))
        away = abs(float(sv25_paper.lorentz_variance_integrand(0.5, 0.5, -4.0)))
        self.assertGreater(near, 100.0 * away)


if __name__ == "__main__":
    unittest.main()
