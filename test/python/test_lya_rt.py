#!/usr/bin/env python3

import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.reference import lmhd, lya_rt


class LymanAlphaAtomicTest(unittest.TestCase):
    def test_voigt_normalization(self):
        x = np.linspace(-200.0, 200.0, 200001)
        profile = lya_rt.voigt_profile(x, 8.6e-4)
        self.assertAlmostEqual(np.trapezoid(profile, x), 1.0, delta=5.0e-6)

    def test_true_absorption_uses_displayed_branching_formula(self):
        temperature = 2.725 * 1101.0
        probability = lya_rt.true_absorption_probability(temperature)
        self.assertAlmostEqual(probability, 1.1791553531894981e-4, delta=2.0e-14)

    def test_response_table_nodes_and_low_k_limit(self):
        redshift = np.asarray([700.0, 900.0, 1100.0])
        k_mpc = np.asarray([100.0, 1000.0, 10000.0])
        local = -np.ones((3, 3))
        local[:, 2] = -2.0
        delta = np.empty((3, 3, 3))
        for iz in range(3):
            for ik in range(3):
                delta[iz, ik] = (
                    -10.0 ** (iz + ik),
                    -2.0 * 10.0 ** (iz + ik),
                    3.0 * 10.0 ** (iz + ik),
                )
        table = lya_rt.ResponseTable(redshift, k_mpc, local, delta)
        np.testing.assert_allclose(
            table.rates(900.0, 1000.0, differential_only=True), delta[1, 1]
        )
        low = table.rates(900.0, 10.0, differential_only=True)
        np.testing.assert_allclose(low, 0.01 * delta[1, 0])
        np.testing.assert_array_equal(table.rates(1200.0, 1000.0), np.zeros(3))

    def test_response_table_loads_npz_and_tsv_equivalently(self):
        redshift = np.asarray([700.0, 900.0])
        k_mpc = np.asarray([100.0, 1000.0, 10000.0])
        local = np.arange(8, dtype=float).reshape(2, 4) + 1.0
        delta = np.arange(24, dtype=float).reshape(2, 3, 4) - 12.0
        header = (
            "redshift\tk_mpc\tlocal_density_rate_s\tlocal_ionization_rate_s\t"
            "local_velocity_divergence_rate_s\tlocal_temperature_rate_s\t"
            "delta_density_rate_s\tdelta_ionization_rate_s\t"
            "delta_velocity_divergence_rate_s\tdelta_temperature_rate_s"
        )
        rows = []
        for iz, z in enumerate(redshift):
            for ik, k in enumerate(k_mpc):
                values = [z, k, *local[iz], *delta[iz, ik]]
                rows.append("\t".join(str(value) for value in values))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            npz_path = root / "response_table.npz"
            tsv_path = root / "response_table.tsv"
            np.savez(
                npz_path,
                redshift=redshift,
                k_mpc=k_mpc,
                local_rates=local,
                delta_rates=delta,
            )
            tsv_path.write_text(header + "\n" + "\n".join(rows) + "\n")
            from_npz = lya_rt.ResponseTable.load(npz_path)
            from_tsv = lya_rt.ResponseTable.load(tsv_path)

        np.testing.assert_array_equal(from_tsv.redshift, from_npz.redshift)
        np.testing.assert_array_equal(from_tsv.k_mpc, from_npz.k_mpc)
        np.testing.assert_array_equal(from_tsv.local_rates, from_npz.local_rates)
        np.testing.assert_array_equal(from_tsv.delta_rates, from_npz.delta_rates)


class LymanAlphaHierarchyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cosmology = lmhd.Cosmology()
        cls.ionization = lmhd.PeeblesIonization(
            cls.cosmology, zmax=3500, zmin=500, samples=1200
        )
        cls.background = lya_rt.lya_background(
            cls.cosmology, 1100.0, cls.ionization
        )
        cls.homogeneous = lya_rt.solve_homogeneous_line(
            cls.background,
            xmax=20.0,
            bins_per_width=0.5,
            tolerance=2.0e-3,
            max_nodes=30000,
        )

    def test_homogeneous_boundaries_and_scale(self):
        line = self.homogeneous
        self.assertLess(line.residual_max, 1.0e-10)
        self.assertAlmostEqual(
            line.occupation[-1], self.background.f_blackbody, delta=1.0e-25
        )
        self.assertLess(abs(line.derivative[0]), 1.0e-12)
        self.assertGreater(line.line_average, self.background.f_blackbody)
        self.assertLess(line.line_average, 1.1 * self.background.f_equilibrium)

    def test_zero_k_hierarchy_has_no_higher_multipoles(self):
        result = lya_rt.solve_scalar_basis(
            self.homogeneous,
            0.0,
            "A",
            lmax=3,
            bins_per_width=0.5,
            tolerance=2.0e-3,
            max_nodes=30000,
        )
        self.assertLess(result.residual_max, 1.0e-12)
        self.assertEqual(float(np.max(np.abs(result.multipoles[1:]))), 0.0)

    def test_c_basis_has_expected_destruction_plateau(self):
        result = lya_rt.solve_scalar_basis(
            self.homogeneous,
            1.0e5,
            "C",
            lmax=3,
            bins_per_width=0.5,
            tolerance=3.0e-3,
            max_nodes=40000,
        )
        plateau = 1.0 / self.background.p_ab
        self.assertGreater(float(np.max(result.multipoles[0])), 0.8 * plateau)
        self.assertLess(float(np.max(result.multipoles[0])), 1.02 * plateau)

    def test_sparse_trapezoid_matches_collocation(self):
        sparse_line = lya_rt.solve_homogeneous_line(
            self.background,
            xmax=20.0,
            bins_per_width=0.5,
            method="sparse",
        )
        collocation = lya_rt.solve_scalar_basis(
            self.homogeneous,
            1.0e5,
            "A",
            lmax=4,
            bins_per_width=0.5,
            tolerance=2.0e-3,
        )
        sparse = lya_rt.solve_scalar_basis(
            sparse_line,
            1.0e5,
            "A",
            lmax=4,
            bins_per_width=0.5,
            method="sparse",
        )
        relative = abs(sparse.line_average / collocation.line_average - 1.0)
        self.assertLess(relative, 0.02)

    def test_vector_hierarchies_have_zero_streaming_symmetry(self):
        plus = lya_rt.solve_vector_basis(
            self.homogeneous,
            0.0,
            "+",
            lmax=4,
            bins_per_width=0.5,
            method="sparse",
        )
        minus = lya_rt.solve_vector_basis(
            self.homogeneous,
            0.0,
            "-",
            lmax=4,
            bins_per_width=0.5,
            method="sparse",
        )
        self.assertEqual(float(np.max(np.abs(plus.multipoles[0]))), 0.0)
        self.assertEqual(float(np.max(np.abs(plus.multipoles[[1, 3, 4]]))), 0.0)
        self.assertGreater(float(np.max(np.abs(plus.multipoles[2]))), 0.0)
        np.testing.assert_allclose(plus.multipoles, minus.multipoles, rtol=1.0e-12)

    def test_vector_sparse_matches_collocation(self):
        collocation = lya_rt.solve_vector_basis(
            self.homogeneous,
            1.0e5,
            "+",
            lmax=4,
            bins_per_width=0.5,
            tolerance=2.0e-3,
            method="bvp",
        )
        sparse = lya_rt.solve_vector_basis(
            self.homogeneous,
            1.0e5,
            "+",
            lmax=4,
            # The transverse Doppler source is sharply concentrated in the
            # line core. The paper uses 50 bins per Doppler width here.
            bins_per_width=50.0,
            method="sparse",
        )
        collocation_l2 = np.trapezoid(
            np.abs(collocation.multipoles[2]) ** 2,
            collocation.x,
        )
        sparse_l2 = np.trapezoid(np.abs(sparse.multipoles[2]) ** 2, sparse.x)
        self.assertLess(abs(sparse_l2 / collocation_l2 - 1.0), 0.03)

    def test_vector_marching_matches_sparse(self):
        sparse = lya_rt.solve_vector_basis(
            self.homogeneous,
            1.0e5,
            "-",
            lmax=5,
            bins_per_width=8.0,
            method="sparse",
        )
        marching = lya_rt.solve_vector_basis(
            self.homogeneous,
            1.0e5,
            "-",
            lmax=5,
            bins_per_width=8.0,
            method="marching",
        )
        np.testing.assert_allclose(
            marching.multipoles,
            sparse.multipoles,
            rtol=2.0e-10,
            atol=1.0e-30,
        )
        np.testing.assert_allclose(
            marching.derivatives,
            sparse.derivatives,
            rtol=2.0e-10,
            atol=1.0e-30,
        )
        self.assertLess(marching.residual_max, 1.0e-11)

    def test_vector_second_order_source_uses_equation_derivative(self):
        plus = lya_rt.solve_vector_basis(
            self.homogeneous, 1.0e5, "+", lmax=4,
            bins_per_width=8.0, method="marching",
        )
        minus = lya_rt.solve_vector_basis(
            self.homogeneous, 1.0e5, "-", lmax=4,
            bins_per_width=8.0, method="marching",
        )
        source_grid, source = lya_rt.vector_second_order_source(plus, minus)
        np.testing.assert_array_equal(source_grid, plus.x)
        np.testing.assert_allclose(
            source,
            -(plus.equation_derivatives[2] + minus.equation_derivatives[2]) / 5.0,
            rtol=0.0,
            atol=0.0,
        )
        self.assertTrue(np.all(np.isfinite(source)))

    def test_second_order_bases_reduce_to_zero_k_A_and_C(self):
        for name, source in (
            ("A", self.homogeneous.derivative),
            ("C", -self.background.tau_s * self.homogeneous.profile),
        ):
            expected = lya_rt.solve_scalar_basis(
                self.homogeneous,
                0.0,
                name,
                lmax=3,
                bins_per_width=4.0,
                method="sparse",
            )
            result = lya_rt.solve_second_order_source(
                self.homogeneous,
                source,
                bins_per_width=4.0,
                method="sparse",
            )
            np.testing.assert_allclose(
                result.occupation,
                expected.multipoles[0],
                rtol=2.0e-6,
                atol=1.0e-30,
            )
            self.assertLess(
                abs(result.line_average / expected.line_average - 1.0), 2.0e-6
            )

    def test_equilibrium_source_coefficients_reconstruct_C_source(self):
        first_order = lya_rt.first_order_response(
            self.cosmology,
            self.ionization,
            1100.0,
            1.0e5,
            xmax=40.0,
            bins_per_width=0.5,
            lmax=4,
            method="sparse",
        )
        coefficients = lya_rt.equilibrium_source_coefficients(
            first_order, self.cosmology
        )
        background = first_order["background"]
        denominator = float(first_order["denominator"])
        for delta_b, ground, velocity in ((0.2, -0.1, 0.3), (-0.4, 0.5, -0.2)):
            delta_feq = (
                coefficients.feq_density * delta_b
                + coefficients.feq_ground * ground
                + coefficients.feq_velocity * velocity
            )
            delta_fbar = (
                (delta_b + ground) * float(first_order["Abar"])
                + velocity * float(first_order["Bbar"])
                + delta_feq * float(first_order["Cbar"])
            ) / denominator
            direct = delta_feq - background.p_sc * delta_fbar
            reconstructed = (
                coefficients.alpha_bar * delta_b
                + coefficients.beta_bar * ground
                + coefficients.gamma_bar * velocity
            )
            self.assertAlmostEqual(direct, reconstructed, delta=1.0e-26)

    def test_recombination_response_is_finite(self):
        response = lya_rt.first_order_response(
            self.cosmology,
            self.ionization,
            1100.0,
            1.0e5,
            xmax=40.0,
            bins_per_width=0.5,
            lmax=4,
            method="sparse",
        )
        rates = lya_rt.recombination_response(response, self.cosmology)
        self.assertTrue(all(math.isfinite(value) for value in (
            rates.density_rate,
            rates.ionization_rate,
            rates.velocity_divergence_rate,
            rates.temperature_rate,
        )))
        self.assertLess(rates.density_rate, 0.0)
        self.assertLess(rates.ionization_rate, 0.0)
        self.assertGreater(rates.temperature_rate, 0.0)

    def test_free_electron_rate_has_short_wavelength_sign_and_limit(self):
        first_order = lya_rt.first_order_response(
            self.cosmology,
            self.ionization,
            1100.0,
            1.0e7,
            xmax=40.0,
            bins_per_width=0.5,
            lmax=4,
            method="sparse",
        )
        rates = lya_rt.recombination_response(first_order, self.cosmology)
        background = first_order["background"]
        density_limit = -(
            self.cosmology.hydrogen_density(1100.0)
            * background.xe**2
            * background.alpha_b
        )
        ionization_limit = -(
            2.0
            * self.cosmology.hydrogen_density(1100.0)
            * background.xe
            * background.alpha_b
            + 4.0 * background.f_equilibrium * background.beta_b
        )
        self.assertGreater(rates.peebles_factor, 0.99)
        self.assertLess(abs(rates.density_rate / density_limit - 1.0), 0.02)
        self.assertLess(abs(rates.ionization_rate / ionization_limit - 1.0), 0.02)


if __name__ == "__main__":
    unittest.main()
