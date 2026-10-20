import math
from argparse import Namespace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.reference import build_p1_multimode_handoff as builder
from tools.reference import compare_multimode_to_oracle as comparator


class _Result:
    def __init__(self, fields):
        self.fields = fields

    def field(self, name):
        return self.fields[name]


class MultimodeTests(unittest.TestCase):
    def test_component_generator_supports_current_single_mode_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = Namespace(output=Path(directory), resolution=8, zstart=3300., zhandoff=3299.,
                             box_ckpc_h=1., hubble=0.674, field_pg=30., theta=math.pi / 4,
                             velocity_unit=8e5, max_dlna=0.001, response_table=None,
                             nonlocal_scale=1., spectral_epsilon=-0.1, target_gas_delta_rms=1e-8)
            component = builder.component_args(args, args.output / "one", 1, 0.001)
            self.assertEqual(component.amplitude_sign, 1)
            components = builder.build_components(args, {1: 0.001, 2: 0.0005}, "test")
            result = builder.combine(args, components)
            self.assertEqual(len(result["modes"]), 2)
            self.assertGreater(result["rho_min"], 0)
            self.assertTrue((args.output / "arrays.tsv").is_file())

    def test_mode_parser_sorts_and_deduplicates(self) -> None:
        self.assertEqual(builder.parse_modes("4,1,2,2"), (1, 2, 4))

    def test_p1_state_uses_declared_field_order(self) -> None:
        names = (
            "b_x_over_B0", "b_y_over_B0", "delta_b",
            "v_b_x", "v_b_y", "v_b_z", "delta_gamma",
            "v_gamma_x", "v_gamma_y", "v_gamma_z",
            "delta_dm", "v_dm_z", "delta_xe", "delta_temperature",
        )
        mode = {
            "mode_state": {
                name: {"real": float(index), "imag": -float(index)}
                for index, name in enumerate(names)
            }
        }
        state = comparator.p1_state(mode)
        np.testing.assert_array_equal(
            state, np.asarray([complex(index, -index) for index in range(14)])
        )

    def test_vector_power_sums_components_and_converts_velocity(self) -> None:
        result = _Result({
            "b_x_over_B0": np.asarray([3.0 + 0.0j]),
            "b_y_over_B0": np.asarray([4.0 + 0.0j]),
            "v_gamma_x": np.asarray([3.0 + 0.0j]),
            "v_gamma_y": np.asarray([4.0 + 0.0j]),
            "v_gamma_z": np.asarray([0.0 + 0.0j]),
        })
        self.assertEqual(
            comparator.expected_power(result, "B_vector", 0, 10.0), 25.0
        )
        self.assertTrue(math.isclose(
            comparator.expected_power(result, "photon_velocity", 0, 10.0),
            0.25,
        ))

    def test_nearest_spectral_bin_uses_initial_epoch(self) -> None:
        rows = [
            {"field": "B_vector", "z": "3200", "kbin": "2", "k": "8", "Nmodes": "4"},
            {"field": "B_vector", "z": "3200", "kbin": "3", "k": "15", "Nmodes": "8"},
            {"field": "B_vector", "z": "500", "kbin": "8", "k": "12", "Nmodes": "4"},
        ]
        self.assertEqual(
            comparator.nearest_kbin(rows, "B_vector", 3200.0, 13.0), 3
        )


if __name__ == "__main__":
    unittest.main()
