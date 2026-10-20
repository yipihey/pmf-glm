import unittest

from tools.analysis import analyze_temporal_order as temporal


def rows(value, redshift=1000.0):
    return [
        {
            "field": field,
            "z": str(redshift),
            "real": str(value),
            "imag": "0",
        }
        for field in temporal.FIELDS
    ]


class TemporalOrderTests(unittest.TestCase):
    @staticmethod
    def manifest(level: int) -> dict[str, object]:
        divisor = 2**level
        return {
            "backend": "metal",
            "physics": "full",
            "coupling": "radiation",
            "resolution": 128,
            "zstart": 3200.0,
            "zend": 500.0,
            "initial_state_fingerprint_sha256": "initial",
            "solver_source_fingerprint_sha256": "source",
            "environment": {
                "MHD_CFL": str(0.1 / divisor),
                "MHD_RADIATION_NONLINEAR_CFL": str(0.1 / divisor),
                "MHD_RADIATION_MAX_DLNA": str(0.00025 / divisor),
            },
        }

    def test_ladder_requires_all_controls_to_refine(self) -> None:
        manifests = [self.manifest(level) for level in range(3)]
        controls = temporal.validate_temporal_ladder(manifests)
        self.assertEqual(controls["MHD_CFL"], [0.1, 0.05, 0.025])
        manifests[1]["environment"]["MHD_RADIATION_MAX_DLNA"] = "0.00025"
        with self.assertRaises(ValueError):
            temporal.validate_temporal_ladder(manifests)

    def test_ladder_requires_matched_initial_state(self) -> None:
        manifests = [self.manifest(level) for level in range(3)]
        manifests[2]["initial_state_fingerprint_sha256"] = "different"
        with self.assertRaises(ValueError):
            temporal.validate_temporal_ladder(manifests)

    def test_second_order_sequence_passes(self) -> None:
        coarse = rows(1.04)
        medium = rows(1.01)
        fine = rows(1.0025)
        result, failures = temporal.compare_levels(
            coarse, medium, fine, [1000.0], error_tolerance=0.01
        )
        self.assertFalse(failures)
        self.assertTrue(all(item["status"] == "PASS" for item in result))
        self.assertTrue(all(abs(item["observed_order"] - 2.0) < 1.0e-12 for item in result))

    def test_noncontracting_material_error_fails(self) -> None:
        coarse = rows(1.005)
        medium = rows(1.0)
        fine = rows(1.006)
        _, failures = temporal.compare_levels(coarse, medium, fine, [1000.0])
        self.assertTrue(failures)
        self.assertTrue(all(item["contraction_gated"] for item in failures))

    def test_all_levels_must_resolve_signal(self) -> None:
        coarse = rows(1.0)
        medium = rows(1.0)
        fine = rows(1.0)
        for level, value in zip((coarse, medium, fine), (2.0e-8, 0.5e-8, 1.5e-8)):
            next(row for row in level if row["field"] == "x_HII")["real"] = str(value)
        result, failures = temporal.compare_levels(coarse, medium, fine, [1000.0])
        self.assertFalse(failures)
        ionization = next(item for item in result if item["field"] == "x_HII")
        self.assertEqual(ionization["status"], "BELOW_FLOOR")

    def test_requires_exact_output_redshift(self) -> None:
        with self.assertRaises(ValueError):
            temporal.exact_mode_map(rows(1.0, redshift=999.0), 1000.0)


if __name__ == "__main__":
    unittest.main()
