import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from tools.analysis import analyze_production as analysis
from tools.analysis import render_density_frames as frames
from tools.reference import compare_reference, matched_scenario, run_reference


FIXTURES = Path(__file__).parent / "fixtures"


class ProductionAnalysisTests(unittest.TestCase):
    def spectra(self):
        return analysis.read_tsv(FIXTURES / "spectra.tsv", ("field", "P", "Nmodes"))

    def test_angular_average_is_mode_weighted_and_initial_record_retained(self):
        rows = analysis.isotropic_spectra(self.spectra(), 1)
        initial = next(row for row in rows if row["field"] == "gas_delta" and row["cycle"] == 0)
        self.assertEqual(initial["P_field_code2_mpc3"], 2.5)
        self.assertEqual(initial["Nmodes"], 8)
        self.assertEqual(initial["measured"], 0)

    def test_coordinate_conversion_preserves_dimensionless_power(self):
        scale = analysis.length_in_mpc("box", 0.674, 1.0, 1.0)
        raw = analysis.isotropic_spectra(self.spectra(), 1)
        converted = analysis.isotropic_spectra(self.spectra(), scale)
        self.assertAlmostEqual(converted[0]["k_comoving_mpc_inv"], raw[0]["k_comoving_mpc_inv"] * 674)
        for left, right in zip(raw, converted):
            self.assertAlmostEqual(left["Delta2_field_code2"], right["Delta2_field_code2"])
        self.assertAlmostEqual(analysis.length_in_mpc("box", 0.674, 1.0, 2.0), scale / 2)

    def test_isotropic_mu_one_schema_and_distinct_force_powers(self):
        rows = analysis.isotropic_spectra(self.spectra(), 1)
        powers = {row["field"]: row["P_field_code2_mpc3"] for row in rows}
        self.assertEqual(powers["B_pressure_force"], 9)
        self.assertEqual(powers["B_tension_force"], 4)
        self.assertEqual(powers["B_lorentz_force"], 1)

    def test_populated_nan_and_duplicate_bins_are_rejected(self):
        rows = self.spectra()
        with self.assertRaises(ValueError):
            analysis.isotropic_spectra(rows + [rows[0]], 1)
        rows[0]["P"] = "NaN"
        with self.assertRaises(ValueError):
            analysis.isotropic_spectra(rows, 1)

    def test_pdf_integrates_in_log_density_not_density(self):
        rows = analysis.read_tsv(FIXTURES / "density_pdf.tsv", ("pdf",))
        for integral in analysis.pdf_integrals(rows):
            self.assertEqual(integral["integral_per_log10_rho"], 1)
            self.assertEqual(integral["included_cell_count"], 100)

    def test_explicit_units_and_missing_columns_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                analysis.analyze(Path(directory), spectra=FIXTURES / "spectra.tsv")
            with self.assertRaises(ValueError):
                analysis.read_tsv(FIXTURES / "history.tsv", ("missing",))
        with self.assertRaises(ValueError):
            analysis.length_in_mpc("box", 0.674, None, 1)

    def test_complete_figure_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            summary = analysis.analyze(output, history=FIXTURES / "history.tsv",
                                       spectra=FIXTURES / "spectra.tsv", pdf=FIXTURES / "density_pdf.tsv",
                                       length_unit="box", box_ckpc_h=1)
            self.assertEqual(summary["history_rows"], 2)
            self.assertEqual(len(summary["per_epoch_spectra"]), 2)
            self.assertIsNone(summary["history_finite_ranges"]["fH2_mean"])
            self.assertEqual(json.loads((output / "analysis.json").read_text()), summary)
            for stem in ("history", "spectra", "density_pdf"):
                self.assertTrue((output / (stem + ".pdf")).read_bytes().startswith(b"%PDF"))
                self.assertTrue((output / (stem + ".png")).read_bytes().startswith(b"\x89PNG"))

    def test_run_discovery_units_and_cli_output_do_not_overwrite_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            run.mkdir()
            for source, destination in (("spectra.tsv", "spectra.tsv"), ("metadata.toml", "metadata.toml")):
                shutil.copyfile(FIXTURES / source, run / destination)
            original = (run / "metadata.toml").read_bytes()
            output = Path(directory) / "plots"
            process = subprocess.run([sys.executable, analysis.__file__, str(run), "--output", str(output)],
                                     cwd=directory, capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            summary = json.loads((output / "analysis.json").read_text())
            self.assertAlmostEqual(summary["spectral_units"]["mpc_per_coordinate_unit"], 0.001 / 0.674)
            self.assertEqual(summary["run"]["status"], "SYNTHETIC_TEST_FIXTURE")
            self.assertEqual((run / "metadata.toml").read_bytes(), original)
            with self.assertRaises(ValueError):
                analysis.analyze(run, run=run)
            with self.assertRaises(ValueError):
                analysis.discover_run(Path(directory) / "empty")

    def test_frames_use_column_major_layout_and_render_maximum_not_column_density(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = np.arange(1, 17, dtype="<f4").reshape((4, 4), order="F")
            values.ravel(order="F").tofile(root / "slice.f32")
            values.ravel(order="F").tofile(root / "max.f32")
            np.testing.assert_array_equal(frames.load_frame(root / "slice.f32", 4), values)
            (root / "frames.tsv").write_text("cycle\tz\tN\tslice_f32\tmaxproj_f32\tlayout\n"
                                            "1\t1100\t4\tslice.f32\tmax.f32\tjulia-column-major\n")
            records = frames.render(root, root / "plots", box_ckpc_h=1)
            self.assertEqual(len(records), 1)
            self.assertTrue((root / "plots" / (records[0]["figure_stem"] + ".png")).is_file())
            with self.assertRaises(ValueError):
                frames.load_frame(root / "slice.f32", 3)


class PortableReferenceTests(unittest.TestCase):
    def test_extraction_hashes_match_without_accessing_original_source(self):
        root = Path(__file__).resolve().parents[2]
        manifest = json.loads((root / "tools/reference/provenance.json").read_text())
        self.assertEqual(len(manifest["records"]), 18)
        for entry in manifest["records"]:
            self.assertEqual(hashlib.sha256((root / entry["destination_path"]).read_bytes()).hexdigest(),
                             entry["extracted_sha256"])
            self.assertEqual(len(entry["source_sha256"]), 64)

    def test_reference_json_is_deterministic_and_finite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            run_reference.generate(output, zstart=3200, zend=3199, points=2, with_lya=False)
            first = (output / "reference.json").read_bytes()
            run_reference.generate(output, zstart=3200, zend=3199, points=2, with_lya=False)
            self.assertEqual(first, (output / "reference.json").read_bytes())
            self.assertIn("not nonlinear 3D", json.loads(first)["scope"])

    def test_matched_scenario_recovers_fourier_amplitude_and_units(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            payload = matched_scenario.generate(output, resolution=16, kmode=2, zend=3199)
            self.assertEqual(payload["status"], "REFERENCE_ONLY")
            self.assertAlmostEqual(payload["k_comoving_mpc_inv"], 2 * math.pi * 2 * 674)
            with np.load(output / "initial_fields.npz") as arrays:
                phase = 2 * math.pi * 2 * arrays["coordinate"]
                recovered = 2 * np.mean(arrays["b_y_over_B0"] * np.exp(-1j * phase))
                self.assertAlmostEqual(recovered.real, payload["epsilon"])
                self.assertAlmostEqual(recovered.imag, 0)
            with (output / "reference_modes.tsv").open() as stream:
                self.assertGreater(len(list(csv.DictReader(stream, delimiter="\t"))), 20)

    def test_comparison_detects_phase_and_never_matches_nearby_epoch(self):
        reference = {("delta_b", 1000.0): 1 + 1j, ("delta_b", 1100.0): 0j}
        rows, summary = compare_reference.compare(reference, reference)
        self.assertEqual(summary["status"], "PASS")
        self.assertIsNone(rows[-1]["phase_error_rad"])
        candidate = dict(reference)
        candidate["delta_b", 1000.0] *= 1j
        self.assertEqual(compare_reference.compare(reference, candidate)[1]["status"], "FAIL")
        with self.assertRaises(ValueError):
            compare_reference.compare(reference, {("delta_b", 1000.1): 1 + 1j})


if __name__ == "__main__":
    unittest.main()
