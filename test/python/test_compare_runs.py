import json
import math

import pytest

from tools.analysis.compare_runs import compare_runs, main, pair_redshifts


def _write_run(path, solver, scale, z=(3200.0, 3199.9)):
    path.mkdir()
    (path/"metadata.toml").write_text(f'status = "COMPLETED"\nmhd_solver = "{solver}"\n')
    cols = "cycle\tmeasured\tz\ta\tbrms\tdelta_b_rms\tdivBdx_over_brms\tdivb_face_dx_over_brms\n"
    rows = "".join(f"{i}\t{i}\t{zz}\t0.1\t{2*scale}\t{0.0 if i == 0 else 4*scale}\t{1e-3*scale}\t"
                   f"{'1e-7' if solver == 'ct' else 'NaN'}\n" for i, zz in enumerate(z))
    (path/"history.tsv").write_text(cols+rows)
    head = "field\tcycle\tmeasured\tz\ta\tkbin\tmubin\tk\tmu_lo\tmu_hi\tP\tNmodes\n"
    lines = []
    for zz in z:
        lines.append(f"B_vector\t0\t0\t{zz}\t0.1\t1\t1\tNaN\t0\t1\tNaN\t0\n")
        lines.append(f"B_vector\t0\t0\t{zz}\t0.1\t2\t1\t8.0\t0\t1\t{3*scale}\t26\n")
        lines.append(f"gas_delta\t0\t0\t{zz}\t0.1\t2\t1\t8.0\t0\t1\t0.0\t26\n")
    (path/"spectra.tsv").write_text(head+"".join(lines))


def test_pairing_is_by_redshift_not_index():
    assert pair_redshifts([3.0, 2.0], [2.0005, 3.0], 1e-3) == [(0, 1), (1, 0)]
    assert pair_redshifts([3.0], [3.5], 1e-3) == []


def test_compare_reports_relative_differences(tmp_path):
    _write_run(tmp_path/"a", "glm", 1.0)
    _write_run(tmp_path/"b", "ct", 1.02)
    summary, hist, spec = compare_runs(tmp_path/"a", tmp_path/"b")
    assert summary["a"]["mhd_solver"] == "glm" and summary["b"]["mhd_solver"] == "ct"
    assert summary["history"]["common_epochs"] == 2
    assert summary["history"]["columns"]["brms"]["max_abs"] == pytest.approx(0.02)
    # zero reference value (first delta_b_rms) is skipped, not reported as zero
    assert summary["history"]["columns"]["delta_b_rms"]["n"] == 1
    assert math.isnan(hist[0]["delta_b_rms"])
    assert "divb_face_dx_over_brms" not in summary["history"]["columns"]
    assert summary["history"]["divergence_b"]["divb_face_dx_over_brms"]["last"] == pytest.approx(1e-7)
    assert summary["spectra"]["fields"]["B_vector"]["max_abs"] == pytest.approx(0.02)
    assert "gas_delta" not in summary["spectra"]["fields"]  # all-zero reference power: skipped
    assert all(r["kbin"] == "2" for r in spec)


def test_cli_writes_outputs_and_refuses_overwrite(tmp_path, capsys):
    _write_run(tmp_path/"a", "glm", 1.0)
    _write_run(tmp_path/"b", "ct", 1.0)
    assert main([str(tmp_path/"a"), str(tmp_path/"b")]) == 0
    out = tmp_path/"b"/"run_comparison"
    assert (out/"history_differences.tsv").is_file() and (out/"spectra_differences.tsv").is_file()
    assert json.loads((out/"run_comparison.json").read_text())["history"]["common_epochs"] == 2
    with pytest.raises(FileExistsError):
        main([str(tmp_path/"a"), str(tmp_path/"b")])


def test_no_common_epochs_is_an_error(tmp_path):
    _write_run(tmp_path/"a", "glm", 1.0)
    _write_run(tmp_path/"b", "ct", 1.0, z=(100.0, 99.0))
    with pytest.raises(ValueError):
        compare_runs(tmp_path/"a", tmp_path/"b")
