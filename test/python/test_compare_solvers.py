import math

import pytest

from tools.reference.compare_solvers import differences, summarize


def _row(z,field,re,im,status="PASS",floor=1e-7):
    return dict(z=z,field=field,real=re,imag=im,signal_floor=floor,status=status)


def test_differences_and_summary():
    glm=[_row(3100,"B_y",1,0),_row(3100,"gas_v_x",1,0),_row(3100,"dm_delta",1e-9,0,"BELOW_FLOOR")]
    ct=[_row(3100,"B_y",1.02*math.cos(0.01),1.02*math.sin(0.01)),_row(3100,"gas_v_x",1,0),
        _row(3100,"dm_delta",5,0,"BELOW_FLOOR")]
    rows=differences(glm,ct)
    by={r["field"]:r for r in rows}
    assert by["B_y"]["relative_amplitude_difference"] == pytest.approx(0.02)
    assert by["B_y"]["phase_difference_rad"] == pytest.approx(0.01)
    assert math.isnan(by["dm_delta"]["relative_amplitude_difference"])
    gate=dict(status="PASS",amplitude_tolerance=0.05,phase_tolerance_rad=0.03)
    s=summarize(rows,gate,dict(gate,mhd_solver="ct"))
    assert s["families"]["magnetic"]["max_abs_relative_amplitude_difference"] == pytest.approx(0.02)
    assert s["families"]["velocity"]["max_abs_phase_difference_rad"] == 0
    assert "dark_matter" not in s["families"] and s["within_gate_tolerances"]


def test_epoch_mismatch_rejected_and_tiny_offset_paired():
    with pytest.raises(ValueError):
        differences([_row(1,"B_y",1,0)],[_row(2,"B_y",1,0)])
    with pytest.raises(ValueError):
        differences([_row(1,"B_y",1,0)],[_row(1,"gas_delta",1,0)])
    assert differences([_row(3199.92467,"B_y",1,0)],[_row(3199.92466,"B_y",1,0)])[0]["z_ct"] == 3199.92466
