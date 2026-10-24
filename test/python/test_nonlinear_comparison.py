from argparse import Namespace
from pathlib import Path
import tempfile
import tomllib

import numpy as np
import pytest

from tools.reference.nonlinear_comparison import case_settings, mode_at, physical_mode, write_config


def test_epoch_interpolation_preserves_complex_phase():
    rows = [dict(field="gas_delta",z=z,real=real,imag=imag)
            for z,real,imag in [(3200,2,4),(3000,4,8)]]
    middle = np.sqrt(3201*3001)-1
    assert mode_at(rows,"gas_delta",middle) == pytest.approx(3+6j)
    with pytest.raises(ValueError,match="missing"):
        mode_at(rows,"B_y",middle)
    with pytest.raises(ValueError,match="span"):
        mode_at(rows,"gas_delta",4000)


def test_velocity_and_particle_clocks_are_distinct():
    handoff = dict(guide_field_code=2,velocity_unit_cm_s_at_handoff=800000,zhandoff=3200)
    assert physical_mode(0.02j,"B_y",3200,handoff) == 0.01j
    assert physical_mode(1j,"gas_v_x",1600,handoff) == pytest.approx(800000j*np.sqrt(1601/3201))
    assert physical_mode(123j,"dm_v_x",1600,handoff) == 123j


def test_native_case_preserves_handoff_units_and_full_sources():
    handoff=dict(velocity_unit_cm_s_at_handoff=12345,
                 background=dict(xHII=0.9,pressure_code_mean=2.5))
    args=Namespace(hubble=0.674,resolution=32,backend="metal",cfl=0.2,zstart=3200,
        zend=3100,box_ckpc_h=1,field_pg=300,theta=np.pi/4,epsilon=0.025,
        outputs=9,response_table=None)
    settings=case_settings(handoff,Path("input-state"),args)
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)/"case.toml"
        write_config(p,settings)
        values=tomllib.loads(p.read_text())
    assert values["chemistry"]["velocity_unit_cms"] == 12345
    assert values["chemistry"]["ionized_fast_zmin"] == 0
    assert values["physics"]["gravity"] and values["physics"]["gas_gravity"]
    assert values["physics"]["photon_response"] == "moments_exponential"
    assert values["environment"]["MHD_CHEM_FALPHA"] == 0
    assert values["environment"]["MHD_RADIATION_FREE_STREAM_TOL"] == 0


def _args(**kw):
    base = dict(hubble=0.674,resolution=32,backend="cpu",cfl=0.2,zstart=3200,zend=3100,
        box_ckpc_h=1,field_pg=300,theta=np.pi/4,epsilon=0.025,outputs=9,response_table=None)
    return Namespace(**{**base,**kw})


def test_mhd_solver_option_sets_toml_and_drops_glm_projection():
    handoff=dict(velocity_unit_cm_s_at_handoff=1,background=dict(xHII=0.9,pressure_code_mean=2.5))
    glm=case_settings(handoff,Path("s"),_args())
    assert "mhd_solver" not in glm["run"] and glm["physics"]["glm_project_every"] == 8
    ct=case_settings(handoff,Path("s"),_args(mhd_solver="ct"))
    assert ct["run"]["mhd_solver"] == "ct" and "glm_project_every" not in ct["physics"]
    ct["run"].pop("mhd_solver"); ct["physics"]["glm_project_every"] = 8
    assert ct == glm
