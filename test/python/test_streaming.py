from pathlib import Path
import json

import numpy as np
from PIL import Image
import pytest

from tools.reference.streaming import growth
from tools.analysis.render_density_frames import render, animate


def test_matter_growing_mode_and_transverse_streaming():
    a, y = growth()
    np.testing.assert_allclose(y[0], a/a[0], rtol=2e-9)
    np.testing.assert_allclose(y[2], a/a[0], rtol=2e-9)
    _, transverse = growth(vbc_kms=(0, 30, 0))
    np.testing.assert_allclose(y, transverse, rtol=1e-12)


def test_streaming_phase_reversal_and_dm_growth_suppression():
    _, zero = growth(k_mpc=(1000, 0, 0))
    _, plus = growth(k_mpc=(1000, 0, 0), vbc_kms=(30, 0, 0))
    _, minus = growth(k_mpc=(1000, 0, 0), vbc_kms=(-30, 0, 0))
    np.testing.assert_allclose(plus, minus.conj(), rtol=2e-9, atol=1e-10)
    assert abs(plus[0, -1]) < 0.9*abs(zero[0, -1])
    assert abs(plus[2, -1]) < abs(zero[2, -1])
    assert abs(plus[0, -1].imag) > 1e-3


def test_gif_uses_manifest_order_and_preserves_epochs_and_hashes(tmp_path: Path):
    index = 'cycle\tz\tN\tslice_f32\tmaxproj_f32\tlayout\n'
    for cycle, z, value in ((2, 900.25, 2), (0, 1100.5, 1)):
        np.full(16, value, dtype='<f4').tofile(tmp_path/f'{cycle}.f32')
        index += f'{cycle}\t{z}\t4\t{cycle}.f32\t{cycle}.f32\tjulia-column-major\n'
    (tmp_path/'frames.tsv').write_text(index)
    out = tmp_path/'plots'
    records = render(tmp_path, out)
    assert [r['cycle'] for r in records] == [0, 2]
    animate(out, records, out/'density.gif', fps=5)
    with Image.open(out/'density.gif') as image:
        assert image.n_frames == 2
        assert image.info['duration'] == 200
    summary = json.loads((out/'frames.json').read_text())
    assert summary['log10_limits'] == [-2, 2]
    assert summary['frames'][0]['z'] == 1100.5
    assert len(summary['frames'][0]['payload_sha256']['slice_f32']) == 64
    with pytest.raises(ValueError):
        animate(out, records, out/'bad.gif', fps=0)
    (tmp_path/'frames.tsv').write_text(index+index.splitlines()[-1]+'\n')
    with pytest.raises(ValueError, match='duplicate'):
        render(tmp_path, out)
