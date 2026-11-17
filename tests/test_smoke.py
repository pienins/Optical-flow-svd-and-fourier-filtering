"""Small self-contained regression/smoke tests (pytest -q)."""
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest
from tifffile import imwrite

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from two_wcf.config import load_config
from two_wcf.consistency import compute_flow_pair
from two_wcf.pipeline import run


@pytest.fixture()
def experiment(tmp_path):
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / 'config.json').read_text(encoding='utf-8'))
    source = tmp_path / 'synthetic_uniform_u1_v0' / 'frames'
    source.mkdir(parents=True)
    cfg['paths'] = {'input_images': str(source), 'output_root': str(tmp_path / 'output')}
    cfg['run'].update({'runID': 1, 'UseParallelFlow': True,
                       'MaxFlowWorkers': 2, 'MaxFramePairs': 2})
    cfg['farneback'].update({'PyramidLevels': 2, 'WindowSize': 9,
                              'FarnebackPyramidMinSize': 12,
                              'GaussKernelStdev': 1.2})
    cfg['visualization'].update({'VideoDPI': 60, 'ArrowDensityStep': 5})
    cfg['diagnostics'].update({
        'SavePyramidVisualizations': False,
        'PrintPyramidWindowResolution': False,
        'SavePyramidVelocityPlots': False,
        'SaveStreamlinePlots': False,
        'SaveInformationMaps': False,
        'SaveDiagnosticMaps': False,
    })
    cfg['outputs']['WriteWXF'] = False
    rng = np.random.default_rng(123)
    base = rng.integers(20, 230, (48, 48), dtype=np.uint8)
    images = []
    for index in range(3):
        frame = cv2.warpAffine(base,
                               np.float32([[1, 0, index], [0, 1, -.4 * index]]),
                               (48, 48), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT_101)
        imwrite(source / f'frame_{index:05d}.tiff', frame)
        images.append(frame.astype(np.float32))
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(cfg), encoding='utf-8')
    return cfg, path, images, tmp_path


def test_config_rejects_invalid_kernel(experiment):
    cfg, path, _, _ = experiment
    cfg['farneback']['PolyFitKernelSize'] = 4
    path.write_text(json.dumps(cfg), encoding='utf-8')
    with pytest.raises(ValueError, match='PolyFitKernelSize'):
        load_config(path)


@pytest.mark.parametrize('mode', ['image', 'coefficients'])
@pytest.mark.parametrize('information', [True, False])
def test_solver_modes_return_valid_arrays(experiment, mode, information):
    cfg, path, images, _ = experiment
    cfg['farneback']['FarnebackWarpMode'] = mode
    cfg['information_solver']['UseInformationAwareSolver'] = information
    path.write_text(json.dumps(cfg), encoding='utf-8')
    parsed = load_config(path)
    result = compute_flow_pair(0, images, 48, 48, 0.0, 255.0, cfg=parsed)
    assert np.isfinite(result['validation_u_raw']).all()
    assert result['validation_valid_mask'].dtype == bool
    assert result['validation_x_all'].shape == result['validation_u_output'].shape
    assert len(result['x']) == result['retained_vectors']


def test_end_to_end_parallel_streaming(experiment):
    _, path, _, _ = experiment
    paths = run(load_config(path))
    assert (paths.root/'run_manifest.json').exists()
    assert (paths.root/'flow_qc_metrics.csv').exists()
    assert (paths.root/'velocity_field.mp4').stat().st_size > 0
    assert len(list(paths.pkl.glob('velocity_field_*.pkl'))) == 2
    assert len(list(paths.validation.glob('validation_*.npz'))) == 2
    with np.load(paths.validation/'validation_00000.npz') as data:
        assert {'x', 'y', 'u_raw', 'v_raw', 'u_processed', 'v_processed',
                'u_output', 'v_output', 'valid_mask', 'consistency_mask',
                'consistency_error', 'confidence'} <= set(data.files)
    with pytest.raises(FileExistsError):
        run(load_config(path))


def test_local_sigma_and_svd_modes(experiment):
    cfg, path, images, _ = experiment
    cfg['preprocessing'].update({'USE_SVD': True, 'KillIDs': [0]})
    cfg['run']['UseParallelFlow'] = False
    cfg['confidence'].update({'UseLocalConfidenceSigma': True,
                              'UseConfidenceSmoothing': True,
                              'LocalSigmaKernelSize': 9})
    cfg['diagnostics']['SaveDiagnosticMaps'] = True
    cfg['outputs']['SaveVelocityVideo'] = False
    path.write_text(json.dumps(cfg), encoding='utf-8')
    paths = run(load_config(path))
    assert (paths.diagnostic_maps/'diagnostic_maps_00000.npz').exists()
    assert (paths.root/'local_sigma.mp4').stat().st_size > 0
    assert not (paths.root/'velocity_field.mp4').exists()
    with np.load(paths.validation/'validation_00000.npz') as data:
        assert np.isfinite(data['confidence']).all()
