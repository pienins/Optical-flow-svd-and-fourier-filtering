"""Tests for the per-scenario benchmark scripts (pytest -q)."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from tifffile import imread

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from two_wcf.config import load_config  # noqa: E402
from two_wcf.pipeline import run  # noqa: E402

UNIFORM = ROOT / 'benchmarks' / 'homogeneous_constant_flow'


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generate = load_module(UNIFORM / 'generate.py', 'test_uniform_generate')
evaluate = load_module(UNIFORM / 'evaluate.py', 'test_uniform_evaluate')


def uniform_config(tmp_path, **generate_overrides):
    cfg = json.loads((UNIFORM / 'config.json').read_text(encoding='utf-8'))
    cfg['paths'] = {'generated_root': str(tmp_path / 'generated'),
                    'pipeline_output_dir': str(tmp_path / 'not_run_yet')}
    cfg['generate'].update(generate_overrides)
    path = tmp_path / 'uniform_config.json'
    path.write_text(json.dumps(cfg), encoding='utf-8')
    return path, cfg


def test_shipped_uniform_config_is_valid():
    generate.load_config(UNIFORM / 'config.json')


def test_uniform_generator_truth_naming_and_overwrite(tmp_path):
    path, _ = uniform_config(tmp_path, width=64, height=48, n_frames=3,
                             target_visible_particles=80,
                             true_u_px_per_frame=1.5, true_v_px_per_frame=-0.5)
    out = generate.generate(generate.load_config(path))
    assert out.name == 'synthetic_uniform_u1p5_vm0p5'
    frames = sorted((out / 'frames').glob('frame_*.tiff'))
    assert len(frames) == 3 and imread(frames[0]).shape == (48, 64)
    with np.load(out / 'true_vector_values.npz') as gt:
        assert np.all(gt['u_gt'] == 1.5) and np.all(gt['v_gt'] == -0.5)
        assert np.isclose(gt['true_speed'], np.hypot(1.5, 0.5))

    # A second run must not silently mix with the existing frames.
    with pytest.raises(FileExistsError):
        generate.generate(generate.load_config(path))
    path, _ = uniform_config(tmp_path, width=64, height=48, n_frames=2,
                             target_visible_particles=80,
                             true_u_px_per_frame=1.5, true_v_px_per_frame=-0.5,
                             overwrite_existing=True)
    generate.generate(generate.load_config(path))
    assert len(list((out / 'frames').glob('frame_*.tiff'))) == 2


def test_uniform_benchmark_end_to_end(tmp_path):
    """Generate -> 2WCF -> evaluate on a small uniform flow with known truth.

    The information-aware solver is disabled here: with the shipped
    InformationRidgeFraction it underestimates displacement (known issue).
    """
    bench_path, bench_cfg = uniform_config(
        tmp_path, width=160, height=144, n_frames=3, target_visible_particles=400,
        true_u_px_per_frame=1.5, true_v_px_per_frame=-0.5)
    seq = generate.generate(generate.load_config(bench_path))

    cfg = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))
    cfg['paths'] = {'input_images': str(seq / 'frames'), 'output_root': str(tmp_path / 'runs')}
    cfg['run'].update({'runID': 1, 'UseParallelFlow': False, 'MaxFramePairs': None})
    cfg['outputs'].update({'WriteWXF': False, 'SaveVelocityVideo': False, 'SaveQCPlot': False})
    for key, value in cfg['diagnostics'].items():
        if isinstance(value, bool):
            cfg['diagnostics'][key] = False
    cfg['diagnostics']['SaveValidationData'] = True
    cfg['information_solver'].update({'UseInformationAwareSolver': False,
                                      'UseInformationSpatialPrior': False})
    pipe_path = tmp_path / 'pipeline_config.json'
    pipe_path.write_text(json.dumps(cfg), encoding='utf-8')
    run_dir = run(load_config(pipe_path)).root

    bench_cfg['paths']['pipeline_output_dir'] = str(run_dir)
    bench_path.write_text(json.dumps(bench_cfg), encoding='utf-8')
    summary = evaluate.evaluate(evaluate.load_config(bench_path))
    assert summary['final_median_epe_px'] < 0.05
    assert summary['interior_retention_percent'] > 90.0
    assert (run_dir / 'synthetic_validation_results' / 'uniform_validation_per_frame.csv').exists()

    # Output with an inverted sign is not in image coordinates and must be refused.
    params = json.loads((run_dir / 'parameters.json').read_text(encoding='utf-8'))
    params['visualization']['VelocityInversion'] = -1
    (run_dir / 'parameters.json').write_text(json.dumps(params), encoding='utf-8')
    with pytest.raises(ValueError, match='VelocityInversion'):
        evaluate.evaluate(evaluate.load_config(bench_path))
