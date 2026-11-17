"""All persistent output operations and run-folder management live here.

Plotting modules build figures but never decide output filenames or write files.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import pickle
import shutil

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import wolframclient.serializers as wxf
except ImportError:
    wxf = None

VALIDATION_MAP = {
    'x': 'validation_x_all', 'y': 'validation_y_all',
    'u_raw': 'validation_u_raw', 'v_raw': 'validation_v_raw',
    'u_processed': 'validation_u_processed', 'v_processed': 'validation_v_processed',
    'u_output': 'validation_u_output', 'v_output': 'validation_v_output',
    'valid_mask': 'validation_valid_mask',
    'consistency_mask': 'validation_consistency_mask',
    'consistency_error': 'validation_consistency_error',
    'confidence': 'validation_confidence',
}


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @property
    def pkl(self): return self.root / 'pkl'
    @property
    def wxf(self): return self.root / 'wxf'
    @property
    def validation(self): return self.root / 'validation'
    @property
    def diagnostic_maps(self): return self.root / 'diagnostic_maps'
    @property
    def rejected_cache(self): return self.root / '_video_rejected_cache'
    @property
    def explicit_diagnostics(self): return self.root / 'explicit_flow_diagnostics'
    @property
    def pyramid_visualization(self): return self.root / 'pyramid_visualization'


def create_run_paths(cfg) -> RunPaths:
    source = cfg.input_images
    if not source.is_dir():
        raise FileNotFoundError(f'Input TIFF folder does not exist: {source}')
    if cfg.WriteWXF and wxf is None:
        raise ImportError('WriteWXF=true but wolframclient is not installed. '
                          'Install wolframclient or set outputs.WriteWXF=false.')
    name = source.parent.name if source.name.lower() == 'frames' else source.name
    run_folder = cfg.output_root / f'{name}_run_{cfg.runID}'
    if run_folder.exists() and any(run_folder.iterdir()):
        if not cfg.OverwriteExistingRun:
            raise FileExistsError(f'Output folder is not empty: {run_folder}\n'
                                  'Change run.runID, or deliberately set '
                                  'outputs.OverwriteExistingRun=true.')
        # A user-authorized rerun clears its OWN computed run folder to avoid
        # mixing stale pair files from previous runs with the new outputs.
        if run_folder.resolve() == cfg.output_root.resolve():
            raise RuntimeError('Refusing to delete the output root directory.')
        shutil.rmtree(run_folder)
    paths=RunPaths(run_folder)
    for folder in (paths.root,paths.pkl):
        folder.mkdir(parents=True,exist_ok=True)
    if cfg.WriteWXF: paths.wxf.mkdir(parents=True,exist_ok=True)
    if cfg.SaveValidationData: paths.validation.mkdir(parents=True,exist_ok=True)
    if cfg.SaveDiagnosticMaps: paths.diagnostic_maps.mkdir(parents=True,exist_ok=True)
    if cfg.PlotRejectedPoints and cfg.SaveVelocityVideo:
        paths.rejected_cache.mkdir(parents=True,exist_ok=True)
    write_json(paths.root/'parameters.json',cfg.snapshot())
    # Flat convenience view for compatibility with previous CSV analysis scripts
    write_csv(paths.root/'parameters.csv',[(name,str(value)) for name,value in cfg._flat.items()],
              header=False)
    return paths


def write_json(path, data):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8') as fh:
        json.dump(data,fh,indent=2,ensure_ascii=False)


def write_text(path: Path, text: str):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(text,encoding='utf-8')


def write_csv(path, records, header=True):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    table=pd.DataFrame(records)
    table.to_csv(path,index=False,header=header)


def write_npz(path, **arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**arrays)


def write_png(path, img):
    """Write already-rendered image data, without changing image values."""
    import imageio.v2 as imageio
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    imageio.imwrite(path,img)


def save_figure(fig, path, dpi=180, show=False, **kwargs):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    try:
        fig.savefig(path,dpi=dpi,**kwargs)
        if show:plt.show()
    finally:
        plt.close(fig)


def save_frame_field(paths: RunPaths, result, *, write_wxf: bool):
    i=result['index']
    field={key:np.asarray(result[key]).ravel() for key in ('x','y','u','v')}
    with (paths.pkl/f'velocity_field_{i:05d}.pkl').open('wb') as fh:
        pickle.dump(field,fh,pickle.HIGHEST_PROTOCOL)
    if write_wxf:
        wxf.export(field,str(paths.wxf/f'velocity_field_{i:05d}.wxf'),target_format='wxf')


def read_frame_field(paths:RunPaths,index:int):
    with (paths.pkl/f'velocity_field_{index:05d}.pkl').open('rb') as fh:
        return pickle.load(fh)


def save_validation(paths:RunPaths,result):
    index=result['index']
    write_npz(paths.validation/f'validation_{index:05d}.npz',
              **{name:result[key] for name,key in VALIDATION_MAP.items()})
    discard_validation(result)


def discard_validation(result):
    for name in VALIDATION_MAP.values():result.pop(name,None)


def save_diagnostic_maps(paths:RunPaths,result):
    i=result['index']
    maps={'local_mad':result.pop('local_mad_map'),
          'local_sigma':result.pop('local_sigma_map'),
          'confidence':result.pop('confidence_map')}
    write_npz(paths.diagnostic_maps/f'diagnostic_maps_{i:05d}.npz',**maps)
    return {name:finite_min_max(array) for name,array in maps.items()}


def finite_min_max(values):
    v=values[np.isfinite(values)]
    if v.size==0:return (float('inf'),float('-inf'))
    return float(np.min(v)),float(np.max(v))


def cache_rejected(paths:RunPaths,result):
    i=result['index']
    write_npz(paths.rejected_cache/f'rejected_{i:05d}.npz',
              x_rejected=result['x_rejected'],y_rejected=result['y_rejected'])


def read_rejected(paths:RunPaths,index):
    path=paths.rejected_cache/f'rejected_{index:05d}.npz'
    if not path.exists():return np.array([],dtype=float), np.array([],dtype=float)
    with np.load(path) as data:
        return data['x_rejected'],data['y_rejected']


def remove_rejected_cache(paths):
    if paths.rejected_cache.exists():shutil.rmtree(paths.rejected_cache)


class VideoWriter:
    """Owns the only MP4 encoder: accept RGB frames from visualization."""
    def __init__(self,path,fps):
        self.path=Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.fps=fps
        self.writer=None
        self.size=None

    def __enter__(self):return self

    def write(self,rgb):
        rgb=np.asarray(rgb)
        if rgb.ndim!=3 or rgb.shape[-1]!=3:
            raise ValueError('Expected RGB frame with shape (height,width,3).')
        if self.writer is None:
            self.size=(rgb.shape[1],rgb.shape[0])
            self.writer=cv2.VideoWriter(str(self.path),cv2.VideoWriter_fourcc(*'mp4v'),
                                        self.fps,self.size)
            if not self.writer.isOpened():
                raise IOError(f'Failed to open MP4 writer: {self.path}')
        bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
        if (bgr.shape[1],bgr.shape[0])!=self.size:
            bgr=cv2.resize(bgr,self.size)
        self.writer.write(bgr)

    def __exit__(self,exc_type,exc,tb):
        if self.writer is not None:self.writer.release()
