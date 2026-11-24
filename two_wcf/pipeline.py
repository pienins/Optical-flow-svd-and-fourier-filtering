"""Workflow orchestration only: load -> optional SVD -> compute/persist -> render.

The in-flight queue remains bounded by worker count; no list of dense flow
fields is retained. Rendering is a second pass from disk-backed records.
"""
from __future__ import annotations
import os
import time
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import numpy as np
from tqdm import tqdm
from .image_io import svd_filter_images,load_images_without_svd
from .consistency import compute_flow_pair
from .diagnostics import (save_pyramid_window_table,save_pyramid_image_diagnostics,
                          save_explicit_flow_diagnostics)
from .outputs import (create_run_paths,save_frame_field,save_validation,
                      discard_validation,save_diagnostic_maps,cache_rejected,
                      write_csv,save_figure,write_json)
from .plots import build_qc_figure
from .videos import render_velocity_video,render_diagnostic_videos

QC_KEYS=(
    'median_err','mad_err','mom_err','thr_frame','relative_consistency_alpha',
    'effective_limit_median','effective_limit_p10','effective_limit_p90',
    'sigma_frame','total_vectors','retained_vectors','rejected_vectors',
    'consistency_rejected_vectors','low_confidence_vectors','retention_fraction',
    'rejection_fraction',
)


def _process_pairs(images,paths,lo,hi,*,cfg):
    indices=list(range(min(len(images)-1,cfg.MaxFramePairs))) if cfg.MaxFramePairs is not None else list(range(len(images)-1))
    if not indices:raise ValueError('Need at least two frames to calculate optical flow.')
    h,w=images[0].shape
    qc=[]
    ranges={'local_mad':[np.inf,-np.inf],
            'local_sigma':[np.inf,-np.inf],
            'confidence':[0.0,1.0]}
    speed_min,speed_max=np.inf,-np.inf

    def consume(result):
        nonlocal speed_min,speed_max
        speeds=np.hypot(np.asarray(result['u'],dtype=float),
                        np.asarray(result['v'],dtype=float))
        values=speeds[np.isfinite(speeds)]
        if values.size:
            speed_min=min(speed_min,float(np.min(values)))
            speed_max=max(speed_max,float(np.max(values)))
        if cfg.SaveDiagnosticMaps:
            observed=save_diagnostic_maps(paths,result)
            for name,(a,b) in observed.items():
                ranges[name][0]=min(ranges[name][0],a)
                ranges[name][1]=max(ranges[name][1],b)
        if cfg.SaveValidationData:save_validation(paths,result)
        else:discard_validation(result)
        save_frame_field(paths,result,write_wxf=cfg.WriteWXF)
        if cfg.PlotRejectedPoints and cfg.SaveVelocityVideo:cache_rejected(paths,result)
        record={'frame_index':result['index']}
        record.update({key:result.get(key,np.nan) for key in QC_KEYS})
        record['use_confidence_smoothing']=bool(cfg.UseConfidenceSmoothing)
        qc.append(record)

    started=time.perf_counter()
    if cfg.UseParallelFlow and len(indices)>1:
        requested=cfg.MaxFlowWorkers or max(1,(os.cpu_count() or 1)-1)
        workers=min(requested,len(indices))
        print(f'[FLOW] Processing {len(indices)} pairs with {workers} workers (bounded queue).')
        remaining=iter(indices)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending={}
            for _ in range(workers):
                try:i=next(remaining)
                except StopIteration:break
                pending[pool.submit(compute_flow_pair,i,images,h,w,lo,hi,cfg=cfg)]=i
            with tqdm(total=len(indices),desc='Explicit Farneback + immediate disk write') as progress:
                while pending:
                    done,_=wait(pending,return_when=FIRST_COMPLETED)
                    for future in done:
                        pending.pop(future)
                        result=future.result()
                        try:consume(result)
                        finally:del result
                        progress.update(1)
                        try:i=next(remaining)
                        except StopIteration:continue
                        pending[pool.submit(compute_flow_pair,i,images,h,w,lo,hi,cfg=cfg)]=i
    else:
        print(f'[FLOW] Processing {len(indices)} pairs sequentially (streaming).')
        for i in tqdm(indices,desc='Explicit Farneback + immediate disk write'):
            result=compute_flow_pair(i,images,h,w,lo,hi,cfg=cfg)
            try:consume(result)
            finally:del result
    print(f'[FLOW] Compute + persistence: {time.perf_counter()-started:.2f} s')
    qc.sort(key=lambda row:row['frame_index'])
    if not np.isfinite(speed_min) or not np.isfinite(speed_max):
        speed_min,speed_max=0.,1.
    if speed_max<=speed_min:
        eps=max(abs(speed_min)*1e-6,1e-6)
        speed_min-=eps
        speed_max+=eps
    return indices,qc,ranges,speed_min,speed_max


def run(cfg):
    started=time.perf_counter()
    paths=create_run_paths(cfg)
    print('\n===== 2WCF SINGLE-SEQUENCE RUN =====')
    print(f'Input:  {cfg.input_images}')
    print(f'Output: {paths.root}')
    print(f'Farneback warp mode: {cfg.FarnebackWarpMode}')
    print(f'Information-aware solver: {cfg.UseInformationAwareSolver}')
    print('====================================')
    if cfg.USE_SVD:
        volume,filenames=svd_filter_images(str(cfg.input_images),cfg.KillIDs)
        images=[volume[i] for i in range(volume.shape[0])]
    else:
        images,filenames=load_images_without_svd(str(cfg.input_images))
    if len(images)<2:
        raise ValueError('The input sequence must contain at least two TIFF frames.')
    lo=float(min(np.min(frame) for frame in images))
    hi=float(max(np.max(frame) for frame in images))
    print(f'Global normalization range: [{lo:.6g}, {hi:.6g}]')
    if cfg.PrintPyramidWindowResolution:
        save_pyramid_window_table(images[0],lo,hi,paths,cfg=cfg)
    if cfg.SavePyramidVisualizations:
        save_pyramid_image_diagnostics(images,paths,lo,hi,cfg=cfg)
    indices,qc,ranges,speed_min,speed_max=_process_pairs(images,paths,lo,hi,cfg=cfg)
    print(f'[VIDEO] Fixed speed range [{speed_min:.6g}, {speed_max:.6g}] px/frame')
    render_velocity_video(images,indices,paths,lo,hi,speed_min,speed_max,cfg=cfg)
    render_diagnostic_videos(indices,paths,ranges,cfg=cfg)
    write_csv(paths.root/'flow_qc_metrics.csv',qc)
    if cfg.SaveQCPlot:
        try:
            figure=build_qc_figure(qc)
            save_figure(figure,paths.root/'flow_qc_median_mad.png',dpi=cfg.VideoDPI)
        except Exception as err:
            print(f'[QC] Failed to build QC plot: {err}')
    if (cfg.SavePyramidVelocityPlots or cfg.ExportPyramidVelocityArrays
        or cfg.SaveStreamlinePlots or cfg.SaveInformationMaps):
        save_explicit_flow_diagnostics(images,paths,lo,hi,cfg=cfg)
    write_json(paths.root/'run_manifest.json',{
        'input_images':str(cfg.input_images),
        'source_config':str(cfg.source),
        'image_filenames':filenames,
        'image_count':len(images),
        'processed_pair_indices':indices,
        'global_intensity_min':lo,'global_intensity_max':hi,
        'velocity_color_min':speed_min,'velocity_color_max':speed_max,
        'elapsed_seconds':time.perf_counter()-started,
    })
    print(f'[DONE] Total: {time.perf_counter()-started:.2f}s; output: {paths.root}')
    return paths
