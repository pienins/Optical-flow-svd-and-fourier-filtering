"""Diagnostic workflows: compute selected data, ask plots.py to build figures,
and ask outputs.py to persist them. Normal frame-pair computation is untouched.
"""
from __future__ import annotations
import numpy as np
import cv2
from .image_io import normalize_frame
from .solver import build_farneback_visualization_pyramid, calc_optical_flow_farneback_explicit
from .plots import (build_pyramid_contact_sheet,build_pyramid_velocity_figure,
                    build_streamline_figure,build_information_figure)
from .outputs import save_figure,write_csv,write_text,write_npz,write_png


def save_pyramid_window_table(frame,lo,hi,paths,*,cfg):
    initial=normalize_frame(frame,lo,hi)
    levels=build_farneback_visualization_pyramid(initial,cfg=cfg)
    h0,w0=initial.shape
    width=int(cfg.WindowSize)
    if int(cfg.FilterFlags)&int(cv2.OPTFLOW_FARNEBACK_GAUSSIAN):
        width=max(1,width)|1
    rows=[]
    for level in levels:
        w,h=level['width'],level['height']
        sx,sy=w/w0,h/h0
        rows.append({'level':level['level'],'width_px':w,'height_px':h,
                     'scale_x':sx,'scale_y':sy,
                     'polynomial_window_level_px':int(cfg.PolyFitKernelSize),
                     'motion_window_level_px':width,
                     'polynomial_window_original_px_x':float(cfg.PolyFitKernelSize)/sx,
                     'polynomial_window_original_px_y':float(cfg.PolyFitKernelSize)/sy,
                     'motion_window_original_px_x':width/sx,
                     'motion_window_original_px_y':width/sy})
        print(f" [PYRAMID] L{level['level']}: {w}x{h}; motion window original px="
              f'{width/sx:.1f}x{width/sy:.1f}')
    write_csv(paths.root/'pyramid_window_resolution.csv',rows)
    return rows


def save_pyramid_image_diagnostics(images,paths,lo,hi,*,cfg):
    if not cfg.SavePyramidVisualizations or len(images)<2:return
    root=paths.pyramid_visualization
    readme=(
        'Farneback pyramid visualization\n================================\n\n'
        'These are the actual pyramid levels used by this explicit/manual solver.\n'
        'No cv2.calcOpticalFlowFarneback call is used.\n'
        f'PyramidDownsampleFactor = {cfg.PyramidDownsampleFactor}\n'
        f'PyramidLevels requested = {cfg.PyramidLevels}\n'
        f'WindowSize = {cfg.WindowSize}\n'
        f'FarnebackPyramidMinSize = {cfg.FarnebackPyramidMinSize}\n'
    )
    write_text(root/'README.txt',readme)
    records=[]
    speed=(float(np.hypot(cfg.PyramidKnownU,cfg.PyramidKnownV))
           if cfg.PyramidKnownU is not None and cfg.PyramidKnownV is not None else None)
    for i in dict.fromkeys(cfg.PyramidVisualizationPairs):
        if not 0<=i<len(images)-1:
            print(f'[PYRAMID] Skipping invalid pair index {i}')
            continue
        a=normalize_frame(images[i],lo,hi)
        b=normalize_frame(images[i+1],lo,hi)
        a_levels=build_farneback_visualization_pyramid(a,cfg=cfg)
        b_levels=build_farneback_visualization_pyramid(b,cfg=cfg)
        n=min(len(a_levels),len(b_levels))
        if not n:continue
        folder=root/f'frame_pair_{i:05d}'
        fig,metadata=build_pyramid_contact_sheet(a_levels,b_levels,i,speed,cfg=cfg)
        save_figure(fig,folder/'pyramid_contact_sheet.png',dpi=cfg.PyramidVisualizationDPI,
                    bbox_inches='tight')
        records.extend(metadata)
        if cfg.SavePyramidLevelImages:
            for a_level,b_level in zip(a_levels,b_levels):
                level=a_level['level']
                images_to_save={
                    'prev':a_level['image'], 'next':b_level['image'],
                    'absdiff':cv2.absdiff(a_level['image'],b_level['image']),
                }
                for name,array in images_to_save.items():
                    write_png(folder/f'level_{level:02d}_{name}.png',
                              np.clip(np.rint(array),0,255).astype(np.uint8))
        print(f'[PYRAMID] Pair {i}: wrote {n} actual pyramid levels.')
    if records:
        write_csv(root/'pyramid_levels.csv',records)


def save_explicit_flow_diagnostics(images,paths,lo,hi,*,cfg):
    """Sequentially rerun ONLY selected pairs, snapshotting one level at a time."""
    if not (cfg.SavePyramidVelocityPlots or cfg.ExportPyramidVelocityArrays
            or cfg.SaveStreamlinePlots or cfg.SaveInformationMaps):return
    h0,w0=images[0].shape
    for i in dict.fromkeys(cfg.DiagnosticFramePairs):
        if not 0<=i<len(images)-1:
            print(f'[EXPLICIT DIAGNOSTICS] Skipping invalid pair {i}')
            continue
        folder=paths.explicit_diagnostics/f'frame_pair_{i:05d}'
        prev=normalize_frame(images[i],lo,hi)
        nxt=normalize_frame(images[i+1],lo,hi)
        records=[]

        def capture_level(level,meta,flow):
            if not (cfg.SavePyramidVelocityPlots or cfg.ExportPyramidVelocityArrays):return
            target=folder/f'level_{level:02d}_flow.npz'
            write_npz(target,flow=flow,image=meta['image'],
                      level=np.int32(level),scale=np.float32(meta['scale']))
            h,w=flow.shape[:2]
            magnitude=np.hypot(flow[...,0]*(w0/w),flow[...,1]*(h0/h))
            values=magnitude[np.isfinite(magnitude)]
            records.append({'level':level,'width_px':w,'height_px':h,
                            'median_speed_original_px_per_frame':float(np.median(values)) if values.size else float('nan'),
                            'p99_speed_original_px_per_frame':float(np.percentile(values,99)) if values.size else float('nan'),
                            'flow_npz':target})

        def capture_information(level,meta,info):
            if not cfg.SaveInformationMaps:return
            write_npz(folder/f'level_{level:02d}_G_information.npz',**info,level=level)
            fig=build_information_figure(info,i,level,cfg=cfg)
            save_figure(fig,folder/f'level_{level:02d}_G_information.png',dpi=160)

        print(f'[EXPLICIT DIAGNOSTICS] Recomputing pair {i} sequentially.')
        flow=calc_optical_flow_farneback_explicit(
            prev,nxt,pyr_scale=cfg.PyramidDownsampleFactor,
            levels=cfg.PyramidLevels,winsize=cfg.WindowSize,
            iterations=cfg.IterationCount,poly_n=cfg.PolyFitKernelSize,
            poly_sigma=cfg.GaussKernelStdev,flags=cfg.FilterFlags,
            regularization=cfg.ExplicitFarnebackRegularization,
            max_increment=cfg.ExplicitMaxResidualIncrement,
            level_callback=capture_level,
            information_callback=capture_information if cfg.SaveInformationMaps else None,
            cfg=cfg,
        )
        if records:
            write_csv(folder/'pyramid_flow_summary.csv',[
                {key:value for key,value in r.items() if key!='flow_npz'}
                for r in sorted(records,key=lambda r:r['level'])])
            candidates=[r['p99_speed_original_px_per_frame'] for r in records
                        if np.isfinite(r['p99_speed_original_px_per_frame'])]
            max_color=(max(candidates,default=1.0) if cfg.PyramidVelocityColorMax is None
                       else float(cfg.PyramidVelocityColorMax))
            for record in sorted(records,key=lambda r:r['level']):
                path=record['flow_npz']
                if cfg.SavePyramidVelocityPlots:
                    with np.load(path) as payload:
                        fig=build_pyramid_velocity_figure(payload['flow'],payload['image'],
                            record['level'],w0,h0,max_color,cfg=cfg)
                    save_figure(fig,folder/f"level_{record['level']:02d}_velocity.png",
                                dpi=cfg.PyramidVelocityDPI,bbox_inches='tight',show=cfg.ShowDiagnosticPlots)
                if not cfg.ExportPyramidVelocityArrays:
                    path.unlink()
        if cfg.SaveStreamlinePlots:
            fig=build_streamline_figure(flow,prev,cfg=cfg)
            save_figure(fig,folder/'raw_farneback_streamlines.png',dpi=cfg.PyramidVelocityDPI,
                        bbox_inches='tight',show=cfg.ShowDiagnosticPlots)
            if cfg.SaveStreamlineSVG:
                # Build a second identical figure to retain a single save/close contract.
                fig=build_streamline_figure(flow,prev,cfg=cfg)
                save_figure(fig,folder/'raw_farneback_streamlines.svg',
                            bbox_inches='tight')
    print(f'[EXPLICIT DIAGNOSTICS] Outputs: {paths.explicit_diagnostics}')
