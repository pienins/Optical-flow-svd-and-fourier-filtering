"""MP4 workflows. Read one cached field/map at a time; build via plots;
write the rendered RGB frames through outputs.VideoWriter.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from tqdm import tqdm
from .image_io import normalize_frame
from .outputs import (VideoWriter,read_frame_field,read_rejected,
                      remove_rejected_cache)
from .plots import (build_velocity_frame,build_scalar_map_figure,figure_to_rgb)


def render_velocity_video(images, indices, paths, lo, hi, speed_min, speed_max, *, cfg):
    if not cfg.SaveVelocityVideo:return
    norm=Normalize(vmin=speed_min,vmax=speed_max)
    destination=paths.root/'velocity_field.mp4'
    with VideoWriter(destination,cfg.VideoFPS) as writer:
        for index in tqdm(indices,desc='Velocity video (disk-backed)'):
            field=read_frame_field(paths,index)
            rejected=(read_rejected(paths,index) if cfg.PlotRejectedPoints
                      else (np.array([],dtype=float),np.array([],dtype=float)))
            img=normalize_frame(images[index],lo,hi)
            fig=build_velocity_frame(img,field,index,norm,rejected,cfg=cfg)
            try:
                writer.write(figure_to_rgb(fig))
            finally:
                plt.close(fig)
    if cfg.PlotRejectedPoints:remove_rejected_cache(paths)
    print(f'[VIDEO] Saved: {destination}')


def render_diagnostic_videos(indices,paths,ranges,*,cfg):
    if not cfg.SaveDiagnosticMaps:return
    diagnostics=(('local_mad','Local MAD','Local scaled MAD [px]',
                  paths.root/'local_mad.mp4'),
                 ('local_sigma','Local Sigma','Local confidence sigma [px]',
                  paths.root/'local_sigma.mp4'),
                 ('confidence','Optical Flow Confidence','Confidence',
                  paths.root/'confidence.mp4'))
    for key,title,label,destination in diagnostics:
        lo,hi= ranges.get(key,(float('inf'),float('-inf')))
        if key=='confidence':lo,hi=0.0,1.0
        if not np.isfinite(lo):lo=0.0
        if not np.isfinite(hi):hi=1.0
        if hi<=lo:hi=lo+1e-6
        with VideoWriter(destination,cfg.VideoFPS) as writer:
            for index in tqdm(indices,desc=f'Writing {key} MP4'):
                with np.load(paths.diagnostic_maps/f'diagnostic_maps_{index:05d}.npz') as item:
                    scalar_map=item[key].astype(np.float32)
                fig=build_scalar_map_figure(scalar_map,index,title,label,lo,hi,cfg=cfg)
                try:writer.write(figure_to_rgb(fig,tight=False))
                finally:plt.close(fig)
        print(f'[DIAGNOSTIC VIDEO] Saved: {destination}')
