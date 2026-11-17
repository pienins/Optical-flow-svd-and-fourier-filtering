"""Build scientific figures: NO filenames, disk I/O, or video encoders.

Every function returns a live Matplotlib Figure. The caller must pass it to
outputs.save_figure or convert it with figure_to_rgb then close it.
"""
from __future__ import annotations
from io import BytesIO
import imageio.v2 as imageio
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def figure_to_rgb(fig, *, tight=True):
    """Convert a figure into a video-ready RGB ndarray, without saving to disk."""
    with BytesIO() as buffer:
        fig.savefig(buffer,format='png',bbox_inches='tight' if tight else None)
        buffer.seek(0)
        frame=imageio.imread(buffer)
    if frame.ndim==2:
        return cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
    if frame.shape[-1]==4:
        return frame[:,:,:3]
    return frame


def build_velocity_frame(img, field, index, color_norm, rejected, *, cfg):
    """Vector glyphs over the previous image, retaining speed as color."""
    x=np.asarray(field['x'],dtype=float)
    y=np.asarray(field['y'],dtype=float)
    u=np.asarray(field['u'],dtype=float)
    v=np.asarray(field['v'],dtype=float)
    x_rejected,y_rejected=rejected
    fig,ax=plt.subplots(figsize=(6,6),dpi=cfg.VideoDPI)
    ax.imshow(img,cmap='gray')
    speed=np.hypot(u,v)
    usable=np.isfinite(speed)&(speed>np.finfo(float).eps)
    u_norm=np.divide(u,speed,out=np.zeros_like(u),where=usable)
    v_norm=np.divide(v,speed,out=np.zeros_like(v),where=usable)
    glyphs=ax.quiver(x,y,u_norm,v_norm,speed,cmap='jet',norm=color_norm,
                     scale=cfg.ArrowScale,angles='xy',zorder=3)
    fig.colorbar(glyphs,ax=ax,fraction=0.046,pad=0.04).set_label('Velocity magnitude [px/frame]')
    if cfg.PlotRejectedPoints and x_rejected.size:
        ax.scatter(x_rejected,y_rejected,s=4,c='magenta',marker='.',alpha=0.9,
                   zorder=4,label='Rejected by final validity test')
    ax.set_title(f'Frame {index} → {index+1}')
    ax.axis('off')
    return fig


def build_scalar_map_figure(scalar_map,index,title,colorbar_label,vmin,vmax,*,cfg):
    fig,ax=plt.subplots(figsize=(7,6),dpi=cfg.VideoDPI)
    display=ax.imshow(scalar_map,cmap='jet',vmin=vmin,vmax=vmax)
    fig.colorbar(display,ax=ax,fraction=0.046,pad=0.04).set_label(colorbar_label)
    ax.set_title(f'{title}\nFrame {index} → {index+1}')
    ax.axis('off')
    fig.tight_layout()
    return fig


def build_qc_figure(records):
    frames=[v['frame_index'] for v in records]
    medians=[v['median_err'] for v in records]
    mads=[v['mad_err'] for v in records]
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(8,6),sharex=True)
    fig.suptitle('Forward–Backward Consistency Diagnostics')
    ax1.plot(frames,medians,marker='o')
    ax1.set_ylabel('median_err')
    ax1.grid(True,alpha=0.3)
    ax2.plot(frames,mads,marker='o')
    ax2.set_xlabel('frame index')
    ax2.set_ylabel('mad_err')
    ax2.grid(True,alpha=0.3)
    fig.tight_layout(rect=[0,0,1,0.96])
    return fig


def build_information_figure(info,pair_index,level,*,cfg):
    eig_min=np.maximum(info['eig_min_raw'],1e-20)
    eig_max=np.maximum(info['eig_max_raw'],1e-20)
    fig,axes=plt.subplots(1,3,figsize=(12,3.6))
    axes[0].imshow(np.log10(eig_min),cmap='viridis')
    axes[0].set_title('log10(lambda min), raw G')
    axes[1].imshow(np.log10(eig_max),cmap='viridis')
    axes[1].set_title('log10(lambda max), raw G')
    axes[2].imshow(info['observability'],cmap='viridis',vmin=0,vmax=1)
    axes[2].set_title('2D observability, 0..1')
    for ax in axes:ax.axis('off')
    fig.suptitle(f'Frame {pair_index} -> {pair_index+1} | pyramid L{level} '
                 f'| info-aware solver={cfg.UseInformationAwareSolver}')
    fig.tight_layout()
    return fig


def build_pyramid_velocity_figure(flow,frame,level,original_w,original_h,color_max,*,cfg):
    """Speed, signed components and quiver, all in original-pixel units except arrows."""
    h,w=frame.shape
    u_orig=flow[...,0]*(original_w/w)
    v_orig=flow[...,1]*(original_h/h)
    speed=np.hypot(u_orig,v_orig)
    fig,axes=plt.subplots(2,2,figsize=(12,10))
    ax_speed,ax_u,ax_v,ax_vectors=axes.flat
    color_max=max(float(color_max),1e-6)
    heatmap=ax_speed.imshow(speed,cmap='viridis',origin='upper',vmin=0,vmax=color_max)
    fig.colorbar(heatmap,ax=ax_speed,label='Speed [original px/frame]',fraction=0.046)
    ax_speed.set_title('Raw speed magnitude')
    for axis,component,name in ((ax_u,u_orig,'Horizontal u'),(ax_v,v_orig,'Vertical v')):
        display=axis.imshow(component,cmap='coolwarm',origin='upper',
                            vmin=-color_max,vmax=color_max)
        fig.colorbar(display,ax=axis,label=f'{name} [original px/frame]',fraction=0.046)
        axis.set_title(name)
    ax_vectors.imshow(frame,cmap='gray',origin='upper',vmin=0,vmax=255)
    step=max(1,int(np.ceil(max(h,w)/max(2,int(cfg.PyramidQuiverVectorsAcross)))))
    gy,gx=np.mgrid[step//2:h:step,step//2:w:step]
    if gx.size:
        ax_vectors.quiver(gx,gy,flow[step//2:h:step,step//2:w:step,0],
                          flow[step//2:h:step,step//2:w:step,1],
                          angles='xy',scale_units='xy',scale=float(cfg.PyramidQuiverScale),
                          color='tomato',width=0.003,pivot='mid')
    ax_vectors.set_title('Raw flow vectors on image (level px/frame)')
    for axis in axes.flat:
        axis.set_xlim(-0.5,w-0.5)
        axis.set_ylim(h-0.5,-0.5)
        axis.set_aspect('equal')
        axis.set_xlabel('x [level px]')
        axis.set_ylabel('y [level px]')
    fig.suptitle(f'Explicit Farneback | pyramid level {level} | {w} x {h} px | '
                 f'motion window {int(cfg.WindowSize)} px')
    fig.tight_layout(rect=[0,0,1,0.97])
    return fig


def build_streamline_figure(flow,frame,*,cfg):
    """Instantaneous streamlines of RAW, unfiltered dense flow (not trajectories)."""
    h,w=frame.shape
    step=max(1,int(np.ceil(max(h,w)/max(2,int(cfg.StreamlineMaxGridSize)))))
    x=np.arange(0,w,step,dtype=np.float32)
    y=np.arange(0,h,step,dtype=np.float32)
    u=flow[::step,::step,0]
    v=flow[::step,::step,1]
    speed=np.hypot(u,v)
    valid=np.isfinite(u)&np.isfinite(v)&(speed>=float(cfg.StreamlineMinSpeed))
    u_masked=np.ma.masked_where(~valid,u)
    v_masked=np.ma.masked_where(~valid,v)
    speed_masked=np.ma.masked_where(~valid,speed)
    fig,ax=plt.subplots(figsize=(9,7))
    ax.imshow(frame,cmap='gray',origin='upper',vmin=0,vmax=255,
              extent=(-0.5,w-0.5,h-0.5,-0.5))
    if np.count_nonzero(valid)>=4 and len(x)>=2 and len(y)>=2:
        top_speed=max(float(np.percentile(speed[valid],99)),1e-6)
        try:
            curves=ax.streamplot(x,y,u_masked,v_masked,color=speed_masked,
                                 cmap='viridis',norm=Normalize(0,top_speed),
                                 density=float(cfg.StreamlineDensity),
                                 linewidth=1.1,arrowsize=1.0)
            fig.colorbar(curves.lines,ax=ax,label='Raw speed [px/frame]',fraction=0.046)
        except ValueError as exc:
            ax.text(.5,.5,f'Cannot integrate streamlines: {exc}',
                    ha='center',va='center',transform=ax.transAxes)
    else:
        ax.text(.5,.5,'Insufficient nonzero flow for streamlines',
                ha='center',va='center',transform=ax.transAxes)
    ax.set_xlim(-.5,w-.5)
    ax.set_ylim(h-.5,-.5)
    ax.set_aspect('equal')
    ax.set_xlabel('x [image px]')
    ax.set_ylabel('y [image px; downward positive]')
    ax.set_title('Raw explicit Farneback streamlines (NOT 2WCF validated)')
    fig.tight_layout()
    return fig


def build_pyramid_contact_sheet(prev_levels,next_levels,pair_index,known_speed,*,cfg):
    """Return a contact-sheet figure and metadata; no PNG is saved here."""
    n_levels=min(len(prev_levels),len(next_levels))
    fig,axes=plt.subplots(n_levels,4,figsize=(14,max(3.0,3.0*n_levels)),
                          squeeze=False,gridspec_kw={'width_ratios':[1.15,2.0,2.0,2.0]})
    records=[]
    for row in range(n_levels):
        prev=prev_levels[row]
        nxt=next_levels[row]
        image_a=prev['image']
        image_b=nxt['image']
        diff=cv2.absdiff(image_a,image_b)
        level,scale,width,height=prev['level'],prev['scale'],prev['width'],prev['height']
        sigma,kernel=prev['sigma'],prev['smooth_kernel']
        d=known_speed*scale if known_speed is not None else float('nan')
        ratio=d/float(cfg.WindowSize) if known_speed is not None and cfg.WindowSize>0 else float('nan')
        annotations=[f'LEVEL {level}',f'{width} x {height} px',f'scale = {scale:.4g}',
                     f'sigma = {sigma:.3g}',f'kernel = {kernel}']
        if known_speed is not None:
            annotations.extend([f'GT d = {d:.3g} px',f'd / W = {ratio:.3g}'])
        axes[row,0].axis('off')
        axes[row,0].text(.5,.5,'\n'.join(annotations),transform=axes[row,0].transAxes,
                         ha='center',va='center',fontsize=9)
        for col,content in enumerate((image_a,image_b,diff),start=1):
            axes[row,col].imshow(content,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
            axes[row,col].set_xticks([])
            axes[row,col].set_yticks([])
        records.append({'frame_pair':pair_index,'level':level,'scale':scale,
                        'width_px':width,'height_px':height,
                        'gaussian_sigma_px_at_level0':sigma,
                        'gaussian_kernel_px_at_level0':kernel,
                        'known_u_px_per_frame':float(cfg.PyramidKnownU) if cfg.PyramidKnownU is not None else float('nan'),
                        'known_v_px_per_frame':float(cfg.PyramidKnownV) if cfg.PyramidKnownV is not None else float('nan'),
                        'known_speed_px_per_frame':known_speed if known_speed is not None else float('nan'),
                        'effective_displacement_px':d,'window_size_px':float(cfg.WindowSize),
                        'displacement_to_window_ratio':ratio})
    for ax,txt in zip(axes[0,:],('Level metadata','Previous frame','Next frame','Absolute difference')):
        ax.set_title(txt,fontsize=11)
    fig.suptitle('Farneback diagnostic image pyramid\n'
                 f'Frame {pair_index} -> {pair_index+1} | pyr_scale={cfg.PyramidDownsampleFactor} | '
                 f'requested levels={cfg.PyramidLevels} | visualized levels={n_levels} | '
                 f'winsize={cfg.WindowSize}',fontsize=12)
    fig.tight_layout(rect=[.02,.02,1,.95])
    return fig,records
