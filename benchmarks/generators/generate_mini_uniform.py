"""New small deterministic example, adapted from supplied uniform particle generator.

Not an original 99-pair result. Run from repository root:
    python benchmarks/generators/generate_mini_uniform.py
"""
from pathlib import Path
import json
import numpy as np
import cv2
from tifffile import imwrite

W = H = 128
N_FRAMES = 6
U, V = 1.0, 0.0
TARGET_VISIBLE = 600
SIGMA = 1.1
PEAK = 220.0
SEED = 42
OUT = Path(__file__).resolve().parents[2] / "data/examples/mini_uniform"
FRAME_DIR = OUT / "frames"


def main():
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    pad = max(6, int(np.ceil(4.0 * SIGMA)) + 2)
    margin_x = int(np.ceil(abs(U) * (N_FRAMES-1))) + pad + 10
    margin_y = int(np.ceil(abs(V) * (N_FRAMES-1))) + pad + 10
    area = (W+2*margin_x)*(H+2*margin_y)
    count = int(np.ceil(TARGET_VISIBLE / (W*H) * area))
    rng = np.random.default_rng(SEED)
    x0 = rng.uniform(-margin_x,W+margin_x,count)
    y0 = rng.uniform(-margin_y,H+margin_y,count)
    test = np.zeros((101,101),dtype=np.float32)
    test[50,50]=1
    scale = PEAK/cv2.GaussianBlur(test,(0,0),SIGMA,borderType=cv2.BORDER_CONSTANT).max()
    for t in range(N_FRAMES):
        x=x0+t*U+pad; y=y0+t*V+pad
        i=np.floor(x).astype(np.int32);j=np.floor(y).astype(np.int32)
        dx=x-i;dy=y-j
        canvas=np.zeros((H+2*pad,W+2*pad),np.float32)
        for ii,jj,wt in ((i,j,(1-dx)*(1-dy)),(i+1,j,dx*(1-dy)),(i,j+1,(1-dx)*dy),(i+1,j+1,dx*dy)):
            ok=(ii>=0)&(ii<canvas.shape[1])&(jj>=0)&(jj<canvas.shape[0])
            np.add.at(canvas,(jj[ok],ii[ok]),wt[ok])
        canvas=cv2.GaussianBlur(canvas,(0,0),SIGMA,borderType=cv2.BORDER_CONSTANT)
        img=np.clip(canvas*scale,0,255)[pad:pad+H,pad:pad+W].astype(np.uint8)
        imwrite(FRAME_DIR/f"frame_{t:05d}.tiff",img)
    np.savez_compressed(OUT/'true_vector_values.npz',u_gt=np.full((H,W),U,np.float32),v_gt=np.full((H,W),V,np.float32),true_u=np.float32(U),true_v=np.float32(V),true_speed=np.float32(np.hypot(U,V)))
    (OUT/'parameters.json').write_text(json.dumps(dict(width=W,height=H,n_frames=N_FRAMES,true_u_px_per_frame=U,true_v_px_per_frame=V,particle_sigma_px=SIGMA,particle_peak=PEAK,seed=SEED),indent=2))
    print(f'Created {N_FRAMES} TIFFs, truth NPZ and parameters in {OUT}')


if __name__=='__main__':
    main()
