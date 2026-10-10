"""CPU data-interface and adjacent-frame geometry evidence, no network execution."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from train import load


def reprojection(data, poses):
    errors = []
    for t in range(len(poses)-1):
        v, u = np.where(data['valid'][t])
        u, v = u[::16], v[::16]
        rays = np.linalg.solve(data['K'][t], np.stack([u, v, np.ones_like(u)]))
        xyz = rays*data['depth'][t,v,u]
        relative = np.linalg.inv(poses[t+1]) @ poses[t]
        moved = relative[:3,:3] @ xyz + relative[:3,3:4]
        projected = data['K'][t+1] @ moved
        uv = np.rint(projected[:2]/np.maximum(projected[2:], 1e-9)).astype(int)
        good = (moved[2] > .05) & (uv[0]>=0) & (uv[0]<640) & (uv[1]>=0) & (uv[1]<640)
        ids = np.flatnonzero(good)
        uu, vv = uv[:, ids]
        ids = ids[data['valid'][t+1,vv,uu]]
        uu, vv = uv[:, ids]
        errors.extend(np.abs(moved[2,ids]-data['depth'][t+1,vv,uu]).tolist())
    return {'samples':len(errors), 'abs_z_error_quantiles_m':np.quantile(errors,[.5,.9,.95]).tolist(),
            'fraction_under_10cm':float(np.mean(np.array(errors)<.1)),
            'note':'Includes occlusions, moving surfaces and nearest-pixel errors; diagnostic, not full geometry ground truth.'}


def main():
    root = Path('prepared/tartanair-starter')
    results = {}
    fig, axes = plt.subplots(2,3, figsize=(12,8), constrained_layout=True)
    torch.set_num_threads(4)
    for row, traj in enumerate(['P000','P005']):
        path = root/f'{traj}_000000_008.npz'
        with np.load(path) as z:
            data = {k:z[k] for k in z.files}
        with np.load(path.with_suffix('.supervision.npz')) as z:
            poses = z['T_world_enu_from_camera_optical']
        loaded = load(path, 'cpu')
        assert not any('pose' in k for k in loaded)
        assert np.isfinite(data['depth']).all() and np.all(data['depth'][~data['valid']] == 0)
        assert np.all(~data['known'][data['category'] == 2])
        assert np.allclose(np.diff(data['timestamp']), .1)
        assert np.allclose(poses[:,:3,:3].transpose(0,2,1) @ poses[:,:3,:3], np.eye(3), atol=1e-6)
        assert np.allclose(np.linalg.det(poses[:,:3,:3]), 1, atol=1e-6)
        result = reprojection(data, poses)
        result.update(input_shape=list(loaded['rgb'].shape), queries_per_frame=int(data['known'].shape[1]),
                      no_pose_in_loader=True, invalid_excluded=True,
                      valid_fraction=float(data['valid'].mean()), archive_clip_bytes=path.stat().st_size)
        results[traj] = result
        axes[row,0].imshow(data['rgb'][0]); axes[row,0].set_title(traj+' RGB, frame 0')
        im=axes[row,1].imshow(np.where(data['valid'][0],data['depth'][0],np.nan),vmin=0,vmax=50,cmap='viridis')
        axes[row,1].set_title('Optical-axis depth (m)'); fig.colorbar(im,ax=axes[row,1],fraction=.04)
        axes[row,2].imshow(data['valid'][0],vmin=0,vmax=1,cmap='gray'); axes[row,2].set_title('Valid 0.05–50 m')
        for ax in axes[row]: ax.axis('off')
    fig.savefig('evidence/tartanair-samples.png',dpi=130)
    Path('evidence/tartanair-readcheck.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))


if __name__ == '__main__': main()
