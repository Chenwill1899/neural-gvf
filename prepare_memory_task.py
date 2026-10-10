"""Synthetic current-observation occlusion, with independently checked history labels."""
import json
from pathlib import Path
import numpy as np
from convert import transform, rays_at

SOURCE=Path('prepared/tartanair-long')
DEST=Path('prepared/tartanair-memory-task')


def project(points,K):
    uvz=points@K.T
    uv=np.rint(uvz[:,:2]/np.maximum(uvz[:,2:],1e-8)).astype(int)
    u,v=uv.T
    inside=(points[:,2]>.05)&(u>=200)&(u<440)&(v>=120)&(v<520)
    return uv,inside


def apply_history_mask(data, poses):
    original_depth=data['depth'].copy();original_valid=data['valid'].copy()
    hidden=np.zeros_like(data['known']);rng=np.random.default_rng(17)
    counts=[]
    for t in range(3,len(data['rgb']),4):
        # t-1 is an intact input frame; no dynamic/static assumption is needed:
        # current unmasked GT verifies each candidate before input masking.
        v,u=np.where(original_valid[t-1]);v,u=v[::4],u[::4]
        rays=rays_at(np.column_stack((u,v)),data['K'][t-1])
        surf=rays*original_depth[t-1,v,u,None]
        relative=np.linalg.inv(poses[t])@poses[t-1]
        candidates=[transform(surf*rng.uniform(.25,.75,(len(surf),1)),relative),transform(surf,relative)]
        groups=[]
        for kind,points in enumerate(candidates):
            uv,inside=project(points,data['K'][t]);selected=np.flatnonzero(inside)
            uu,vv=uv[selected].T
            measured=original_depth[t,vv,uu]
            z=points[selected,2]
            ok=original_valid[t,vv,uu]
            if kind==0: ok &= z<measured-.06
            else:
                # Euclidean distance along current ray within half the3cm band.
                lengths=np.linalg.norm(points[selected],axis=1)/z
                ok &= np.abs(z-measured)*lengths<.015
            selected=selected[ok]
            if len(selected)<64: break
            groups.append(points[rng.choice(selected,64,replace=False)])
            counts.append(len(selected))
        if len(groups)!=2: continue
        data['query_xyz'][t]=0;data['query_xyz'][t,:128]=np.concatenate(groups).astype(np.float32)
        data['occupancy'][t]=0;data['occupancy'][t,64:128]=1
        data['known'][t]=False;data['known'][t,:128]=True
        data['query_valid'][t]=False;data['query_valid'][t,:128]=True
        data['visible'][t]=False
        data['category'][t]=2;data['category'][t,:64]=0;data['category'][t,64:128]=1
        hidden[t,:128]=True
        data['rgb'][t,80:560,160:480]=0
        data['depth'][t,80:560,160:480]=0
        data['valid'][t,80:560,160:480]=False
        uv,_=project(data['query_xyz'][t,:128],data['K'][t])
        assert not data['valid'][t,uv[:,1],uv[:,0]].any()
        assert not hidden[t-1].any()
    data['hidden_query']=hidden
    return hidden, counts


def main():
    DEST.mkdir(exist_ok=True)
    inventory=json.loads((SOURCE/'inventory.json').read_text())
    reports=[]
    for clip in inventory['clips']:
        path=Path(clip['path'])
        with np.load(path) as z: data={k:z[k].copy() for k in z.files}
        with np.load(path.with_suffix('.supervision.npz')) as z:
            poses=z['T_world_enu_from_camera_optical'];ids=z['frame_id']
        assert np.all(np.diff(ids)==1) and np.allclose(np.diff(data['timestamp']),.1)
        hidden,counts=apply_history_mask(data,poses)
        target=DEST/path.name
        np.savez_compressed(target,**data)
        reports.append({'path':str(target),'trajectory':clip['trajectory'],'split':clip['split'],
                        'start':clip['start'],'frames':clip['frames'],'hidden_frames':int(hidden.any(1).sum()),
                        'hidden_queries':int(hidden.sum()),'minimum_candidates':min(counts) if counts else 0,
                        'source_frame_stride':1,'timestamp_delta_s':.1})
        print(json.dumps(reports[-1]),flush=True)
    (DEST/'inventory.json').write_text(json.dumps({'clips':reports,'task':'synthetic RGB-D central occlusion; history t-1; current GT verification'},indent=2))


if __name__=='__main__':main()
