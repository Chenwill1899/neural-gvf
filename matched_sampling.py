"""Change hidden queries only; reuse exactly the original RGB-D feature caches."""
import json
from pathlib import Path
import numpy as np
import torch
from adapt_tartanair import depth_png,pose_matrix,K
from convert import transform,rays_at
from prepare_memory_task import project
from multiscene_experiment import ROOT,CACHE,LABELS,key

DEST=Path('prepared/matched-hidden-v1');OUT=Path('evidence/matched-hidden-v1')
RAY_BIN=.1
LOG_DEPTH_BIN=.1


def bins(points):
    z=points[:,2]
    return np.column_stack((np.floor(points[:,0]/z/RAY_BIN),np.floor(points[:,1]/z/RAY_BIN),
                            np.floor(np.log(z)/LOG_DEPTH_BIN))).astype(np.int64)


def match_pools(free,surface,rng):
    """Equal counts per joint ray/depth bin; no duplicate points within a class."""
    groups=[]
    for points in [free,surface]:
        mapping={}
        for i,b in enumerate(bins(points)):
            mapping.setdefault(tuple(b),[]).append(i)
        groups.append(mapping)
    pairs=[]
    for b in sorted(groups[0].keys() & groups[1].keys()):
        a=np.asarray(groups[0][b]);c=np.asarray(groups[1][b]);n=min(len(a),len(c))
        # Uniform within each class/bin, independent of class-specific depth order.
        a=rng.choice(a,n,replace=False);c=rng.choice(c,n,replace=False)
        pairs.extend(zip(a,c))
    if not pairs:return np.empty((0,3)),np.empty((0,3)),0
    pairs=np.asarray(pairs);available=len(pairs)
    chosen=pairs[rng.choice(available,min(64,available),replace=False)]
    a,c=free[chosen[:,0]],surface[chosen[:,1]]
    assert np.array_equal(bins(a),bins(c))
    return a,c,available


def candidates(previous,current,previous_pose,current_pose,rng):
    previous_valid=np.isfinite(previous)&(previous>=.05)&(previous<=50)
    current_valid=np.isfinite(current)&(current>=.05)&(current<=50)
    v,u=np.where(previous_valid);v,u=v[::4],u[::4]
    surface=rays_at(np.column_stack((u,v)),K)*previous[v,u,None]
    relative=np.linalg.inv(current_pose)@previous_pose
    pools=[]
    for kind,points in enumerate([transform(surface*rng.uniform(.25,.75,(len(surface),1)),relative),
                                  transform(surface,relative)]):
        # Verify the exact representation supplied to the model: rounding a
        # float64 query later can select a different depth pixel at an edge.
        points=points.astype(np.float32)
        uv,inside=project(points,K);indices=np.flatnonzero(inside);uu,vv=uv[indices].T
        z=points[indices,2];measured=current[vv,uu];ok=current_valid[vv,uu]
        if kind==0:ok &= z<measured-.06
        else:ok &= np.abs(z-measured)*np.linalg.norm(points[indices],axis=1)/z<.015
        pools.append(points[indices[ok]])
    return pools


def prepare(limit=None):
    DEST.mkdir(parents=True,exist_ok=True);(DEST/'labels').mkdir(exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    records={r['episode_id']:r for r in json.loads((ROOT/'manifest.json').read_text())['trajectories']}
    windows=[{**w,'start':s,'frames':64}
             for w in json.loads((ROOT/'windows.json').read_text()) if w['split'] in ['train','development']
             for s in range(w['start'],w['start']+w['frames'],64)]
    if limit:windows=windows[:limit]
    inventory=[]
    for index,w in enumerate(windows):
        path=DEST/'labels'/key(w)
        if path.exists():
            cached=json.loads(path.with_suffix('.json').read_text())
            if cached.get('sampling_version')==2:
                inventory.append(cached);continue
        rec=records[w['episode_id']];directory=Path(rec['pose_path']).parent
        poses=np.loadtxt(rec['pose_path']);rng=np.random.default_rng(17)
        old=torch.load(CACHE/key(w),map_location='cpu',weights_only=True,mmap=True)
        labels={k:old[k].clone() for k in LABELS};counts=[]
        for t in range(3,64,4):
            frame=w['start']+t
            depths=[depth_png((directory/'depth_lcam_front'/f'{f:06}_lcam_front_depth.png').read_bytes()) for f in [frame-1,frame]]
            free,surface=candidates(*depths,pose_matrix(poses[frame-1]),pose_matrix(poses[frame]),rng)
            a,b,available=match_pools(free,surface,rng);n=len(a)
            # The cached input is masked regardless of how many pairs survive.
            for name in ['query_xyz','occupancy','known','visible','query_valid','hidden_query']:
                labels[name][t].zero_()
            labels['category'][t].fill_(2)
            if n:
                points=np.concatenate((a,b)).astype(np.float32)
                labels['query_xyz'][t,:2*n]=torch.from_numpy(points)
                labels['occupancy'][t,n:2*n]=1
                for name in ['known','query_valid','hidden_query']:labels[name][t,:2*n]=True
                labels['category'][t,:n]=0;labels['category'][t,n:2*n]=1
                uv,inside=project(points,K)
                assert inside.all() and ((uv[:,0]>=160)&(uv[:,0]<480)&(uv[:,1]>=80)&(uv[:,1]<560)).all()
                # Recheck saved float32 labels against measured target-time geometry.
                measured=depths[1][uv[:,1],uv[:,0]];z=points[:,2]
                assert (z[:n]<measured[:n]-.05999).all()
                assert (np.abs(z[n:]-measured[n:])*np.linalg.norm(points[n:],axis=1)/z[n:]<.01501).all()
            counts.append({'frame':frame,'verified_free_candidates':len(free),'verified_surface_candidates':len(surface),
                           'matched_pairs_available':available,'pairs_used':n})
        unchanged=[t for t in range(64) if t%4!=3]
        assert all(torch.equal(labels[k][unchanged],old[k][unchanged]) for k in LABELS)
        torch.save(labels,path)
        meta={**w,'sampling_version':2,'path':str(path),'events':counts,'hidden_queries':int(labels['hidden_query'].sum()),
              'hidden_frames':sum(r['pairs_used']>0 for r in counts),'empty_hidden_frames':sum(r['pairs_used']==0 for r in counts),
              'feature_cache_reused':str(CACHE/key(w)),'only_queries_changed':True}
        path.with_suffix('.json').write_text(json.dumps(meta,indent=2));inventory.append(meta)
        print(json.dumps({'prepared':index+1,'of':len(windows),'window':key(w),
                          'min_pairs':min(r['pairs_used'] for r in counts),'empty_frames':meta['empty_hidden_frames']}),flush=True)
    (OUT/'label-inventory.json').write_text(json.dumps(inventory,indent=2))
    return inventory


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int);a=p.parse_args();prepare(a.limit)
