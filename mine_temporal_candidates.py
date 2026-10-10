"""CPU geometry-only candidate mining; never certifies hidden occupancy or identity."""
import json,time
from pathlib import Path
import numpy as np
from adapt_tartanair import depth_png,pose_matrix,K
from convert import transform,rays_at
ROOT=Path('prepared/multiscene-v1')


def project(points):
    p=points@K.T
    uv=np.rint(p[:,:2]/np.maximum(p[:,2:],1e-9)).astype(int)
    inside=(points[:,2]>.05)&(uv[:,0]>=0)&(uv[:,0]<640)&(uv[:,1]>=0)&(uv[:,1]<640)
    return uv,inside


def main():
    started=time.perf_counter();manifest=json.loads((ROOT/'manifest.json').read_text())
    summaries=[]
    with (ROOT/'temporal-candidates.jsonl').open('w') as out:
        for rec in manifest['trajectories']:
            directory=Path(rec['pose_path']).parent
            poses=np.loadtxt(rec['pose_path']);cache={}
            def depth(t):
                if t not in cache:cache[t]=depth_png((directory/'depth_lcam_front'/f'{t:06}_lcam_front_depth.png').read_bytes())
                return cache[t]
            events=points_total=0
            for a in range(0,224,32):
                b,c=a+16,a+32
                da,db,dc=depth(a),depth(b),depth(c)
                v,u=np.mgrid[8:640:16,8:640:16];uv=np.column_stack((u.ravel(),v.ravel()))
                z=da[uv[:,1],uv[:,0]];good=np.isfinite(z)&(z>=.05)&(z<=50)
                uv,z=uv[good],z[good];pa=rays_at(uv,K)*z[:,None]
                world=transform(pa,pose_matrix(poses[a]))
                pb=transform(world,np.linalg.inv(pose_matrix(poses[b])))
                pc=transform(world,np.linalg.inv(pose_matrix(poses[c])))
                ub,ib=project(pb);uc,ic=project(pc);ids=np.flatnonzero(ib&ic)
                if not len(ids):continue
                zb=db[ub[ids,1],ub[ids,0]];zc=dc[uc[ids,1],uc[ids,0]]
                valid=np.isfinite(zb)&np.isfinite(zc)&(zb>=.05)&(zb<=50)&(zc>=.05)&(zc<=50)
                hidden=zb<pb[ids,2]-.1
                lengths=np.linalg.norm(pc[ids],axis=1)/pc[ids,2]
                reobserved=np.abs(zc-pc[ids,2])*lengths<.03
                ids=ids[valid&hidden&reobserved]
                if len(ids):
                    events+=1;points_total+=len(ids)
                    out.write(json.dumps({'episode_id':rec['episode_id'],'environment':rec['environment'],'split':rec['split'],
                                          'source_frame':a,'occluded_frame':b,'reobserved_frame':c,
                                          'source_uv':uv[ids].tolist(),'source_xyz_optical':pa[ids].tolist(),
                                          'candidate_count':len(ids),'certifies_persistent_object':False,
                                          'hidden_current_occupancy_label':None})+'\n')
            summaries.append({'episode_id':rec['episode_id'],'environment':rec['environment'],'split':rec['split'],
                              'tested_triplets':7,'triplets_with_candidates':events,'candidate_points':points_total})
    result={'scope':'first256 frames per trajectory; t,t+16,t+32; source pixel grid stride16; no GPU',
            'meaning':'surface measurement, intervening nearer depth, later matching surface; not identity/staticness proof',
            'trajectories':summaries,'cpu_seconds':time.perf_counter()-started}
    Path('evidence/temporal-candidates-summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'cpu_seconds':result['cpu_seconds'],'total_candidate_points':sum(x['candidate_points'] for x in summaries),
                      'triplets_with_candidates':sum(x['triplets_with_candidates'] for x in summaries),
                      'by_environment':{e:sum(x['candidate_points'] for x in summaries if x['environment']==e) for e in ['Office','CoalMine','CarWelding','SoulCity']}},indent=2))


if __name__=='__main__':main()
