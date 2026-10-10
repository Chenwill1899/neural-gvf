"""CPU-only read-only source audit and environment-isolated indexing; no torch."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from PIL import Image
from adapt_tartanair import depth_png,pose_matrix,K

SOURCE=Path('/home/ubuntu5090/cwl/datasets/tartanair_v2_extracted')
OUTPUT=Path('prepared/multiscene-v1')
ENVIRONMENTS=['Office','CoalMine','CarWelding','SoulCity']


def image_index(directory,suffix):
    records={}
    for path in directory.glob('*.png'):
        frame=int(path.name.split('_')[0])
        if path.name!=f'{frame:06}_lcam_front{suffix}.png' or frame in records:
            raise ValueError('unexpected/duplicate image name')
        records[frame]=path
    return records


def main():
    started=time.perf_counter();OUTPUT.mkdir(parents=True,exist_ok=True)
    trajectories=[];windows=[];total=0
    with (OUTPUT/'frames.jsonl').open('w') as f:
        for environment in ENVIRONMENTS:
            dirs=sorted((SOURCE/environment/'Data_easy').glob('P[0-9][0-9][0-9]'))
            if not dirs:raise ValueError('environment missing')
            for directory in dirs:
                trajectory=directory.name
                split='validation' if environment=='SoulCity' else 'development' if directory==dirs[-1] else 'train'
                episode=f'tartanair_v2/{environment}/Data_easy/{trajectory}/lcam_front'
                rgb=image_index(directory/'image_lcam_front','');depth=image_index(directory/'depth_lcam_front','_depth')
                ids=sorted(rgb)
                if ids!=sorted(depth) or ids!=list(range(len(ids))):raise ValueError('frame pairing/continuity mismatch')
                pose_path=directory/'pose_lcam_front.txt';poses=np.loadtxt(pose_path)
                if poses.shape!=(len(ids),7):raise ValueError('pose row mismatch')
                for p in poses:pose_matrix(p)
                samples=[]
                for frame in sorted(set([0,len(ids)//2,len(ids)-1])):
                    raw=depth_png(depth[frame].read_bytes())
                    with Image.open(rgb[frame]) as image:
                        image.load()
                        if image.size!=(640,640) or image.mode not in ('RGB','RGBA'):raise ValueError('RGB format mismatch')
                    valid=np.isfinite(raw)&(raw>=.05)&(raw<=50)
                    if not valid.any():raise ValueError('no usable measured depth')
                    samples.append({'frame':frame,'valid_fraction':float(valid.mean()),
                                    'valid_depth_quantiles_m':np.quantile(raw[valid],[0,.5,.95,1]).tolist(),
                                    'invalid_count':int((~valid).sum()),'depth_png_sha256':hashlib.sha256(depth[frame].read_bytes()).hexdigest()})
                record={'environment':environment,'trajectory':trajectory,'episode_id':episode,'split':split,'frames':len(ids),
                        'pose_path':str(pose_path),'pose_sha256':hashlib.sha256(pose_path.read_bytes()).hexdigest(),
                        'source_stride':1,'nominal_hz':10,'timestamp_source':'frame_index / official nominal10Hz, no recorded hardware timestamps',
                        'available_modalities':sorted(p.name for p in directory.iterdir()),'depth_samples':samples}
                trajectories.append(record)
                for frame in ids:
                    f.write(json.dumps({'episode_id':episode,'environment':environment,'split':split,'frame_id':frame,
                                        'rgb':str(rgb[frame]),'depth':str(depth[frame]),'pose_file':str(pose_path),'pose_row':frame,
                                        'nominal_timestamp_s':frame*.1})+'\n')
                if split=='train':
                    starts=[0,64,128,192]
                    if starts[-1]+64>len(ids) or starts[-1]<64:raise ValueError('trajectory too short for four consecutive64-frame windows')
                    windows.extend({'episode_id':episode,'environment':environment,'split':split,'start':s,'frames':64,'stride':1} for s in starts)
                else:
                    if len(ids)<256:raise ValueError('held-out trajectory shorter than256')
                    windows.append({'episode_id':episode,'environment':environment,'split':split,'start':0,'frames':256,'stride':1})
                total+=len(ids)
                print(json.dumps({k:record[k] for k in ['environment','trajectory','split','frames']}),flush=True)
    counts={}
    for env in ENVIRONMENTS:
        selected=[t for t in trajectories if t['environment']==env]
        counts[env]={'trajectories':len(selected),'frames':sum(t['frames'] for t in selected)}
    splits={s:{'trajectories':sum(t['split']==s for t in trajectories),'available_frames':sum(t['frames'] for t in trajectories if t['split']==s),
               'planned_windows':sum(w['split']==s for w in windows),'planned_unique_frames':sum(w['frames'] for w in windows if w['split']==s)} for s in ['train','development','validation']}
    assert not {t['episode_id'] for t in trajectories if t['split']=='train'} & {t['episode_id'] for t in trajectories if t['split']!='train'}
    manifest={'source_root':str(SOURCE),'total_frames':total,'total_trajectories':len(trajectories),'counts':counts,'splits':splits,
              'K_integer_pixel_centres':K.tolist(),'pose_role':'offline supervision only',
              'supervision':'instantaneous measured free/surface/unknown only; no dynamic identity or hidden occupancy GT',
              'trajectories':trajectories,'cpu_seconds':time.perf_counter()-started}
    (OUTPUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (OUTPUT/'windows.json').write_text(json.dumps(windows,indent=2))
    summary={k:v for k,v in manifest.items() if k!='trajectories'}
    Path('evidence/multiscene-preparation.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
