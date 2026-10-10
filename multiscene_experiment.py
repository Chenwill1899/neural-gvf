"""Streaming frozen-frontend, fixed-budget four-map experiment; no downloads."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from adapt_tartanair import depth_png,pose_matrix,K
from convert import sample_queries
from prepare_memory_task import apply_history_mask
from gvf.frontend import load_frontend
from gvf.model import Perception,query_loss

ROOT=Path('prepared/multiscene-v1');OUT=Path('evidence/multiscene-v1');CACHE=ROOT/'feature-clips-v1'
LABELS=['query_xyz','occupancy','known','visible','query_valid','category','hidden_query']
DEVICE='cuda'


def key(w):return w['episode_id'].replace('/','_')+f"_{w['start']:06}.pt"


def raw_clip(w,rec):
    directory=Path(rec['pose_path']).parent;poses=np.loadtxt(rec['pose_path']);rng=np.random.default_rng(7)
    fields={k:[] for k in ['rgb','depth','valid','K','query_xyz','occupancy','known','visible','category']};digest=hashlib.sha256()
    for frame in range(w['start'],w['start']+64):
        rp=directory/'image_lcam_front'/f'{frame:06}_lcam_front.png';dp=directory/'depth_lcam_front'/f'{frame:06}_lcam_front_depth.png'
        rb=rp.read_bytes();db=dp.read_bytes();digest.update(rb);digest.update(db)
        with Image.open(rp) as image:rgb=np.asarray(image.convert('RGB'))
        depth=depth_png(db);valid=np.isfinite(depth)&(depth>=.05)&(depth<=50)
        xyz,occ,known,visible,category,_,_=sample_queries(depth,valid,K,rng,64,.03)
        vals=(rgb,np.where(valid,depth,0).astype(np.float32),valid,K,xyz.astype(np.float32),occ.astype(np.float32),known,visible,category)
        for name,val in zip(fields,vals):fields[name].append(val)
    data={k:np.stack(v) for k,v in fields.items()};data['query_valid']=np.ones_like(data['known'])
    matrices=np.stack([pose_matrix(p) for p in poses[w['start']:w['start']+64]])
    hidden,counts=apply_history_mask(data,matrices)
    return data,digest.hexdigest(),{'hidden_frames':int(hidden.any(1).sum()),'hidden_queries':int(hidden.sum()),'minimum_candidates':min(counts) if counts else 0}


def cache_all(windows,records,frontend):
    CACHE.mkdir(parents=True,exist_ok=True);started=time.perf_counter();inventory=[];parity_done=False
    for index,w in enumerate(windows):
        path=CACHE/key(w)
        if path.exists():
            meta=json.loads(path.with_suffix('.json').read_text());inventory.append(meta);continue
        data,digest,stats=raw_clip(w,records[w['episode_id']]);features=[];classes=[];geometries=[]
        with torch.no_grad():
            for t in range(64):
                rgb=torch.from_numpy(data['rgb'][t:t+1]).permute(0,3,1,2).cuda().float()/255
                depth=torch.from_numpy(data['depth'][t:t+1]).cuda();valid=torch.from_numpy(data['valid'][t:t+1]).cuda()
                calibration=torch.from_numpy(data['K'][t:t+1]).cuda()
                feat,cls=frontend.forward_feat(rgb,1200,depth,enable_depth_mask=True)
                geometry,_=Perception.geometry_inputs(depth,valid,calibration,*feat.shape[-2:])
                if not parity_done:
                    check=Perception(frontend,'memory',normalize_state=True).cuda().eval();z=torch.zeros(1,device='cuda');q=torch.from_numpy(data['query_xyz'][t:t+1]).cuda()
                    direct=check(rgb,depth,valid,calibration,z,z,q,native_features=(feat,cls))[0]
                    cached=check.forward_cached(feat,cls,geometry,z,z,q)[0]
                    error=(direct-cached).abs().max().item();assert torch.allclose(direct,cached,atol=1e-6,rtol=1e-6)
                    (OUT/'cached-path-parity.json').write_text(json.dumps({'actual_frame':w,'max_logit_difference':error,'scope':'actual RGB-D first-frame original versus cached geometry path; frozen frontend'},indent=2))
                    del check;parity_done=True
                features.append(feat.cpu());classes.append(cls.cpu());geometries.append(geometry.cpu())
        pack={k:torch.from_numpy(data[k]) for k in LABELS};pack.update(F=torch.cat(features),CLS=torch.cat(classes),geometry=torch.cat(geometries))
        torch.save(pack,path)
        meta={**w,**stats,'path':str(path),'source_rgb_depth_sha256':digest,'pose_sha256':records[w['episode_id']]['pose_sha256'],
              'cache_bytes':path.stat().st_size,'feature_shape':list(pack['F'].shape),'float_dtype':'float32'}
        path.with_suffix('.json').write_text(json.dumps(meta,indent=2));inventory.append(meta)
        print(json.dumps({'cache':index+1,'of':len(windows),'episode':w['episode_id'],'start':w['start'],'elapsed_s':time.perf_counter()-started,**stats}),flush=True)
    (OUT/'cache-inventory.json').write_text(json.dumps(inventory,indent=2))
    return time.perf_counter()-started


def get(w):
    return {k:v.to(DEVICE) for k,v in torch.load(CACHE/key(w),map_location='cpu',weights_only=True).items()}


def step(model,seq,t,state,frame):
    device=seq['F'].device
    elapsed=torch.tensor([frame*.1],device=device);delta=torch.tensor([0. if frame==0 else .1],device=device)
    return model.forward_cached(seq['F'][t:t+1],seq['CLS'][t:t+1],seq['geometry'][t:t+1],elapsed,delta,seq['query_xyz'][t:t+1],state)


def measure(logits,seq,t):
    known=seq['known'][t]&seq['query_valid'][t];hidden=seq['hidden_query'][t];target=seq['occupancy'][t]
    logits=logits[0];prob=logits[:,0];pred=prob>0
    groups={}
    for name,mask in [('all',known),('hidden',hidden),('visible',known&~hidden)]:
        if mask.any():
            groups[name]={'count':int(mask.sum()),'bce':F.binary_cross_entropy_with_logits(prob[mask],target[mask]).item(),
                          'balanced_accuracy':float(np.mean([(pred[mask&(target==c)]==(c==1)).float().mean().item() for c in (0,1)]))}
    active=seq['query_valid'][t]
    groups['evidence_accuracy']=((logits[:,1]>0)[active]==seq['visible'][t,active]).float().mean().item()
    return groups


def aggregate(frames):
    result={}
    for group in ['all','hidden','visible']:
        rows=[f[group] for f in frames if group in f]
        result[group]={'query_count':sum(r['count'] for r in rows),'frames':len(rows),
                       'bce':float(np.mean([r['bce'] for r in rows])) if rows else None,
                       'balanced_accuracy':float(np.mean([r['balanced_accuracy'] for r in rows])) if rows else None}
    return result


@torch.no_grad()
def evaluate(model,episodes,records,mode,save_predictions=False,erase=False,donor=None):
    model.eval();reports=[]
    for episode in episodes:
        state=None;frames=[];predictions=[];donor_state=None
        for start in [0,64,128,192]:
            seq=get({'episode_id':episode,'start':start})
            dseq=get({'episode_id':donor,'start':start}) if donor else None
            for t in range(64):
                frame=start+t;hidden=bool(seq['hidden_query'][t].any())
                before=state
                logits,state=step(model,seq,t,state,frame)
                # Paired intervention: keep the real recurrent path unchanged afterward.
                if hidden and (erase or donor):logits,_=step(model,seq,t,None if erase else donor_state,frame)
                if donor:_,donor_state=step(model,dseq,t,donor_state,frame)
                frames.append({'frame':frame,**measure(logits,seq,t),'state_rms':state.square().mean().sqrt().item() if state is not None else None})
                if save_predictions:predictions.append(logits.cpu().numpy())
            del seq,dseq
        report={'episode_id':episode,'environment':records[episode]['environment'],'metrics':aggregate(frames),'frames':frames};reports.append(report)
        if save_predictions:np.savez_compressed(OUT/(mode+'-'+episode.replace('/','_')+'.predictions.npz'),logits=np.concatenate(predictions))
    macro={g:{k:float(np.mean([r['metrics'][g][k] for r in reports if r['metrics'][g][k] is not None])) for k in ['bce','balanced_accuracy']} for g in ['all','hidden','visible']}
    return {'macro_trajectory':macro,'episodes':reports,'intervention':'paired erase' if erase else 'paired equal-age wrong-trajectory history' if donor else 'none'}


def main():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.manual_seed(7)
    manifest=json.loads((ROOT/'manifest.json').read_text());records={r['episode_id']:r for r in manifest['trajectories']}
    train=[w for w in json.loads((ROOT/'windows.json').read_text()) if w['split']=='train']
    dev=[e for e,r in records.items() if r['split']=='development'];val=[e for e,r in records.items() if r['split']=='validation']
    assert len(train)==76 and len(dev)==3 and len(val)==9
    assert all(records[w['episode_id']]['environment']!='SoulCity' for w in train)
    assert all(records[e]['environment']=='SoulCity' for e in val)
    all_windows=train+[{'episode_id':e,'environment':records[e]['environment'],'split':records[e]['split'],'start':s,'frames':64,'stride':1} for e in dev+val for s in [0,64,128,192]]
    rng=np.random.default_rng(7);order=[rng.permutation(76).tolist() for _ in range(4)]
    config={'variants':['current','memory_norm'],'train_frames':4864,'development_frames':768,'soulcity_frames':2304,'epochs':4,'updates_per_model':304,'seed':7,'lr':1e-4,
            'frontend':'frozen LingBot v0.5 masked_sdpa_experimental; full float32 F/CLS and exact per-patch geometry cache',
            'selection':'lowest development macro-trajectory hidden-query occupancy BCE after each epoch; ties keep earlier epoch',
            'evaluation':'SoulCity once per selected checkpoint, plus preregistered paired history erasure and Office/P000 donor at equal time/age',
            'no_soulcity_tuning':True,'task':'instantaneous visible and target-time-verified artificial current RGB-D occlusion; not natural hidden occupancy',
            'train_order':order,'dev_episodes':dev,'validation_episodes':val,'windows':all_windows}
    (OUT/'config.json').write_text(json.dumps(config,indent=2));print(json.dumps({'START':{k:v for k,v in config.items() if k not in ['windows','train_order']}}),flush=True)
    frontend=load_frontend('weights/model.pt',0,'masked_sdpa_experimental').cuda().eval()
    config['cache_seconds']=cache_all(all_windows,records,frontend);(OUT/'config.json').write_text(json.dumps(config,indent=2))
    for mode in config['variants']:
        torch.manual_seed(7);model=Perception(frontend,'current' if mode=='current' else 'memory',normalize_state=mode=='memory_norm').cuda()
        params={k:p for k,p in model.named_parameters() if p.requires_grad};before={k:p.detach().cpu().clone() for k,p in params.items()}
        optimizer=torch.optim.AdamW(params.values(),lr=1e-4);report={'mode':mode,'parameters':sum(p.numel() for p in params.values()),'epochs':[]};best=float('inf');total=0.
        torch.cuda.reset_peak_memory_stats()
        for epoch,indices in enumerate(order,1):
            model.train();frontend.eval();steps=[];torch.cuda.synchronize();started=time.perf_counter()
            for n,index in enumerate(indices):
                seq=get(train[index]);optimizer.zero_grad(set_to_none=True);state=None;losses=[]
                for t in range(64):
                    logits,state=step(model,seq,t,state,t)
                    losses.append(query_loss(logits,seq['occupancy'][t:t+1],seq['known'][t:t+1],seq['visible'][t:t+1],seq['query_valid'][t:t+1]))
                loss=torch.stack(losses).mean();assert torch.isfinite(loss);loss.backward()
                grads=[p.grad for p in params.values() if p.grad is not None];assert grads and all(torch.isfinite(g).all() for g in grads)
                norm=torch.nn.utils.clip_grad_norm_(params.values(),1.).item();assert norm>0;optimizer.step()
                steps.append({'window':key(train[index]),'loss':loss.item(),'grad_norm':norm})
                if n%19==0:print(json.dumps({'mode':mode,'epoch':epoch,'update_in_epoch':n+1,'loss':loss.item()}),flush=True)
                del seq,losses,loss,logits,state,grads
            torch.cuda.synchronize();seconds=time.perf_counter()-started;total+=seconds
            development=evaluate(model,dev,records,mode);score=development['macro_trajectory']['hidden']['bce']
            if score<best:
                best=score;report['selected_epoch']=epoch;torch.save({k:p.detach().cpu() for k,p in params.items()},ROOT/(mode+'-selected.pt'))
            report['epochs'].append({'epoch':epoch,'mean_update_loss':float(np.mean([s['loss'] for s in steps])),'train_seconds':seconds,'steps':steps,'development':development})
            (OUT/(mode+'-progress.json')).write_text(json.dumps(report,indent=2));print(json.dumps({'mode':mode,'epoch_done':epoch,'train_seconds':seconds,'dev_hidden_bce':score,'dev_hidden_balanced_accuracy':development['macro_trajectory']['hidden']['balanced_accuracy'],'selected_epoch':report['selected_epoch']}),flush=True)
        report['training_seconds']=total;report['peak_gpu_allocated_bytes']=torch.cuda.max_memory_allocated()
        weights=torch.load(ROOT/(mode+'-selected.pt'),map_location='cpu',weights_only=True);assert set(weights)==set(params);model.load_state_dict(weights,strict=False)
        report['selected_parameter_delta_l2']=sum((p.detach().cpu()-before[k]).square().sum().item() for k,p in params.items())**.5
        print(json.dumps({'mode':mode,'selected_epoch':report['selected_epoch'],'stage':'single SoulCity evaluation'}),flush=True)
        started=time.perf_counter();report['soulcity']=evaluate(model,val,records,mode,save_predictions=True)
        if mode=='memory_norm':
            report['soulcity_erased']=evaluate(model,val,records,mode,erase=True)
            donor='tartanair_v2/Office/Data_easy/P000/lcam_front';report['soulcity_wrong_history']=evaluate(model,val,records,mode,donor=donor)
        report['final_evaluation_seconds']=time.perf_counter()-started
        (OUT/(mode+'.json')).write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps({'FINISHED':mode,'soulcity':report['soulcity']['macro_trajectory'],'training_seconds':total}),flush=True)
        del model,optimizer,params,before
    print('GPU_EXPERIMENT_COMPLETE',flush=True)


if __name__=='__main__':main()
