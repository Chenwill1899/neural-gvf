"""One fixed-budget current/memory experiment, whole-trajectory TartanAir holdout."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from gvf.frontend import load_frontend
from gvf.model import Perception, query_loss
from train import load, rollout

ROOT = Path('prepared/tartanair-long')
OUT = Path('evidence/tartanair-long')


def gpu(seq):
    return {k: ([tuple(x.cuda() for x in pair) for pair in v] if k == '_features'
                else v.cuda() if isinstance(v, torch.Tensor) else v) for k,v in seq.items()}


def metrics(logits, seq):
    loss = query_loss(logits, seq['occupancy'], seq['known'], seq['visible'], seq['query_valid']).item()
    by = {}
    for c in (0,1,2):
        mask = (seq['category'] == c) & seq['query_valid']
        by[str(c)] = {'queries': int(mask.sum()),
                     'evidence_accuracy': ((logits[...,1]>0)[mask] == seq['visible'][mask]).float().mean().item() if mask.any() else None}
        if c < 2:
            by[str(c)]['surface_free_accuracy'] = ((logits[...,0]>0)[mask] == seq['occupancy'][mask].bool()).float().mean().item()
    result={'loss':loss,'balanced_accuracy':(by['0']['surface_free_accuracy']+by['1']['surface_free_accuracy'])/2,'by_category':by}
    result['hidden_balanced_accuracy']=None
    if 'hidden_query' in seq and seq['hidden_query'].any():
        result['hidden_balanced_accuracy']=float(np.mean([((logits[...,0]>0)[seq['hidden_query'] & (seq['category']==c)] == seq['occupancy'][seq['hidden_query'] & (seq['category']==c)].bool()).float().mean().item() for c in (0,1)]))
    return result


@torch.no_grad()
def evaluate(model, sequences, reset_every=None, erase_history=False):
    model.eval()
    state, frames, chunks, predictions = None, [], [], []
    offset = 0
    for sequence in sequences:
        seq = gpu(sequence)
        logits = []
        for t in range(len(seq['rgb'])):
            if (reset_every and (offset+t) % reset_every == 0) or (erase_history and seq['hidden_query'][t].any()):
                state = None
            elapsed = seq['timestamp'][t:t+1] + offset*.1
            delta = elapsed*0 if offset+t == 0 else elapsed*0+.1
            pred,state,_ = model(seq['rgb'][t:t+1],seq['depth'][t:t+1],seq['valid'][t:t+1],seq['K'][t:t+1],
                                 elapsed,delta,seq['query_xyz'][t:t+1],state,seq['_features'][t])
            logits.append(pred)
            frame = {k:v[t:t+1] for k,v in seq.items() if isinstance(v,torch.Tensor)}
            frames.append({'frame':offset+t,**metrics(pred,frame),
                           'state_rms':state.square().mean().sqrt().item() if state is not None else None})
        logits = torch.cat(logits)
        chunks.append({'first_frame':offset,**metrics(logits,seq)})
        predictions.append(logits.cpu().numpy())
        offset += len(seq['rgb'])
        del seq
    return {'loss':float(np.mean([f['loss'] for f in frames])),
            'balanced_accuracy':float(np.mean([f['balanced_accuracy'] for f in frames])),
            'hidden_balanced_accuracy':float(np.mean([f['hidden_balanced_accuracy'] for f in frames if f['hidden_balanced_accuracy'] is not None])) if any(f['hidden_balanced_accuracy'] is not None for f in frames) else None,
            'chunks':chunks,'frames':frames}, np.concatenate(predictions)


def main():
    global ROOT,OUT
    parser=argparse.ArgumentParser();parser.add_argument('--task',choices=['visible','masked'],default='visible');args=parser.parse_args()
    if args.task=='masked': ROOT=Path('prepared/tartanair-memory-task');OUT=Path('evidence/tartanair-memory-task')
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(7)
    inventory = json.loads((ROOT/'inventory.json').read_text())
    paths = [Path(c['path']) for c in inventory['clips']]
    train_paths = [p for p in paths if not p.name.startswith('P005')]
    val_paths = sorted([p for p in paths if p.name.startswith('P005')])
    assert len(train_paths)==10 and len(val_paths)==4
    assert [int(p.stem.split('_')[1]) for p in val_paths] == [0,64,128,192]
    assert {p.stem.split('_')[0] for p in train_paths} == {'P000','P001','P002','P003','P004'}
    config = {'train_paths':[str(p) for p in train_paths], 'val_paths':[str(p) for p in val_paths],
              'seed':7,'updates_per_model':100,'frames_per_update':64,'train_frames_unique':640,'val_frames':256,
              'train_frame_exposures_per_model':6400,'lr':1e-4,'frontend':'frozen LingBot v0.5, masked_sdpa_experimental',
              'state':'reset per training clip; full 64-frame BPTT, no detach; continuous 256-frame validation',
              'split':'P000-P004 train, P005 validation; same map AbandonedFactory2',
              'labels':'visible free/surface and unknown behind first surface; no historical or dynamic labels',
              'file_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    config['task']=args.task
    config['variants']=['current','memory','memory_norm'] if args.task=='masked' else ['current','memory']
    config['labels']='verified historical free/surface queries under synthetic current RGB-D occlusion' if args.task=='masked' else config['labels']
    (OUT/'config.json').write_text(json.dumps(config,indent=2))
    print(json.dumps({'configuration':config}),flush=True)
    frontend = load_frontend('weights/model.pt',0,'masked_sdpa_experimental').cuda().eval()
    cache_root = ROOT/'features';cache_root.mkdir(exist_ok=True)
    sequences = {}
    cache_start = time.perf_counter()
    for path in paths:
        seq = load(path,'cpu')
        if args.task=='masked':
            with np.load(path) as z: seq['hidden_query']=torch.as_tensor(z['hidden_query'].copy())
        assert len(seq['rgb'])==64
        assert seq['split']==('validation' if path in val_paths else 'train')
        key = hashlib.sha256(path.read_bytes()+b'LingBot-v0.5-masked-sdpa-1200-float32').hexdigest()
        cache = cache_root/(key+'.pt')
        if cache.exists():
            features = torch.load(cache,map_location='cpu',weights_only=True)
        else:
            features=[]
            original_features=None
            if args.task=='masked':
                original=Path('prepared/tartanair-long')/path.name
                oldkey=hashlib.sha256(original.read_bytes()+b'LingBot-v0.5-masked-sdpa-1200-float32').hexdigest()
                original_features=torch.load(original.parent/'features'/(oldkey+'.pt'),map_location='cpu',weights_only=True)
            with torch.no_grad():
                for t in range(64):
                    if original_features is not None and not seq['hidden_query'][t].any():
                        features.append(original_features[t]);continue
                    rgb=seq['rgb'][t:t+1].cuda();depth=seq['depth'][t:t+1].cuda()
                    pair=frontend.forward_feat(rgb,1200,depth,enable_depth_mask=True)
                    features.append(tuple(x.cpu() for x in pair))
            torch.save(features,cache)
        seq['_features']=features;sequences[path]=seq
        print(json.dumps({'cache_complete':str(path),'elapsed_s':time.perf_counter()-cache_start}),flush=True)
    config['cache_stage_seconds']=time.perf_counter()-cache_start
    (OUT/'config.json').write_text(json.dumps(config,indent=2))
    rng=np.random.default_rng(7)
    order=np.concatenate([rng.permutation(len(train_paths)) for _ in range(10)]).tolist()
    (OUT/'update-order.json').write_text(json.dumps(order))
    validation=[sequences[p] for p in val_paths]
    for mode in config['variants']:
        torch.manual_seed(7)
        model=Perception(frontend,'memory' if mode=='memory_norm' else mode,normalize_state=mode=='memory_norm').cuda()
        parameters={k:p for k,p in model.named_parameters() if p.requires_grad}
        before={k:p.detach().cpu().clone() for k,p in parameters.items()}
        optimizer=torch.optim.AdamW(parameters.values(),lr=1e-4)
        report={'mode':mode,'trainable_parameters':sum(p.numel() for p in parameters.values()),'steps':[]}
        report['initial_validation'],_=evaluate(model,validation)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize();started=time.perf_counter()
        for step,idx in enumerate(order):
            seq=gpu(sequences[train_paths[idx]])
            model.train();frontend.eval();optimizer.zero_grad(set_to_none=True)
            loss,_=rollout(model,seq)
            if not torch.isfinite(loss): raise RuntimeError('nonfinite loss')
            loss.backward()
            grads=[p.grad for p in parameters.values() if p.grad is not None]
            if not grads or not all(torch.isfinite(g).all() for g in grads): raise RuntimeError('missing/nonfinite gradient')
            norm=torch.nn.utils.clip_grad_norm_(parameters.values(),1.).item()
            if norm==0: raise RuntimeError('zero gradient')
            optimizer.step()
            record={'update':step,'clip':train_paths[idx].stem,'loss':loss.item(),'gradient_norm':norm}
            report['steps'].append(record)
            if step%10==0 or step==99: print(json.dumps({'mode':mode,**record}),flush=True)
            del seq,loss,grads
        torch.cuda.synchronize()
        report['training_seconds']=time.perf_counter()-started
        report['gpu_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        report['parameter_delta_l2']=sum((p.detach().cpu()-before[k]).square().sum().item() for k,p in parameters.items())**.5
        report['first_epoch_mean_loss']=float(np.mean([s['loss'] for s in report['steps'][:10]]))
        report['last_epoch_mean_loss']=float(np.mean([s['loss'] for s in report['steps'][-10:]]))
        torch.save({k:p.detach().cpu() for k,p in parameters.items()},ROOT/(mode+'.pt'))
        eval_started=time.perf_counter()
        report['validation'],pred=evaluate(model,validation)
        np.savez_compressed(OUT/(mode+'.predictions.npz'),logits=pred)
        if args.task=='masked' and mode!='current':
            report['history_erased_validation'],erased=evaluate(model,validation,erase_history=True)
            np.savez_compressed(OUT/(mode+'-history-erased.predictions.npz'),logits=erased)
        if mode=='memory':
            report['reset64_validation'],pred=evaluate(model,validation,64)
            np.savez_compressed(OUT/'memory-reset64.predictions.npz',logits=pred)
        report['evaluation_seconds']=time.perf_counter()-eval_started
        (OUT/(mode+'.json')).write_text(json.dumps(report,indent=2))
        print(json.dumps({'finished':mode,'training_seconds':report['training_seconds'],
                          'first_epoch_loss':report['first_epoch_mean_loss'],'last_epoch_loss':report['last_epoch_mean_loss'],
                          'validation_loss':report['validation']['loss'],'balanced_accuracy':report['validation']['balanced_accuracy']}),flush=True)
        del model,optimizer,parameters,before
    print('EXPERIMENT_COMPLETE',flush=True)


if __name__=='__main__': main()
