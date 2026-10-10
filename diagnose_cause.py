import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from gvf.frontend import load_frontend
from gvf.model import Perception,query_loss
from train import load

torch.set_num_threads(4)
start=time.perf_counter()
net=Perception(load_frontend('weights/model.pt',backend='masked_sdpa_experimental'),'memory')
fixed=torch.load('prepared/memory-fixed.pt',weights_only=True)
params=dict(net.named_parameters())
assert set(fixed)=={k for k,p in params.items() if p.requires_grad}
with torch.no_grad():
 for k,v in fixed.items():params[k].copy_(v)
net.eval()
seq=load(Path('prepared/val.npz'),'cpu')
key=hashlib.sha256(Path('prepared/val.npz').read_bytes()+b'masked_sdpa_experimental'+b'v0.5-1200-float32').hexdigest()
features=torch.load(Path('prepared/features')/(key+'.pt'),weights_only=True)
with np.load('prepared/train.npz') as d:cap=float(d['timestamp'][-1])
result={'replay_max_abs_diff':float(np.max(np.abs(np.load('evidence/memory-cpu.predictions.npz')['logits']-np.load('evidence/memory-replay.predictions.npz')['logits']))),'elapsed_cap_seconds':cap,'conditions':{}}
with torch.no_grad():
 for name in ['original24','reset_every8','clamp_elapsed']:
  state=None;rows=[];pred=[]
  for t in range(len(seq['rgb'])):
   actual=seq['timestamp'][t:t+1];elapsed=actual.clamp(max=cap) if name=='clamp_elapsed' else actual
   delta=actual*0 if t==0 else actual-seq['timestamp'][t-1:t]
   if name=='reset_every8' and t%8==0:state=None
   logits,state,_=net(seq['rgb'][t:t+1],seq['depth'][t:t+1],seq['valid'][t:t+1],seq['K'][t:t+1],elapsed,delta,seq['query_xyz'][t:t+1],state,features[t])
   loss=query_loss(logits,seq['occupancy'][t:t+1],seq['known'][t:t+1],seq['visible'][t:t+1],seq['query_valid'][t:t+1])
   rows.append({'frame':t,'loss':loss.item(),'state_rms':state.square().mean().sqrt().item(),'elapsed_input':elapsed.item()});pred.append(logits)
  x=torch.cat(pred);c=seq['category'];balanced=sum(((x[...,0]>0)[c==k]==seq['occupancy'][c==k].bool()).float().mean().item() for k in [0,1])/2
  result['conditions'][name]={'loss':float(np.mean([r['loss'] for r in rows])),'visible_balanced_accuracy':balanced,'frames':rows}
  np.savez_compressed(f'evidence/diagnostic-{name}.npz',logits=x.numpy())
result['elapsed_seconds']=time.perf_counter()-start
Path('evidence/causal-diagnosis.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:{'loss':v['loss'],'balanced_accuracy':v['visible_balanced_accuracy'],'final_state_rms':v['frames'][-1]['state_rms']} for k,v in result['conditions'].items()},indent=2))
