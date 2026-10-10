"""Fixed-weight, equal-age wrong-history control on first64 held-out frames."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from train import load
from gvf.frontend import load_frontend
from gvf.model import Perception
from tartanair_experiment import gpu,metrics
root=Path('prepared/tartanair-memory-task');out=Path('evidence/tartanair-memory-task')
torch.set_num_threads(4);torch.manual_seed(7)
model=Perception(load_frontend('weights/model.pt',0,'masked_sdpa_experimental'),'memory',normalize_state=True).cuda().eval()
weights=torch.load(root/'memory_norm.pt',map_location='cpu',weights_only=True)
assert set(weights)=={k for k,p in model.named_parameters() if p.requires_grad}
model.load_state_dict(weights,strict=False)
def read(name):
 path=root/name;seq=load(path,'cpu')
 with np.load(path) as z:seq['hidden_query']=torch.as_tensor(z['hidden_query'].copy())
 key=hashlib.sha256(path.read_bytes()+b'LingBot-v0.5-masked-sdpa-1200-float32').hexdigest()
 seq['_features']=torch.load(root/'features'/(key+'.pt'),map_location='cpu',weights_only=True)
 return gpu(seq)
def forward(seq,t,state):
 time=seq['timestamp'][t:t+1];delta=time*0 if t==0 else time*0+.1
 return model(seq['rgb'][t:t+1],seq['depth'][t:t+1],seq['valid'][t:t+1],seq['K'][t:t+1],time,delta,seq['query_xyz'][t:t+1],state,seq['_features'][t])[:2]
val=read('P005_000000_064.npz');donor=read('P004_000000_064.npz')
state=donor_state=None;records=[]
with torch.no_grad():
 for t in range(64):
  if val['hidden_query'][t].any():
   wrong,_=forward(val,t,donor_state)
  correct,state=forward(val,t,state)
  _,donor_state=forward(donor,t,donor_state)
  if val['hidden_query'][t].any():
   labels={k:v[t:t+1] for k,v in val.items() if isinstance(v,torch.Tensor)}
   records.append({'frame':t,'correct':metrics(correct,labels)['hidden_balanced_accuracy'],
                   'wrong_history':metrics(wrong,labels)['hidden_balanced_accuracy']})
result={'scope':'P005 first64frames,16masked targets; donor P004 first64, same step age/time/normalization; no state reset; original validation recurrence preserved',
        'correct':float(np.mean([r['correct'] for r in records])),
        'wrong_history':float(np.mean([r['wrong_history'] for r in records])),'frames':records}
(out/'history-content-control.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
