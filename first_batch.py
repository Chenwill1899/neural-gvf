import argparse,json,time
from pathlib import Path
import torch
from gvf.frontend import load_frontend
from gvf.model import Perception,query_loss
from train import load

parser=argparse.ArgumentParser()
parser.add_argument('--device',default='cpu')
args=parser.parse_args()
torch.set_num_threads(4)
torch.manual_seed(7)
start=time.perf_counter()
model=Perception(load_frontend('weights/model.pt',1,'masked_sdpa_experimental'),'current').to(args.device)
model.frontend.eval()
s=load(Path('prepared/train.npz'),args.device)
params={k:p for k,p in model.named_parameters() if p.requires_grad}
before={k:p.detach().clone() for k,p in params.items()}
opt=torch.optim.AdamW(params.values(),lr=1e-5)
def loss():
 logits,_,_=model(s['rgb'][:1],s['depth'][:1],s['valid'][:1],s['K'][:1],s['timestamp'][:1],s['timestamp'][:1]*0,s['query_xyz'][:1])
 return query_loss(logits,s['occupancy'][:1],s['known'][:1],s['visible'][:1],s['query_valid'][:1])
l=loss();l.backward()
grads={k:p.grad for k,p in params.items() if p.grad is not None}
assert all(torch.isfinite(g).all() for g in grads.values())
frontend_norm=sum(g.float().square().sum().item() for k,g in grads.items() if k.startswith('frontend.'))**.5
assert frontend_norm>0
opt.step()
with torch.no_grad(): after=loss().item()
if args.device.startswith('cuda'): torch.cuda.synchronize()
report={'device':args.device,'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else 0,'scope':'one real frame; final encoder block and output projections + current readout trainable; not full backbone fine-tuning','loss_before':l.item(),'loss_after':after,'frontend_grad_norm':frontend_norm,'all_gradients_finite':True,'gradient_norm':sum(g.float().square().sum().item() for g in grads.values())**.5,'parameter_delta_l2':sum((p.detach()-before[k]).square().sum().item() for k,p in params.items())**.5,'elapsed_seconds':time.perf_counter()-start}
Path('evidence/first-batch-'+args.device.replace(':','-')+'.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
