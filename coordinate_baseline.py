"""Nonlinear coordinate-only audit: fit training trajectories, score development."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from matched_sampling import DEST,OUT
from multiscene_experiment import ROOT,CACHE,key


def auc(y,s):
    order=np.argsort(s,kind='stable');v=s[order];ys=y[order]
    _,starts,counts=np.unique(v,return_index=True,return_counts=True)
    ranks=np.repeat(starts+(counts+1)/2,counts);n1=int(y.sum());n0=len(y)-n1
    return float((ranks[ys==1].sum()-n1*(n1+1)/2)/(n1*n0))


def collect(variant):
    inventory=json.loads((OUT/'label-inventory.json').read_text());groups={}
    assert sum(w['split']=='train' for w in inventory)==76
    assert sum(w['split']=='development' for w in inventory)==12
    for w in inventory:
        path=CACHE/key(w) if variant=='original' else DEST/'labels'/key(w)
        p=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        mask=p['hidden_query']&p['query_valid'];xyz=p['query_xyz'][mask];y=p['occupancy'][mask]
        z=xyz[:,2:3];x=torch.cat((xyz,xyz[:,:2]/z,z.log()),1)
        label='train' if w['split']=='train' else w['episode_id']
        groups.setdefault(label,[[],[]]);groups[label][0].append(x);groups[label][1].append(y)
    return {k:(torch.cat(v[0]),torch.cat(v[1])) for k,v in groups.items()}


def metric(y,logits):
    a=y.numpy();s=logits.numpy();pred=s>0
    return {'queries':len(a),'bce':F.binary_cross_entropy_with_logits(logits,y).item(),
            'auc':auc(a,s),'balanced_accuracy':float(np.mean([np.mean(pred[a==c]==(c==1)) for c in [0,1]]))}


def main():
    torch.set_num_threads(4);reports={}
    for variant in ['original','matched']:
        torch.manual_seed(7);groups=collect(variant);x,y=groups.pop('train')
        mean=x.mean(0);std=x.std(0).clamp_min(1e-5);x=(x-mean)/std
        z_auc={k:auc(b.numpy(),a[:,2].numpy()) for k,(a,b) in groups.items()}
        groups={k:((a-mean)/std,b) for k,(a,b) in groups.items()}
        model=nn.Sequential(nn.Linear(6,64),nn.GELU(),nn.Linear(64,64),nn.GELU(),nn.Linear(64,1))
        opt=torch.optim.AdamW(model.parameters(),lr=1e-3);best=float('inf');epochs=[];best_weights=None
        for epoch in range(1,21):
            model.train();losses=[]
            for idx in torch.randperm(len(x)).split(2048):
                opt.zero_grad(set_to_none=True);loss=F.binary_cross_entropy_with_logits(model(x[idx])[:,0],y[idx]);loss.backward();opt.step();losses.append(loss.item())
            model.eval()
            with torch.no_grad():scores={k:metric(b,model(a)[:,0]) for k,(a,b) in groups.items()}
            macro={k:float(np.mean([r[k] for r in scores.values()])) for k in ['bce','auc','balanced_accuracy']}
            row={'epoch':epoch,'train_loss':float(np.mean(losses)),'macro_development':macro,'episodes':scores};epochs.append(row)
            if macro['bce']<best:
                best=macro['bce'];selected=row;best_weights={k:v.clone() for k,v in model.state_dict().items()}
            if epoch%5==0:print(json.dumps({'coordinate_variant':variant,**row}),flush=True)
        torch.save({'model':best_weights,'mean':mean,'std':std},DEST/(variant+'-coordinate.pt'))
        reports[variant]={'train_queries':len(x),'features':['x','y','z','x/z','y/z','log(z)'],
            'architecture':'6-64-64-1 GELU MLP, seed7, AdamW1e-3, 20 epochs, batch2048; no image/history/index input',
            'selection':'minimum development macro BCE; all epochs retained','z_auc':z_auc,'selected':selected,'epochs':epochs}
        (OUT/'coordinate-baselines.json').write_text(json.dumps(reports,indent=2))
    reduction=reports['original']['selected']['macro_development']['auc']-reports['matched']['selected']['macro_development']['auc']
    verdict={'coordinate_mlp_auc_reduction':reduction,'matched_macro_auc':reports['matched']['selected']['macro_development']['auc'],
             'scope':'Two finite coordinate-only probes, not proof that all coordinate leakage is absent',
             'proceed_rule':'Matched nonlinear macro AUC <=0.60 and lower than original; otherwise inspect sampling before RGB-D training'}
    verdict['proceed']=verdict['matched_macro_auc']<=.60 and reduction>0
    (OUT/'coordinate-verdict.json').write_text(json.dumps(verdict,indent=2));print(json.dumps({'VERDICT':verdict}),flush=True)


if __name__=='__main__':main()
