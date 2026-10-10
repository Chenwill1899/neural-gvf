"""Read completed fixed-protocol reports and render cross-map evidence."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('evidence/multiscene-v1')
reports={m:json.loads((root/(m+'.json')).read_text()) for m in ['current','memory_norm']}
fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for mode,r in reports.items():
    axes[0,0].plot([e['epoch'] for e in r['epochs']],[e['mean_update_loss'] for e in r['epochs']],marker='o',label=mode)
    axes[0,1].plot([e['epoch'] for e in r['epochs']],[e['development']['macro_trajectory']['hidden']['bce'] for e in r['epochs']],marker='o',label=f'{mode}, selected epoch {r["selected_epoch"]}')
    vals=[e['metrics']['hidden']['balanced_accuracy'] for e in r['soulcity']['episodes']]
    axes[1,0].plot(range(9),vals,marker='o',label=mode)
    axes[1,1].bar(mode,r['soulcity']['macro_trajectory']['hidden']['balanced_accuracy'])
r=reports['memory_norm']
axes[1,1].scatter('memory_norm',r['soulcity_erased']['macro_trajectory']['hidden']['balanced_accuracy'],marker='x',color='black',s=90,label='paired erased history')
axes[1,1].scatter('memory_norm',r['soulcity_wrong_history']['macro_trajectory']['hidden']['balanced_accuracy'],marker='D',color='darkred',s=40,label='paired wrong history')
axes[0,0].set(title='Equal budget:304 updates / model',xlabel='Epoch (76 clips)',ylabel='Mean training update loss')
axes[0,1].set(title='Development only: checkpoint selection',xlabel='Epoch',ylabel='Hidden-query occupancy BCE')
axes[1,0].set(title='SoulCity held-out environment',xlabel='Trajectory P000–P008',ylabel='Hidden-query balanced accuracy',ylim=(0,1))
axes[1,1].set(title='SoulCity trajectory-macro average',ylabel='Hidden-query balanced accuracy',ylim=(0,1))
for ax in axes.flat:ax.grid(alpha=.2);ax.legend()
fig.suptitle('Four-map training protocol: no SoulCity model selection; verified artificial RGB-D masking')
fig.savefig(root/'cross-map-results.png',dpi=160);plt.close(fig)
summary={}
for mode,r in reports.items():
    summary[mode]={k:r[k] for k in ['parameters','selected_epoch','training_seconds','peak_gpu_allocated_bytes','selected_parameter_delta_l2','final_evaluation_seconds']}
    for key in ['soulcity','soulcity_erased','soulcity_wrong_history']:
        if key in r:summary[mode][key]=r[key]['macro_trajectory']
    summary[mode]['epoch_losses']=[e['mean_update_loss'] for e in r['epochs']]
(root/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
# Selected masked queries at actual held-out frames; raw image shown only for context.
import torch
from PIL import Image
cache=Path('prepared/multiscene-v1/feature-clips-v1')
fig,axes=plt.subplots(3,4,figsize=(13,9),layout='constrained')
for row,traj in enumerate(['P000','P004','P008']):
    episode=f'tartanair_v2/SoulCity/Data_easy/{traj}/lcam_front';name=episode.replace('/','_')
    pack=torch.load(cache/(name+'_000192.pt'),map_location='cpu',weights_only=True)
    candidates=np.flatnonzero(pack['hidden_query'].any(1).numpy());t=int(candidates[-1]);frame=192+t
    image=Image.open(Path('/home/ubuntu5090/cwl/datasets/tartanair_v2_extracted/SoulCity/Data_easy')/traj/'image_lcam_front'/f'{frame:06}_lcam_front.png').convert('RGB')
    rgb=np.asarray(image).copy();rgb[80:560,160:480]=0
    axes[row,0].imshow(rgb);axes[row,0].set_title(f'SoulCity {traj}, frame{frame}');axes[row,0].axis('off')
    mask=pack['hidden_query'][t].numpy();xyz=pack['query_xyz'][t].numpy()[mask];target=pack['occupancy'][t].numpy()[mask]
    colors=[target]
    for mode in reports:
        pred=np.load(root/(mode+'-'+name+'.predictions.npz'))['logits'][frame,mask,0]
        colors.append(1/(1+np.exp(-np.clip(pred,-40,40))))
    for col,(title,c) in enumerate(zip(['Verified historical labels','Current-frame probability','Normalized-memory probability'],colors),1):
        sc=axes[row,col].scatter(xyz[:,0],xyz[:,2],c=c,cmap='coolwarm',vmin=0,vmax=1,s=12)
        axes[row,col].set(title=title,xlabel='Camera x (m)',ylabel='Camera z (m)');axes[row,col].grid(alpha=.2)
fig.colorbar(sc,ax=axes[:,1:],label='0 = verified free; 1 = measured surface',shrink=.7)
fig.suptitle('Cross-map held-out readouts; target-time depth used only for labels, then removed from inputs')
fig.savefig(root/'soulcity-readouts.png',dpi=150);plt.close(fig)
