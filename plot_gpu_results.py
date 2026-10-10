import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

names=['current','memory']
reports={n:json.loads(Path(f'evidence/{n}24-cuda.json').read_text()) for n in names}
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for name,r in reports.items():
 losses=[s['loss'] for s in r['steps']]+[r['final_train_loss']]
 axes[0].plot(range(len(losses)),losses,label=name)
 axes[1].bar(name,r['val_loss'])
axes[0].set(xlabel='Optimizer updates',ylabel='Surface/free BCE + current-evidence BCE',title='Real RGB-D optimization (GPU, frozen frontend; 24 train frames)')
axes[0].legend();axes[0].grid(alpha=.25)
axes[1].set(ylabel='Loss',title='Same warehouse; held-out obstacle layout, 24 frames')
fig.savefig('evidence/gpu-loss-curves.png',dpi=170)
fig.savefig('evidence/gpu-loss-curves.svg')
plt.close(fig)
with np.load('prepared/val.npz',allow_pickle=False) as data:
 rgb=data['rgb']; xyz=data['query_xyz']; known=data['known']; occ=data['occupancy']
pred={n:np.load(f'evidence/{n}24-cuda.predictions.npz',allow_pickle=False)['logits'] for n in names}
fig,axes=plt.subplots(3,4,figsize=(13,9),layout='constrained')
for row,t in enumerate([0,12,23]):
 axes[row,0].imshow(rgb[t]);axes[row,0].set_title(f'Validation RGB / frame {t}');axes[row,0].axis('off')
 mask=known[t];p=xyz[t,mask]
 colors=[occ[t,mask]]+[1/(1+np.exp(-pred[n][t,mask,0])) for n in names]
 for col,(label,c) in enumerate(zip(['Supervised surface/free','Current probability','Memory probability'],colors),1):
  scatter=axes[row,col].scatter(p[:,0],p[:,2],c=c,cmap='coolwarm',vmin=0,vmax=1,s=10)
  axes[row,col].set(xlabel='Camera x (m)',ylabel='Camera z (m)',title=label)
  axes[row,col].grid(alpha=.2)
fig.colorbar(scatter,ax=axes[:,1:],label='0 = observed free; 1 = surface band',shrink=.7)
fig.suptitle('Known queries only; unknowns excluded. Same warehouse, changed obstacles; one validation motion case.')
fig.savefig('evidence/gpu-heldout-readouts.png',dpi=150)
plt.close(fig)
summary={}
for n,r in reports.items():
 c=r['val_by_category'];summary[n]={'initial_loss':r['initial_train_loss'],'final_loss':r['final_train_loss'],'heldout_loss':r['val_loss'],'parameter_delta':r['parameter_delta_l2'],'visible_surface_free_balanced_accuracy':(c['0']['surface_free_accuracy']+c['1']['surface_free_accuracy'])/2,'unknown_current_evidence_accuracy':c['2']['current_evidence_accuracy'],'known_surface_free_accuracy':r['val_known_occupancy_accuracy'],'grad_min':min(s['grad_norm'] for s in r['steps']),'grad_max':max(s['grad_norm'] for s in r['steps'])}
Path('evidence/gpu-comparison.json').write_text(json.dumps(summary,indent=2))
