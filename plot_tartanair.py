"""Plot the fixed-budget long-sequence experiment, without additional training."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('evidence/tartanair-long')
reports={m:json.loads((root/(m+'.json')).read_text()) for m in ['current','memory']}
fig,axes=plt.subplots(1,3,figsize=(15,4.3),layout='constrained')
for mode,r in reports.items():
    epochs=np.array([s['loss'] for s in r['steps']]).reshape(10,10).mean(1)
    axes[0].plot(np.arange(1,11),epochs,marker='o',label=mode)
    vals=np.array([f['balanced_accuracy'] for f in r['validation']['frames']])
    axes[1].plot(np.arange(256).reshape(16,16).mean(1),vals.reshape(16,16).mean(1),label=mode)
reset=reports['memory']['reset64_validation']
vals=np.array([f['balanced_accuracy'] for f in reset['frames']])
axes[1].plot(np.arange(256).reshape(16,16).mean(1),vals.reshape(16,16).mean(1),label='memory reset / 64',linestyle='--')
for label,report in [('continuous',reports['memory']['validation']),('reset / 64',reset)]:
    axes[2].plot([f['state_rms'] for f in report['frames']],label=label)
axes[0].set(title='Same 640 frames; 100 updates each',xlabel='Epoch (10 clips, 64 frames each)',ylabel='Mean update loss')
axes[1].set(title='P005: 256 unseen contiguous frames',xlabel='Validation frame (16-frame bins)',ylabel='Visible free/surface balanced accuracy',ylim=(0,1))
axes[2].set(title='Fixed memory weights: state growth',xlabel='Validation frame',ylabel='Slot state RMS')
for ax in axes: ax.grid(alpha=.2);ax.legend()
fig.suptitle('TartanAir: same-map trajectory holdout; frozen LingBot; no dynamic or occlusion-memory claims')
fig.savefig(root/'training-and-state.png',dpi=160);plt.close(fig)
config=json.loads((root/'config.json').read_text())
pred={m:np.load(root/(m+'.predictions.npz'))['logits'] for m in ['current','memory']}
fig,axes=plt.subplots(3,4,figsize=(13,9),layout='constrained')
for row,t in enumerate([0,127,255]):
    with np.load(config['val_paths'][t//64]) as z:
        i=t%64;rgb=z['rgb'][i];mask=z['known'][i];xyz=z['query_xyz'][i,mask];target=z['occupancy'][i,mask]
    axes[row,0].imshow(rgb);axes[row,0].set_title(f'P005 RGB, frame {t}');axes[row,0].axis('off')
    colors=[target]+[1/(1+np.exp(-np.clip(pred[m][t,mask,0],-40,40))) for m in ['current','memory']]
    for col,(label,c) in enumerate(zip(['Measured free/surface','Current prediction','Continuous memory'],colors),1):
        sc=axes[row,col].scatter(xyz[:,0],xyz[:,2],c=c,cmap='coolwarm',vmin=0,vmax=1,s=10)
        axes[row,col].set(title=label,xlabel='Camera x (m)',ylabel='Camera z (m)');axes[row,col].grid(alpha=.2)
fig.colorbar(sc,ax=axes[:,1:],label='0 = measured free; 1 = surface band',shrink=.7)
fig.suptitle('TartanAir held-out trajectory; unknown queries excluded; no poses supplied to model')
fig.savefig(root/'readouts.png',dpi=150);plt.close(fig)
summary={m:{k:r[k] for k in ['first_epoch_mean_loss','last_epoch_mean_loss','training_seconds','evaluation_seconds','gpu_peak_allocated_bytes','parameter_delta_l2','trainable_parameters']} for m,r in reports.items()}
for m,r in reports.items():
    summary[m]['validation']={k:r['validation'][k] for k in ['loss','balanced_accuracy','chunks']}
summary['memory']['reset64_validation']={k:reset[k] for k in ['loss','balanced_accuracy','chunks']}
(root/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
