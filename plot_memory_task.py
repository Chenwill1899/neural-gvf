import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('evidence/tartanair-memory-task')
reports={m:json.loads((root/(m+'.json')).read_text()) for m in ['current','memory','memory_norm']}
fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for name,r in reports.items():
    axes[0,0].plot(np.arange(1,11),np.array([s['loss'] for s in r['steps']]).reshape(10,10).mean(1),label=name)
    hidden=[f for f in r['validation']['frames'] if f['hidden_balanced_accuracy'] is not None]
    axes[0,1].plot(np.array([f['frame'] for f in hidden]).reshape(8,8).mean(1),np.array([f['hidden_balanced_accuracy'] for f in hidden]).reshape(8,8).mean(1),label=name)
    axes[1,0].bar(name,r['validation']['hidden_balanced_accuracy'])
    if name!='current':
        axes[1,0].scatter(name,r['history_erased_validation']['hidden_balanced_accuracy'],color='black',marker='x',s=90)
        axes[1,1].plot([f['state_rms'] for f in r['validation']['frames']],label=name)
axes[0,0].set(title='Identical windows,100 updates each',xlabel='Epoch',ylabel='Mean update loss')
axes[0,1].set(title='Historically observed, currently masked queries',xlabel='P005 validation frame',ylabel='Hidden-query balanced accuracy',ylim=(0,1))
axes[1,0].set(title='Bars: history retained; x: history cleared',ylabel='Hidden-query balanced accuracy',ylim=(0,1))
axes[1,1].set(title='Post-update state RMS',xlabel='Validation frame',ylabel='RMS (log scale)',yscale='log')
for ax in axes.flat:ax.grid(alpha=.2)
for ax in [axes[0,0],axes[0,1],axes[1,1]]:ax.legend()
fig.suptitle('Controlled RGB-D masking; same-map trajectory holdout; one LayerNorm change')
fig.savefig(root/'memory-comparison.png',dpi=160);plt.close(fig)
config=json.loads((root/'config.json').read_text())
preds={m:np.load(root/(m+'.predictions.npz'))['logits'] for m in ['current','memory_norm']}
fig,axes=plt.subplots(3,5,figsize=(16,9),layout='constrained')
for row,t in enumerate([3,127,255]):
    with np.load(config['val_paths'][t//64]) as z:
        i=t%64;mask=z['hidden_query'][i];p=z['query_xyz'][i,mask];target=z['occupancy'][i,mask]
        axes[row,0].imshow(z['rgb'][i-1]);axes[row,1].imshow(z['rgb'][i])
    axes[row,0].set_title(f'Previous input, frame {t-1}');axes[row,1].set_title(f'Masked current input, frame {t}')
    axes[row,0].axis('off');axes[row,1].axis('off')
    for col,(label,c) in enumerate([('Verified historical labels',target)]+[(m,1/(1+np.exp(-np.clip(preds[m][t,mask,0],-40,40)))) for m in ['current','memory_norm']],2):
        sc=axes[row,col].scatter(p[:,0],p[:,2],c=c,cmap='coolwarm',vmin=0,vmax=1,s=12)
        axes[row,col].set(title=label,xlabel='Camera x (m)',ylabel='Camera z (m)');axes[row,col].grid(alpha=.2)
fig.colorbar(sc,ax=axes[:,2:],label='0 = verified free; 1 = surface band',shrink=.7)
fig.suptitle('Artificial current RGB-D removal; previous observations intact; poses used for labels only')
fig.savefig(root/'masked-query-readouts.png',dpi=150);plt.close(fig)
summary={}
for m,r in reports.items():
    summary[m]={k:r[k] for k in ['first_epoch_mean_loss','last_epoch_mean_loss','training_seconds','gpu_peak_allocated_bytes','parameter_delta_l2']}
    for key in ['initial_validation','validation','history_erased_validation']:
        if key in r: summary[m][key]={k:r[key][k] for k in ['loss','balanced_accuracy','hidden_balanced_accuracy']}
    summary[m]['final_state_rms']=r['validation']['frames'][-1]['state_rms']
(root/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
