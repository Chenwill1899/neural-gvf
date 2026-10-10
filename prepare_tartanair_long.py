from pathlib import Path
import json, subprocess
root=Path('/home/ubuntu5090/cwl/neural-gvf')
inv=json.loads((root/'prepared/tartanair-starter/inventory.json').read_text())
clips=[]
for t,r in inv['trajectories'].items():
 if t!='P005': clips += [f'{t}:0:64',f'{t}:{r["frames"]//2}:64']
clips += [f'P005:{s}:64' for s in [0,64,128,192]]
subprocess.run(['/home/ubuntu5090/miniconda3/envs/end2end/bin/python','adapt_tartanair.py','--source','/home/ubuntu5090/cwl/datasets/tartanair_v2','--output','prepared/tartanair-long','--clips',*clips],cwd=root,check=True)
