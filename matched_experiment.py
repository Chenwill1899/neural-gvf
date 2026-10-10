"""Equal additional-budget comparison; models/loss unchanged, labels replaced."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
import multiscene_experiment as base
from gvf.model import Perception,CurrentWithMemory,query_loss
from matched_sampling import DEST,OUT

ANCHOR=base.ROOT/'current-selected.pt'


def write(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False))


def build(mode):
    torch.manual_seed(7);current=Perception(None,'current')
    current.load_state_dict(torch.load(ANCHOR,map_location='cpu',weights_only=True),strict=True)
    return (current if mode=='current' else CurrentWithMemory(current)).cuda()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    assert json.loads((OUT/'coordinate-verdict.json').read_text())['proceed'], 'Coordinate audit did not pass; no RGB-D training'
    assert torch.cuda.is_available(), 'Use approved host CUDA context'
    torch.set_num_threads(4);base.DEVICE='cuda';base.OUT=OUT
    original_get=base.get;original_measure=base.measure
    def get(w):
        seq=original_get(w)
        seq.update({k:v.cuda() for k,v in torch.load(DEST/'labels'/base.key(w),map_location='cpu',weights_only=True).items()})
        return seq
    def measure(logits,seq,t):
        return original_measure(logits,seq,t) if seq['query_valid'][t].any() else {'evidence_accuracy':None}
    base.get=get;base.measure=measure
    old=json.loads(Path('evidence/multiscene-v1/config.json').read_text())
    train=[w for w in json.loads((base.ROOT/'windows.json').read_text()) if w['split']=='train']
    dev=old['dev_episodes'];records={r['episode_id']:r for r in json.loads((base.ROOT/'manifest.json').read_text())['trajectories']}
    config={'variants':['current','current_direct_memory'],'seed':7,'epochs':4,'updates_per_variant':304,
            'lr':1e-4,'bptt_frames':64,'order':old['train_order'],'development':dev,
            'anchor_sha256':hashlib.sha256(ANCHOR.read_bytes()).hexdigest(),
            'initialization':'Both start at existing current-selected; current is trainable, residual freezes same anchor and adds fresh zero-initialized memory residual',
            'fairness':'Equal additional data/order/updates, same pretrained anchor; unequal trainable capacity and compute',
            'models_and_loss':'Unchanged gvf/model.py; only hidden label overlay differs',
            'selection':'Minimum development macro hidden BCE subject to visible BCE <= initial current; epoch-zero exact-current fallback; ties earlier',
            'evaluation':'Development only; memory paired erase and equal-age Office/P000 donor; current has no recurrent history',
            'no_soulcity_hospital_sewerage':True}
    if args.resume and (OUT/'training-config.json').exists():assert json.loads((OUT/'training-config.json').read_text())==config
    else:write('training-config.json',config)
    anchor=torch.load(ANCHOR,map_location='cpu',weights_only=True)
    baseline=build('current');baseline.eval();baseline_report=base.evaluate(baseline,dev,records,'anchor');del baseline
    write('initial-current.json',baseline_report)
    for mode in config['variants']:
        result_path=OUT/(mode+'.json')
        if args.resume and result_path.exists():continue
        model=build(mode);params={k:p for k,p in model.named_parameters() if p.requires_grad}
        initial={k:p.detach().cpu().clone() for k,p in params.items()}
        optimizer=torch.optim.AdamW(params.values(),lr=1e-4)
        report={'mode':mode,'trainable_parameters':sum(p.numel() for p in params.values()),'epochs':[],
                'selected_epoch':0,'best_hidden_bce':baseline_report['macro_trajectory']['hidden']['bce']}
        selected=DEST/(mode+'-selected.pt');latest=DEST/(mode+'-latest.pt')
        epoch_start=offset_start=0;steps=[];seconds=0.
        if args.resume and latest.exists():
            ck=torch.load(latest,map_location='cuda',weights_only=True);model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer'])
            report=ck['report'];epoch_start=ck['epoch'];offset_start=ck['offset'];steps=ck['steps'];seconds=ck['seconds']
        else:
            model.eval()
            if mode=='current_direct_memory':
                seq=get(train[0]);state=None
                with torch.no_grad():
                    for t in range(4):
                        direct,_=base.step(model.current,seq,t,None,t);actual,state=base.step(model,seq,t,state,t)
                        assert torch.equal(direct,actual),'Matched queries changed zero-residual equivalence'
                write('matched-query-equivalence.json',{'max_logit_difference':0.,'frames':4,'window':train[0],'exact':True})
                del seq,state
            torch.save(model.state_dict(),selected)
        visible_limit=baseline_report['macro_trajectory']['visible']['bce']
        def checkpoint(epoch,offset,rows,elapsed):
            torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'report':report,
                        'epoch':epoch,'offset':offset,'steps':rows,'seconds':elapsed},latest.with_suffix('.tmp'))
            latest.with_suffix('.tmp').replace(latest);write(mode+'-progress.json',report)
        torch.cuda.reset_peak_memory_stats()
        for epoch in range(epoch_start,4):
            model.train();started=time.perf_counter()
            for offset,index in enumerate(config['order'][epoch]):
                if epoch==epoch_start and offset<offset_start:continue
                seq=get(train[index]);optimizer.zero_grad(set_to_none=True);state=None;terms=[]
                for t in range(64):
                    logits,state=base.step(model,seq,t,state,t)
                    terms.append(query_loss(logits,*(seq[k][t:t+1] for k in ['occupancy','known','visible','query_valid'])))
                loss=torch.stack(terms).mean();assert torch.isfinite(loss);loss.backward()
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in params.values())
                norm=torch.nn.utils.clip_grad_norm_(params.values(),1.).item();assert norm>0
                optimizer.step();steps.append({'window':base.key(train[index]),'loss':loss.item(),'gradient_norm':norm})
                del seq,terms,state,logits,loss
                if (offset+1)%19==0:
                    checkpoint(epoch,offset+1,steps,seconds+time.perf_counter()-started)
                    print(json.dumps({'mode':mode,'epoch':epoch+1,'update':offset+1,'loss':steps[-1]['loss']}),flush=True)
            seconds+=time.perf_counter()-started
            if mode=='current_direct_memory':assert all(torch.equal(v.cpu(),anchor[k]) for k,v in model.current.state_dict().items())
            development=base.evaluate(model,dev,records,mode);metrics=development['macro_trajectory']
            eligible=metrics['visible']['bce']<=visible_limit
            if eligible and metrics['hidden']['bce']<report['best_hidden_bce']:
                report['selected_epoch']=epoch+1;report['best_hidden_bce']=metrics['hidden']['bce'];torch.save(model.state_dict(),selected)
            report['epochs'].append({'epoch':epoch+1,'mean_loss':float(np.mean([r['loss'] for r in steps])),
                                    'steps':steps,'development':development,'visible_guard':eligible})
            steps=[];offset_start=0;checkpoint(epoch+1,0,steps,seconds)
            print(json.dumps({'mode':mode,'epoch_done':epoch+1,'development':metrics,'selected_epoch':report['selected_epoch']}),flush=True)
        report['training_seconds']=seconds;report['peak_gpu_allocated_bytes']=torch.cuda.max_memory_allocated()
        weights=torch.load(selected,map_location='cuda',weights_only=True);model.load_state_dict(weights)
        report['selected_parameter_delta_l2']=sum((p.detach().cpu()-initial[k]).square().sum().item() for k,p in params.items())**.5
        report['development']=base.evaluate(model,dev,records,mode)
        if mode=='current_direct_memory':
            assert all(torch.equal(v.cpu(),anchor[k]) for k,v in model.current.state_dict().items());report['frozen_anchor_unchanged']=True
            report['development_erased']=base.evaluate(model,dev,records,mode,erase=True)
            report['development_wrong_history']=base.evaluate(model,dev,records,mode,donor='tartanair_v2/Office/Data_easy/P000/lcam_front')
        else:report['history_controls']='Identical by construction: current has no state input dependence'
        write(mode+'.json',report);print(json.dumps({'FINISHED':mode,'selected_epoch':report['selected_epoch'],'development':report['development']['macro_trajectory']}),flush=True)
        del model,optimizer,params,initial,weights
    write('status.json',{'state':'complete','scope':'training/development only; no final held-out test','models_and_loss_unchanged':True})
    print('MATCHED_EXPERIMENT_COMPLETE',flush=True)


if __name__=='__main__':main()
