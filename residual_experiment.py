"""Current-direct ablation on existing real RGB-D caches; no frontend rerun."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

import multiscene_experiment as base
from gvf.model import CurrentWithMemory, Perception, query_loss

OUT = Path('evidence/current-direct-v1')
SAVE = Path('prepared/current-direct-v1')
ANCHOR = base.ROOT / 'current-selected.pt'


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False))


def build(device):
    torch.manual_seed(7)
    current = Perception(None, 'current')
    current.load_state_dict(torch.load(ANCHOR, map_location='cpu', weights_only=True), strict=True)
    return CurrentWithMemory(current).to(device)


def frozen_equal(model, anchor):
    return all(torch.equal(v.cpu(), anchor[k]) for k, v in model.current.state_dict().items())


def check(device, window):
    """One real four-frame optimization check, including a masked frame."""
    model = build(device)
    anchor = {k: v.detach().cpu().clone() for k, v in model.current.state_dict().items()}
    seq = base.get(window)
    assert seq['hidden_query'][3].any(), 'Need an actual hidden-query frame'
    model.eval()
    initial_max_error = 0.
    with torch.no_grad():
        state = None
        for t in range(4):
            baseline, _ = base.step(model.current, seq, t, None, t)
            result, state = base.step(model, seq, t, state, t)
            assert torch.equal(result, baseline), 'Zero residual must be bitwise baseline-equivalent'
            initial_max_error = max(initial_max_error, (result-baseline).abs().max().item())
    params = [p for p in model.parameters() if p.requires_grad]
    before = {k: p.detach().cpu().clone() for k, p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(params, lr=1e-4)
    losses, gradients = [], []
    # Zero output initialization blocks upstream gradients on update one only.
    for update in range(3):
        model.train(); optimizer.zero_grad(set_to_none=True); state = None; terms = []
        for t in range(4):
            result, state = base.step(model, seq, t, state, t)
            terms.append(query_loss(result, *(seq[k][t:t+1] for k in
                ['occupancy', 'known', 'visible', 'query_valid'])))
        loss = torch.stack(terms).mean()
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in params)
        norm = torch.nn.utils.clip_grad_norm_(params, 1.).item()
        upstream = sum(p.grad.square().sum().item() for p in model.memory.parameters() if p.grad is not None)**.5
        assert norm > 0
        if update > 0:
            assert upstream > 0, 'Memory must receive gradients after the output layer opens'
        optimizer.step(); losses.append(loss.item()); gradients.append({'total': norm, 'memory': upstream})
    model.eval()
    with torch.no_grad():
        baseline, _ = base.step(model.current, seq, 3, None, 3)
        enabled, _ = base.step(model, seq, 3, None, 3)
        assert torch.equal(enabled[..., 1], baseline[..., 1]), 'Evidence must remain current-only'
        model.use_memory = False
        disabled, _ = base.step(model, seq, 3, None, 3)
        assert torch.equal(disabled, baseline), 'Disabled history must exactly recover baseline after training'
    assert frozen_equal(model, anchor)
    delta = sum((p.detach().cpu()-before[k]).square().sum().item()
                for k,p in model.named_parameters() if k in before)**.5
    assert delta > 0
    report = {'device': device, 'window': window, 'real_frames': 4, 'updates': 3,
              'zero_residual_max_logit_error': initial_max_error,
              'disabled_equals_baseline_after_updates': True,
              'current_evidence_unchanged': True, 'frozen_anchor_unchanged': True,
              'losses': losses, 'gradient_norms': gradients, 'parameter_delta_l2': delta,
              'scope': 'Necessary equivalence and short optimization check, not full training or generalization'}
    write('equivalence-and-gradients.json', report)
    print(json.dumps({'CHECK': report}), flush=True)
    del model, seq, optimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); SAVE.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); base.DEVICE = args.device; base.OUT = OUT
    old_config = json.loads(Path('evidence/multiscene-v1/config.json').read_text())
    manifest = json.loads((base.ROOT/'manifest.json').read_text())
    records = {r['episode_id']: r for r in manifest['trajectories']}
    train = [w for w in json.loads((base.ROOT/'windows.json').read_text()) if w['split']=='train']
    dev, val = old_config['dev_episodes'], old_config['validation_episodes']
    config = {'seed': 7, 'epochs': 4, 'updates': 304, 'lr': 1e-4, 'bptt_frames': 64,
              'train_order': old_config['train_order'], 'train_windows': train, 'development': dev,
              'historical_comparison_only': val,
              'anchor': str(ANCHOR), 'anchor_sha256': hashlib.sha256(ANCHOR.read_bytes()).hexdigest(),
              'initialization': 'Frozen selected current baseline; fresh normalized memory; zero residual output',
              'selection': 'Lowest development hidden BCE among epochs with visible BCE no worse than frozen current; epoch 0 is exact-current fallback; ties keep earlier',
              'objective': 'Existing known surface/free BCE plus unchanged current-evidence BCE; same frame weighting and order as previous experiment',
              'fairness': 'Same 304-update data budget as previous runs, plus previously trained frozen current anchor; not equal total compute/capacity',
              'controls': 'Selected checkpoint paired erase and equal-age Office/P000 wrong history on development and historical SoulCity',
              'independent_final_plan': 'Lock architecture/checkpoint before accessing Hospital or Sewerage outcomes; use all eligible trajectories first 256 consecutive frames, no tuning afterward; artificial short occlusion still limits conclusions',
              'no_gt_pose_input': True, 'no_motion_alignment_added': True}
    if args.resume:
        assert json.loads((OUT/'config.json').read_text()) == config, 'Resume protocol mismatch'
    else:
        write('config.json', config)
    if args.device == 'cuda' and not torch.cuda.is_available():
        if not (SAVE/'initial.pt').exists():
            torch.save(build('cpu').state_dict(), SAVE/'initial.pt')
        write('status.json', {'state': 'blocked_before_training', 'reason': 'PyTorch CUDA unavailable; do not modify driver',
                              'initial_checkpoint': str(SAVE/'initial.pt'),
                              'resume_command': 'python residual_experiment.py --device cuda --resume'})
        raise SystemExit('CUDA unavailable: protocol saved; no full training started')
    check(args.device, train[0])
    if args.check_only:
        return
    model = build(args.device)
    anchor = {k:v.detach().cpu().clone() for k,v in model.current.state_dict().items()}
    params = {k:p for k,p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(params.values(), lr=1e-4)
    report = {'trainable_parameters': sum(p.numel() for p in params.values()),
              'frozen_parameters': sum(p.numel() for p in model.current.parameters()), 'epochs': []}
    next_epoch, next_step, steps, seconds = 0, 0, [], 0.
    if args.resume and (SAVE/'latest.pt').exists():
        saved = torch.load(SAVE/'latest.pt', map_location=args.device, weights_only=True)
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        report = saved['report']
        next_epoch, next_step, steps = saved['next_epoch'], saved['next_step'], saved['steps']
        seconds = saved['seconds']
    else:
        report['baseline_development'] = base.evaluate(model.current, dev, records, 'current')
        report['selected_epoch'] = 0
        report['best_hidden_bce'] = report['baseline_development']['macro_trajectory']['hidden']['bce']
        torch.save(model.state_dict(), SAVE/'selected.pt')
    visible_limit = report['baseline_development']['macro_trajectory']['visible']['bce']

    def checkpoint(epoch, offset, rows, elapsed):
        # A single atomic checkpoint contains progress too, so a crash cannot mix epochs.
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'next_epoch': epoch, 'next_step': offset, 'steps': rows, 'seconds': elapsed,
                    'report': report}, SAVE/'latest.tmp')
        (SAVE/'latest.tmp').replace(SAVE/'latest.pt')
        write('progress.json', report)

    for epoch in range(next_epoch, 4):
        model.train(); started = time.perf_counter()
        for offset, index in enumerate(config['train_order'][epoch]):
            if epoch == next_epoch and offset < next_step:
                continue
            seq = base.get(train[index]); optimizer.zero_grad(set_to_none=True); state = None; terms = []
            for t in range(64):
                logits, state = base.step(model, seq, t, state, t)
                terms.append(query_loss(logits, *(seq[k][t:t+1] for k in
                    ['occupancy', 'known', 'visible', 'query_valid'])))
            loss = torch.stack(terms).mean(); assert torch.isfinite(loss); loss.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in params.values())
            norm = torch.nn.utils.clip_grad_norm_(params.values(), 1.).item(); assert norm > 0
            optimizer.step()
            steps.append({'window': base.key(train[index]), 'loss': loss.item(), 'grad_norm': norm})
            del seq, terms, state, logits, loss
            if (offset+1) % 19 == 0:
                checkpoint(epoch, offset+1, steps, seconds+time.perf_counter()-started)
                print(json.dumps({'epoch': epoch+1, 'update': offset+1, 'loss': steps[-1]['loss']}), flush=True)
        seconds += time.perf_counter()-started
        assert frozen_equal(model, anchor)
        development = base.evaluate(model, dev, records, 'residual')
        metrics = development['macro_trajectory']
        eligible = metrics['visible']['bce'] <= visible_limit
        if eligible and metrics['hidden']['bce'] < report['best_hidden_bce']:
            report['selected_epoch'] = epoch+1; report['best_hidden_bce'] = metrics['hidden']['bce']
            torch.save(model.state_dict(), SAVE/'selected.pt')
        report['epochs'].append({'epoch': epoch+1, 'mean_loss': float(np.mean([r['loss'] for r in steps])),
                                 'steps': steps, 'development': development, 'visible_guard_passed': eligible})
        report['training_seconds'] = seconds
        steps = []; next_step = 0; checkpoint(epoch+1, 0, steps, seconds)
        print(json.dumps({'epoch_done': epoch+1, 'development': metrics,
                          'selected_epoch': report['selected_epoch'], 'visible_guard': eligible}), flush=True)
    model.load_state_dict(torch.load(SAVE/'selected.pt', map_location=args.device, weights_only=True))
    assert frozen_equal(model, anchor)
    donor = 'tartanair_v2/Office/Data_easy/P000/lcam_front'
    for label, episodes in [('development', dev), ('historical_soulcity', val)]:
        report[label] = base.evaluate(model, episodes, records, 'residual')
        report[label+'_erased'] = base.evaluate(model, episodes, records, 'residual', erase=True)
        report[label+'_wrong_history'] = base.evaluate(model, episodes, records, 'residual', donor=donor)
        write('results.json', report)
    write('status.json', {'state': 'complete', 'selected_epoch': report['selected_epoch'],
                          'independent_final_validation': 'Not run; Hospital/Sewerage remain untouched'})
    print('CURRENT_DIRECT_EXPERIMENT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
