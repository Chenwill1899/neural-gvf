"""Decisive sequence optimization; NPZ adapter pending simulator contract.

No execution until third-party approval, data availability and GPU release.
"""
import argparse
import json
import time
import hashlib
from pathlib import Path
import numpy as np
import torch
from gvf.frontend import load_frontend
from gvf.model import Perception, query_loss


def load(path, device):
    required = ('rgb', 'depth', 'valid', 'K', 'timestamp', 'query_xyz', 'occupancy', 'known', 'visible', 'query_valid', 'category')
    with np.load(path, allow_pickle=False) as data:
        seq = {k: torch.as_tensor(data[k].copy(), device=device) for k in required}
        seq['episode_id'], seq['scene_id'] = str(data['episode_id']), str(data['scene_id'])
        seq['split'] = str(data['split'])
    if seq['rgb'].dtype != torch.uint8 or seq['rgb'].ndim != 4:
        raise ValueError('rgb must be uint8 T,H,W,3')
    seq['rgb'] = seq['rgb'].permute(0, 3, 1, 2).float()/255
    seq['valid'] = seq['valid'].bool()
    seq['known'] = seq['known'].bool()
    seq['visible'] = seq['visible'].bool()
    seq['query_valid'] = seq['query_valid'].bool()
    for k in ('depth', 'K', 'query_xyz', 'occupancy'):
        seq[k] = seq[k].float()
    seq['timestamp'] = (seq['timestamp'].double()-seq['timestamp'][0].double()).float()
    if not torch.all(seq['timestamp'][1:] > seq['timestamp'][:-1]):
        raise ValueError('timestamps must increase strictly')
    if not seq['known'].any():
        raise ValueError('sequence has no known query supervision')
    return seq


def rollout(model, seq):
    state, losses, predictions = None, [], []
    for t in range(len(seq['rgb'])):
        elapsed = seq['timestamp'][t:t+1]
        delta = elapsed*0 if t == 0 else elapsed-seq['timestamp'][t-1:t]
        logits, state, _ = model(seq['rgb'][t:t+1], seq['depth'][t:t+1], seq['valid'][t:t+1],
                                 seq['K'][t:t+1], elapsed, delta, seq['query_xyz'][t:t+1], state,
                                 None if '_features' not in seq else seq['_features'][t])
        losses.append(query_loss(logits, seq['occupancy'][t:t+1], seq['known'][t:t+1], seq['visible'][t:t+1], seq['query_valid'][t:t+1]))
        predictions.append(logits)
    return torch.stack(losses).mean(), torch.cat(predictions)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train', type=Path, required=True)
    p.add_argument('--val', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--mode', choices=['full', 'current', 'history', 'memory'], default='memory')
    p.add_argument('--device', default='cpu')
    p.add_argument('--feature-cache-dir', type=Path)
    p.add_argument('--normalize-state', action='store_true')
    p.add_argument('--validation-scope', choices=['scene', 'trajectory'], default='scene')
    p.add_argument('--steps', type=int, default=30)
    p.add_argument('--train-last-blocks', type=int, default=0)
    p.add_argument('--backend', choices=['official', 'masked_sdpa_experimental', 'dense_sdpa_experimental'], default='official')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path)
    args = p.parse_args()
    if args.train.resolve() == args.val.resolve():
        raise ValueError('validation must be a separate trajectory')
    torch.manual_seed(7)
    torch.set_num_threads(4)
    started = time.perf_counter()
    policy = 'dense_experimental' if args.backend == 'dense_sdpa_experimental' else 'official'
    model = Perception(load_frontend(args.weights, args.train_last_blocks, args.backend), args.mode, depth_policy=policy, normalize_state=args.normalize_state).to(args.device)
    train, val = load(args.train, args.device), load(args.val, args.device)
    if train['episode_id'] == val['episode_id']:
        raise ValueError('validation requires a held-out trajectory')
    if args.validation_scope == 'scene' and train['scene_id'] == val['scene_id']:
        raise ValueError('scene holdout requires a different scene; use explicit trajectory scope for same-scene data')
    if train['split'] != 'train' or val['split'] != 'validation':
        raise ValueError('manifest train/validation split mismatch')
    if args.feature_cache_dir:
        if args.train_last_blocks:
            raise ValueError('feature caching forbidden for trainable frontend')
        model.frontend.eval()
        args.feature_cache_dir.mkdir(parents=True, exist_ok=True)
        for path, seq in ((args.train, train), (args.val, val)):
            key = hashlib.sha256(path.read_bytes() + args.backend.encode() + b'v0.5-1200-float32').hexdigest()
            cached = args.feature_cache_dir/(key+'.pt')
            if cached.exists():
                seq['_features'] = torch.load(cached, map_location=args.device, weights_only=True)
            else:
                seq['_features'] = []
                with torch.no_grad():
                    for t in range(len(seq['rgb'])):
                        d = seq['depth'][t:t+1]
                        mask = seq['valid'][t:t+1] & torch.isfinite(d) & (d>=.05) & (d<=50)
                        feat = model.frontend.forward_feat(seq['rgb'][t:t+1], 1200, torch.where(mask,d,0.), enable_depth_mask=policy=='official')
                        seq['_features'].append(tuple(x.detach() for x in feat))
                        print(f'cached {path.name} frame {t+1}/{len(seq["rgb"])}', flush=True)
                torch.save([tuple(x.cpu() for x in pair) for pair in seq['_features']], cached)
    params = {k: v for k, v in model.named_parameters() if v.requires_grad}
    before = {k: v.detach().cpu().clone() for k, v in params.items()}
    optimizer = torch.optim.AdamW(params.values(), lr=1e-4)
    report = {'device': args.device, 'feature_cache': bool(args.feature_cache_dir), 'normalize_state': args.normalize_state, 'mode': args.mode, 'backend': args.backend, 'depth_policy': policy, 'train': str(args.train), 'val': str(args.val),
              'trainable_parameters': sum(v.numel() for v in params.values()), 'steps': [],
              'validation_scope': args.validation_scope, 'same_scene': train['scene_id'] == val['scene_id'],
              'scope': 'optimization evidence only; not generalization or Orin performance'}
    model.eval()
    with torch.no_grad():
        report['initial_train_loss'] = rollout(model, train)[0].item()
    for step in range(args.steps):
        model.train()
        model.frontend.eval()  # deterministic backbone; eval does not disable autograd
        optimizer.zero_grad(set_to_none=True)
        loss, _ = rollout(model, train)
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite loss')
        loss.backward()
        grads = {k: v.grad for k, v in params.items() if v.grad is not None}
        if not grads or not all(torch.isfinite(g).all() for g in grads.values()):
            raise RuntimeError('missing/nonfinite gradients')
        norm = sum(g.detach().float().square().sum().item() for g in grads.values())**.5
        if norm == 0:
            raise RuntimeError('zero gradient')
        frontend_norm = sum(g.detach().float().square().sum().item() for k,g in grads.items() if k.startswith('frontend.'))**.5
        if args.train_last_blocks and frontend_norm == 0:
            raise RuntimeError('differentiable frontend check failed')
        torch.nn.utils.clip_grad_norm_(params.values(), 1.)
        optimizer.step()
        report['steps'].append({'loss': loss.item(), 'grad_norm': norm, 'frontend_grad_norm': frontend_norm})
        print(json.dumps({'step': step, **report['steps'][-1]}), flush=True)
    model.eval()
    with torch.no_grad():
        report['final_train_loss'] = rollout(model, train)[0].item()
        val_loss, logits = rollout(model, val)
        report['val_loss'] = val_loss.item()
        mask = val['known']
        report['val_known_occupancy_accuracy'] = ((logits[..., 0] > 0)[mask] == val['occupancy'][mask].bool()).float().mean().item()
        report['val_by_category'] = {}
        for category in val['category'].unique().tolist():
            group = (val['category'] == category) & val['query_valid']
            labeled = group & mask
            report['val_by_category'][str(category)] = {
                'queries': group.sum().item(), 'known': labeled.sum().item(),
                'surface_free_accuracy': ((logits[...,0]>0)[labeled] == val['occupancy'][labeled].bool()).float().mean().item() if labeled.any() else None,
                'current_evidence_accuracy': ((logits[...,1]>0)[group] == val['visible'][group]).float().mean().item() if group.any() else None}
    report['elapsed_seconds'] = time.perf_counter()-started
    report['train_frames'], report['val_frames'] = len(train['rgb']), len(val['rgb'])
    report['gpu_peak_allocated_bytes'] = torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else 0
    report['parameter_delta_l2'] = sum((v.detach().cpu()-before[k]).float().square().sum().item() for k,v in params.items())**.5
    report['optimization_pass'] = report['parameter_delta_l2'] > 0 and report['final_train_loss'] < report['initial_train_loss']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    np.savez_compressed(args.output.with_suffix('.predictions.npz'), logits=logits.cpu().numpy(),
                        query_xyz=val['query_xyz'].cpu().numpy(), known=mask.cpu().numpy(), occupancy=val['occupancy'].cpu().numpy(),
                        category=val['category'].cpu().numpy(), query_valid=val['query_valid'].cpu().numpy())
    if args.checkpoint:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({k:p.detach().cpu() for k,p in params.items()}, args.checkpoint)
    if not report['optimization_pass']:
        raise RuntimeError('optimization acceptance failed; see report')


if __name__ == '__main__':
    main()
