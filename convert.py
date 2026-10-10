"""Contract-v1 real RGB-D -> evidence-limited surface/free query supervision.

GT poses are used only in this offline label builder. No third-party execution
has been performed during implementation. PNG loading requires existing Pillow.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image


def within(root, name):
    path = (root/name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('dataset path escapes root')
    return path


def transform(points, matrix):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def rays_at(uv, K):
    return np.linalg.solve(K, np.column_stack((uv, np.ones(len(uv)))).T).T


def sample_queries(depth, valid, K, rng, count, band):
    """Positive = measured surface band, NOT solid-volume interior occupancy."""
    v, u = np.where(valid & (depth > 0.05 + 4*band))
    if not len(u):
        raise ValueError('no valid rays with sufficient free interval')
    take = rng.choice(len(u), count, replace=len(u) < count)
    uv = np.column_stack((u[take], v[take]))
    z = depth[v[take], u[take]]
    rays = rays_at(uv, K)
    # Euclidean ray distance, with a tolerance band around the measured surface.
    lengths = np.linalg.norm(rays, axis=1)
    direction = rays/lengths[:, None]
    distance = z*lengths
    free = rng.uniform(0.05, distance-2*band)
    surface = distance + rng.uniform(-band/2, band/2, count)
    hidden = distance + rng.uniform(2*band, 1.0 + 2*band, count)
    xyz = np.concatenate([direction*s[:, None] for s in (free, surface, hidden)])
    occupied = np.repeat([0., 1., 0.], count)
    known = np.repeat([True, True, False], count)
    visible = known.copy()
    category = np.repeat([0, 1, 2], count).astype(np.int64)
    return xyz, occupied, known, visible, category, uv, rays*z[:, None]


def current_surface_visible(points, depth, valid, K, band):
    h, w = depth.shape
    projected = points @ K.T
    good_z = points[:, 2] > 0.05
    uv = np.rint(projected[:, :2]/np.maximum(projected[:, 2:3], 1e-8)).astype(np.int64)
    inside = good_z & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    result = np.zeros(len(points), dtype=bool)
    ids = np.flatnonzero(inside)
    u, v = uv[ids].T
    # Conservative pixel-ray comparison; a surface behind the first hit is hidden.
    result[ids] = valid[v, u] & (np.abs(depth[v, u]-points[ids, 2]) <= band)
    return result


def convert(root, frames_path, output, count=64, band=.03, max_frames=4, seed=7, static_history=False):
    records = [json.loads(line) for line in frames_path.read_text().splitlines() if line.strip()][:max_frames]
    if len(records) < 2:
        raise ValueError('need at least two real frames')
    if len({r['episode_id'] for r in records}) != 1 or len({r['scene_id'] for r in records}) != 1:
        raise ValueError('mixed episodes/scenes')
    manifest = json.loads((root/'manifest.json').read_text())
    if not manifest.get('complete', False):
        raise ValueError('dataset collection has not completed')
    matches = [e for e in manifest['episodes'] if e['episode_id'] == records[0]['episode_id']]
    if len(matches) != 1 or matches[0]['scene_id'] != records[0]['scene_id']:
        raise ValueError('episode missing/ambiguous in manifest')
    if frames_path.resolve() != within(root, matches[0]['path'])/'frames.jsonl':
        raise ValueError('frame source does not match manifest')
    ns = [int(r['timestamp_ns']) for r in records]
    if any(b <= a for a,b in zip(ns, ns[1:])) or len({r['render_frame_id'] for r in records}) != len(records):
        raise ValueError('nonmonotonic time or duplicate render frames')
    rng = np.random.default_rng(seed)
    fields = {k: [] for k in ('rgb','depth','valid','K','query_xyz','occupancy','known','visible','category')}
    history = []
    for record in records:
        inputs, truth = record['inputs'], record['supervision']
        rgb = np.asarray(Image.open(within(root, inputs['rgb'])).convert('RGB'))
        depth = np.load(within(root, inputs['depth']), allow_pickle=False)
        mask = np.asarray(Image.open(within(root, inputs['valid_depth_mask']))) > 0
        if depth.dtype != np.float32 or depth.shape != rgb.shape[:2] or mask.shape != depth.shape:
            raise ValueError('invalid depth dtype/shape or unsynchronized image sizes')
        if depth.shape != (inputs['height'], inputs['width']):
            raise ValueError('declared image size mismatch')
        valid = np.isfinite(depth) & (depth >= .05) & (depth <= 50)
        if not np.array_equal(mask, valid):
            raise ValueError('saved mask disagrees with valid depth contract')
        calibration = json.loads(within(root, inputs['calibration']).read_text())
        K = np.asarray(calibration['K'], dtype=np.float64).reshape(3,3)
        if not np.isfinite(K).all() or K[0,0] <= 0 or K[1,1] <= 0 or not np.allclose(K[2], [0,0,1]):
            raise ValueError('invalid K')
        xyz, occ, known, visible, category, uv, surfaces = sample_queries(depth, valid, K, rng, count, band)
        if static_history:
            # Explicitly certified immutable geometry only. A missing mapping is
            # a hard error; instance ID 0 or zero velocity is never assumed static.
            ids = truth.get('certified_static_instance_ids')
            if ids is None or 'instance_id' not in truth:
                raise ValueError('static history needs certified_static_instance_ids and instance_id path')
            instance = np.load(within(root, truth['instance_id']), allow_pickle=False)
            if instance.shape != depth.shape:
                raise ValueError('instance image shape mismatch')
            pose = np.asarray(truth['T_world_from_camera'], dtype=np.float64).reshape(4,4)
            if not np.isfinite(pose).all() or not np.allclose(pose[3], [0,0,0,1]) or not np.allclose(pose[:3,:3].T@pose[:3,:3], np.eye(3), atol=1e-4):
                raise ValueError('invalid supervision pose')
            if history:
                pool = np.concatenate(history)
                previous = transform(pool[rng.choice(len(pool), count, replace=len(pool)<count)], np.linalg.inv(pose))
                active = np.ones(count, dtype=bool)
                seen = current_surface_visible(previous, depth, valid, K, band)
            else:
                previous, active, seen = np.zeros((count,3)), np.zeros(count, bool), np.zeros(count, bool)
            xyz = np.concatenate((xyz, previous))
            occ = np.concatenate((occ, np.ones(count)))
            known = np.concatenate((known, active))
            visible = np.concatenate((visible, seen))
            category = np.concatenate((category, np.full(count, 3)))
            # Padding queries are excluded from BOTH heads.
            keep = np.isin(instance[uv[:,1], uv[:,0]], ids)
            if keep.any():
                history.append(transform(surfaces[keep], pose))
        values = (rgb, depth, valid, K.astype(np.float32), xyz.astype(np.float32), occ.astype(np.float32), known, visible, category)
        for key, value in zip(fields, values):
            fields[key].append(value)
    packed = {k: np.stack(v) for k,v in fields.items()}
    packed['query_valid'] = np.ones_like(packed['known'])
    if static_history:
        packed['query_valid'][:, -count:] = packed['known'][:, -count:]
    packed['timestamp'] = np.array([(n-ns[0])*1e-9 for n in ns], dtype=np.float64)
    packed['episode_id'] = np.array(records[0]['episode_id'])
    packed['scene_id'] = np.array(records[0]['scene_id'])
    packed['split'] = np.array(matches[0]['split'])
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **packed)
    report = {'source_frames': str(frames_path), 'source_sha256': hashlib.sha256(frames_path.read_bytes()).hexdigest(),
              'episode_id': records[0]['episode_id'], 'scene_id': records[0]['scene_id'], 'frames': len(records),
              'seed': seed, 'band_m': band, 'static_history': static_history,
              'categories': {'0':'visible_free','1':'visible_surface_band','2':'behind_first_surface_unknown','3':'certified_static_history_surface'},
              'semantics': 'surface-band occupancy surrogate; not volumetric occupancy',
              'dynamic_geometry_export_used': False,
              'collection_policy': matches[0].get('control_method', manifest.get('control_method', 'unspecified'))}
    output.with_suffix('.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--frames', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--count', type=int, default=64)
    p.add_argument('--band', type=float, default=.03)
    p.add_argument('--max-frames', type=int, default=4)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--static-history', action='store_true')
    a = p.parse_args()
    if a.count < 1 or a.band <= 0 or a.max_frames < 2:
        p.error('count/band must be positive; max-frames >= 2')
    convert(a.root, a.frames, a.output, a.count, a.band, a.max_frames, a.seed, a.static_history)
