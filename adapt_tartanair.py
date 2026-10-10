"""Read official V2 front-camera ZIPs; export bounded, trajectory-held-out clips.

No third-party downloader or GPU. Pose stays in a separate supervision file.
See docs/tartanair_adapter.md for authoritative sources and pixel convention.
"""
import argparse
import hashlib
import io
import json
import re
import time
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from convert import sample_queries

# Array indices are integer pixel centres. Official reader uses u+0.5-W/2.
K = np.array([[320., 0, 319.5], [0, 320., 319.5], [0, 0, 1]], np.float32)
NED_FROM_OPTICAL = np.array([[0., 0, 1], [1, 0, 0], [0, 1, 0]])
ENU_FROM_NED = np.array([[0., 1, 0], [1, 0, 0], [0, 0, -1]])
SPLITS = {f'P{i:03}': 'validation' if i == 5 else 'train' for i in range(6)}
PREFIX = 'AbandonedFactory2/Data_easy/'


def depth_png(payload):
    with Image.open(io.BytesIO(payload)) as image:
        if image.mode != 'RGBA' or image.size != (640, 640):
            raise ValueError('expected raw 640x640 four-channel depth PNG')
        rgba = np.asarray(image)
    # cv2.IMREAD_UNCHANGED yields BGRA, whereas Pillow yields RGBA.
    return np.ascontiguousarray(rgba[..., [2, 1, 0, 3]]).view('<f4')[..., 0]


def pose_matrix(pose):
    """t_xyz metres, quaternion xyzw; return ENU-world <- optical-camera."""
    q = pose[3:]
    norm = np.linalg.norm(q)
    if not np.isfinite(pose).all() or abs(norm-1) > 1e-3:
        raise ValueError('invalid pose/quaternion')
    x, y, z, w = q/norm
    r = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                  [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                  [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    out = np.eye(4)
    out[:3, :3] = ENU_FROM_NED @ r @ NED_FROM_OPTICAL
    out[:3, 3] = ENU_FROM_NED @ pose[:3]
    return out


def index(archive, modality):
    pattern = re.compile(re.escape(PREFIX)+r'(P\d{3})/'+modality+r'_lcam_front/(\d{6})_lcam_front'+('_depth' if modality == 'depth' else '')+r'\.png')
    result = {}
    for name in archive.namelist():
        match = pattern.fullmatch(name)
        if match:
            trajectory, frame = match.group(1), int(match.group(2))
            frames = result.setdefault(trajectory, {})
            if frame in frames:
                raise ValueError('duplicate frame')
            frames[frame] = name
    return result


def inventory(rgb_zip, depth_zip):
    rgb, depth = index(rgb_zip, 'image'), index(depth_zip, 'depth')
    if set(rgb) != set(depth) or set(rgb) != set(SPLITS):
        raise ValueError('unexpected/mismatched trajectory inventory')
    records, poses = {}, {}
    for traj in sorted(rgb):
        ids = sorted(rgb[traj])
        if ids != sorted(depth[traj]) or ids != list(range(len(ids))):
            raise ValueError('missing/mismatched/nonconsecutive frame IDs')
        name = PREFIX+traj+'/pose_lcam_front.txt'
        poses[traj] = np.loadtxt(io.BytesIO(depth_zip.read(name)))
        if poses[traj].shape != (len(ids), 7):
            raise ValueError('pose/image count mismatch')
        if name in rgb_zip.namelist() and rgb_zip.read(name) != depth_zip.read(name):
            raise ValueError('RGB and depth pose files differ')
        for p in poses[traj]:
            pose_matrix(p)
        records[traj] = {'frames': len(ids), 'pose_rows': len(poses[traj]), 'split': SPLITS[traj]}
    return records, rgb, depth, poses


def export(rgb_zip, depth_zip, rgb_index, depth_index, poses, traj, start, frames, out):
    ids = list(range(start, start+frames))
    if not set(ids) <= set(rgb_index[traj]):
        raise ValueError('requested clip exceeds trajectory')
    fields = {k: [] for k in ('rgb','depth','valid','K','query_xyz','occupancy','known','visible','category')}
    rng, stats = np.random.default_rng(7), []
    for frame in ids:
        with Image.open(io.BytesIO(rgb_zip.read(rgb_index[traj][frame]))) as image:
            rgb = np.asarray(image.convert('RGB'))
        depth = depth_png(depth_zip.read(depth_index[traj][frame]))
        if rgb.shape != (640, 640, 3):
            raise ValueError('RGB shape mismatch')
        valid = np.isfinite(depth) & (depth >= .05) & (depth <= 50)
        xyz, occ, known, visible, category, _, _ = sample_queries(depth, valid, K, rng, 64, .03)
        values = (rgb, np.where(valid, depth, 0).astype(np.float32), valid, K,
                  xyz.astype(np.float32), occ.astype(np.float32), known, visible, category)
        for key, value in zip(fields, values):
            fields[key].append(value)
        stats.append({'frame_id': frame, 'valid_fraction': float(valid.mean()),
                      'depth_valid_quantiles_m': np.quantile(depth[valid], [0,.5,.95,1]).tolist(),
                      'invalid_pixels': int((~valid).sum())})
    data = {key: np.stack(values) for key, values in fields.items()}
    data.update(timestamp=np.arange(frames, dtype=np.float64)/10,
                query_valid=np.ones_like(data['known']), episode_id=np.array('tartanair_v2_'+traj),
                scene_id=np.array('tartanair_v2_AbandonedFactory2'), split=np.array(SPLITS[traj]))
    stem = out/f'{traj}_{start:06}_{frames:03}'
    np.savez_compressed(stem.with_suffix('.npz'), **data)
    np.savez_compressed(stem.with_suffix('.supervision.npz'), frame_id=ids,
                        timestamp_trajectory_s=np.array(ids)/10,
                        T_world_enu_from_camera_optical=np.stack([pose_matrix(poses[traj][i]) for i in ids]),
                        raw_pose_ned_xyzw=poses[traj][ids])
    report = {'path': str(stem.with_suffix('.npz')), 'trajectory': traj, 'split': SPLITS[traj],
              'start': start, 'frames': frames, 'stats': stats, 'history_supervision': False,
              'surface_band_m': .03, 'rays_per_frame': 64,
              'note': 'Same-environment trajectory holdout, not scene holdout. No object motion labels.'}
    stem.with_suffix('.json').write_text(json.dumps(report, indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--clips', nargs='+', default=['P000:0:8', 'P005:0:8'], help='trajectory:start:length')
    p.add_argument('--verify-archives', action='store_true')
    a = p.parse_args()
    specs = [(t, int(s), int(n)) for t, s, n in (c.split(':') for c in a.clips)]
    if any(t not in SPLITS or s < 0 or n < 2 or n > 64 for t,s,n in specs):
        p.error('known trajectory, start>=0 and 2..64 frames per clip required')
    a.output.mkdir(parents=True, exist_ok=True)
    started, integrity = time.perf_counter(), {}
    with zipfile.ZipFile(a.source/'image_lcam_front.zip') as rgb_zip, zipfile.ZipFile(a.source/'depth_lcam_front.zip') as depth_zip:
        for filename, archive in [('image_lcam_front.zip', rgb_zip), ('depth_lcam_front.zip', depth_zip)]:
            if a.verify_archives:
                bad = archive.testzip()
                if bad is not None:
                    raise ValueError('ZIP CRC failure: '+bad)
                with (a.source/filename).open('rb') as f:
                    digest = hashlib.file_digest(f, 'sha256').hexdigest()
                integrity[filename] = {'crc': 'pass', 'sha256': digest, 'bytes': (a.source/filename).stat().st_size}
        records, rgb, depth, poses = inventory(rgb_zip, depth_zip)
        reports = [export(rgb_zip, depth_zip, rgb, depth, poses, *spec, a.output) for spec in specs]
    result = {'trajectories': records, 'total_paired_frames': sum(r['frames'] for r in records.values()),
              'integrity': integrity, 'clips': reports, 'seconds': time.perf_counter()-started,
              'depth': 'float32 little-endian optical-axis metres, BGRA bytes; invalid zeroed with separate mask',
              'K_integer_pixel_centres': K.tolist(), 'pose_input_to_network': False}
    (a.output/'inventory.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k != 'clips'}, indent=2))


if __name__ == '__main__':
    main()
