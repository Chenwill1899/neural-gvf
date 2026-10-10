"""One CPU check for evidence boundaries, geometry transforms and unknown loss.

Analytic fixtures validate label math only; not a simulated-data training result.
"""
import numpy as np
import torch
from convert import sample_queries, transform, current_surface_visible
from gvf.model import query_loss


def main():
    depth = np.full((8,8), 2., np.float32)
    valid = np.ones_like(depth, bool)
    valid[0] = False
    depth[0] = np.nan
    K = np.array([[8,0,3.5],[0,8,3.5],[0,0,1.]])
    xyz, occ, known, visible, category, uv, surface = sample_queries(depth, valid, K, np.random.default_rng(7), 16, .03)
    assert np.isfinite(xyz).all() and (uv[:,1] > 0).all()
    assert (xyz[:16,2] < 2.).all() and (xyz[32:,2] > 2.).all()
    assert not known[32:].any() and not visible[32:].any()
    assert (occ[16:32] == 1).all()
    pose = np.eye(4)
    pose[:3,3] = [1,2,3]
    assert np.allclose(transform(transform(surface, pose), np.linalg.inv(pose)), surface)
    assert current_surface_visible(surface, depth, valid, K, .03).all()
    hidden = surface.copy()
    hidden *= 2
    assert not current_surface_visible(hidden, depth, valid, K, .03).any()
    logits = torch.zeros(1,48,2, requires_grad=True)
    query_loss(logits, torch.tensor(occ[None]).float(), torch.tensor(known[None]),
               torch.tensor(visible[None]), torch.ones(1,48,dtype=torch.bool)).backward()
    assert (logits.grad[0,32:,0] == 0).all(), 'unknown became an occupancy label'
    assert (logits.grad[0,32:,1] != 0).all(), 'current-evidence loss missing'
    assert torch.isfinite(logits.grad).all()
    print('PASS: ray bounds, invalid exclusion, pose transform, visibility, unknown gradient isolation')


if __name__ == '__main__':
    main()
