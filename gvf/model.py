"""RGB-D-only recurrent query perception. No ground-truth pose input."""
import torch
from torch import nn
from torch.nn import functional as F


class Update(nn.Module):
    def __init__(self, width=384, heads=6):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(3)])
        self.cross = nn.MultiheadAttention(width, heads, batch_first=True)
        self.self_attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(width, 4*width), nn.GELU(), nn.Linear(4*width, width))

    def forward(self, slots, tokens):
        q = self.norms[0](slots)
        slots = slots + self.cross(q, tokens, tokens, need_weights=False)[0]
        q = self.norms[1](slots)
        slots = slots + self.self_attn(q, q, q, need_weights=False)[0]
        return slots + self.ffn(self.norms[2](slots))


class Perception(nn.Module):
    """Modes isolate full features, channel compression, short history, recurrence.

    Queries are xyz in the CURRENT camera frame, in meters. Slots have no
    guaranteed geometric meaning; camera motion must be learned from RGB-D.
    """
    def __init__(self, frontend, mode='memory', width=384, history=3, depth_policy='official', normalize_state=False):
        super().__init__()
        if mode not in ('full', 'current', 'history', 'memory'):
            raise ValueError(mode)
        self.frontend, self.mode, self.history = frontend, mode, history
        self.normalize_state = normalize_state
        if depth_policy not in ('official', 'dense_experimental'):
            raise ValueError(depth_policy)
        self.depth_policy = depth_policy
        width = 1024 if mode == 'full' else width
        heads = 16 if mode == 'full' else 6
        self.project = nn.Identity() if mode == 'full' else nn.Linear(1024, width)
        # ray xyz, z depth, validity fraction, elapsed seconds, frame delta.
        self.geometry = nn.Sequential(nn.Linear(7, width), nn.GELU(), nn.Linear(width, width))
        self.cls_type = nn.Parameter(torch.zeros(1, 1, width))
        self.query = nn.Sequential(nn.Linear(3, width), nn.GELU(), nn.Linear(width, width))
        self.read = nn.MultiheadAttention(width, heads, batch_first=True)
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2))
        if mode == 'memory':
            self.initial = nn.Parameter(torch.randn(1, 264, width)*0.02)
            self.slot_type = nn.Embedding(2, width)
            self.blocks = nn.ModuleList([Update(width, heads) for _ in range(4)])
            self.gate = nn.Linear(2*width, width)

    @staticmethod
    def geometry_inputs(depth, observed, K, rows, cols):
        b, h, w = depth.shape
        y, x = torch.meshgrid(torch.arange(rows, device=depth.device),
                              torch.arange(cols, device=depth.device), indexing='ij')
        pixels = torch.stack(((x+.5)*w/cols-.5, (y+.5)*h/rows-.5, torch.ones_like(x)), -1)
        rays = torch.linalg.solve(K.float(), pixels.reshape(-1, 3).T.expand(b, -1, -1)).transpose(1, 2)
        mask = observed.float()[:, None]
        fraction = F.adaptive_avg_pool2d(mask, (rows, cols))
        z = F.adaptive_avg_pool2d(depth[:, None], (rows, cols))/fraction.clamp_min(1e-6)
        return torch.cat((rays, z.flatten(2).transpose(1, 2), fraction.flatten(2).transpose(1, 2)), -1), fraction

    def cached_tokens(self, feat, cls, geometry, elapsed, delta):
        times = torch.stack((elapsed, delta), -1)[:, None].expand(-1, geometry.shape[1], -1)
        tokens = self.project(feat.flatten(2).transpose(1, 2)) + self.geometry(torch.cat((geometry, times), -1))
        return torch.cat((tokens, self.project(cls)[:, None] + self.cls_type), 1)

    def encode(self, rgb, depth, valid, K, elapsed, delta, native_features=None):
        observed = valid & torch.isfinite(depth) & (depth >= 0.05) & (depth <= 50.)
        depth = torch.where(observed, depth, 0.)
        if native_features is None:
            feat, cls = self.frontend.forward_feat(rgb, num_tokens=1200, depth=depth,
                                                  enable_depth_mask=self.depth_policy == 'official')
        else:
            feat, cls = native_features
        rows, cols = feat.shape[-2:]
        h, w = rgb.shape[-2:]
        geometry, fraction = self.geometry_inputs(depth, observed, K, rows, cols)
        tokens = self.cached_tokens(feat, cls, geometry, elapsed, delta)
        # Preserve native feature map, CLS, grid and image shape for decoder audits.
        return tokens, {'F': feat, 'CLS': cls, 'grid': (rows, cols), 'image_hw': (h, w),
                        'observed_depth': observed, 'valid_fraction': fraction,
                        'depth_policy': self.depth_policy}

    def forward(self, rgb, depth, valid, K, elapsed, delta, queries, state=None, native_features=None):
        tokens, native = self.encode(rgb, depth, valid, K, elapsed, delta, native_features)
        logits, state = self.read_tokens(tokens, queries, state)
        return logits, state, native

    def forward_cached(self, feat, cls, geometry, elapsed, delta, queries, state=None):
        return self.read_tokens(self.cached_tokens(feat, cls, geometry, elapsed, delta), queries, state)

    def read_tokens(self, tokens, queries, state=None, return_features=False):
        if self.mode == 'memory':
            types = torch.cat((torch.zeros(256, device=tokens.device, dtype=torch.long),
                               torch.ones(8, device=tokens.device, dtype=torch.long)))
            old = (self.initial + self.slot_type(types)[None]).expand(tokens.shape[0], -1, -1) if state is None else state
            new = old
            for block in self.blocks:
                new = block(new, tokens)
            gate = self.gate(torch.cat((old, new), -1)).sigmoid()
            state = old + gate*(new-old)
            if self.normalize_state:
                state = F.layer_norm(state, (state.shape[-1],))
            context = state
        elif self.mode == 'history':
            state = (([] if state is None else state) + [tokens])[-self.history:]
            context = torch.cat(state, 1)
        else:
            context, state = tokens, None
        q = self.query(queries)
        features = q + self.read(q, context, context, need_weights=False)[0]
        return (features if return_features else self.output(features)), state


class CurrentWithMemory(nn.Module):
    """Frozen current readout plus a query-wise geometric-logit residual.

    The evidence channel remains exactly current-only. Memory slots are still
    latent, not geometrically registered; this tests the current-path bottleneck.
    """
    def __init__(self, current, width=384):
        super().__init__()
        if current.mode != 'current':
            raise ValueError('A trained current-frame baseline is required')
        self.current = current.requires_grad_(False).eval()
        self.memory = Perception(None, 'memory', width=width, normalize_state=True)
        self.memory.output = nn.Identity()  # Read features, not its old two-logit head.
        self.residual_gate = nn.Linear(width + 2, 1)
        self.residual = nn.Linear(width, 1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)
        self.use_memory = True

    def train(self, mode=True):
        super().train(mode)
        self.current.eval()
        return self

    def forward_cached(self, feat, cls, geometry, elapsed, delta, queries, state=None):
        with torch.no_grad():
            current, _ = self.current.forward_cached(feat, cls, geometry, elapsed, delta, queries)
        if not self.use_memory:
            return current, state
        tokens = self.memory.cached_tokens(feat, cls, geometry, elapsed, delta)
        features, state = self.memory.read_tokens(tokens, queries, state, return_features=True)
        gate = self.residual_gate(torch.cat((features, current), -1)).sigmoid()
        occupancy = current[..., :1] + gate * self.residual(features)
        return torch.cat((occupancy, current[..., 1:]), -1), state

    def forward(self, rgb, depth, valid, K, elapsed, delta, queries, state=None, native_features=None):
        with torch.no_grad():
            _, native = self.current.encode(rgb, depth, valid, K, elapsed, delta, native_features)
            observed = native['observed_depth']
            geometry, _ = self.current.geometry_inputs(
                torch.where(observed, depth, 0.), observed, K, *native['grid'])
        logits, state = self.forward_cached(native['F'], native['CLS'], geometry, elapsed, delta, queries, state)
        return logits, state, native


def query_loss(logits, occupancy, known, visible, query_valid):
    """Channel 0: surface/free; channel 1: current-view evidence, not GT availability."""
    known = known.bool() & query_valid.bool()
    occ = F.binary_cross_entropy_with_logits(logits[..., 0][known], occupancy[known]) if known.any() else logits[..., 0].sum()*0
    evidence = F.binary_cross_entropy_with_logits(logits[..., 1][query_valid], visible.float()[query_valid]) if query_valid.any() else logits[..., 1].sum()*0
    return occ + evidence
