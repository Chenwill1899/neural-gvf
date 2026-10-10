"""Import this only after third-party execution approval. Local weights only."""
import os
import sys
import hashlib
from types import MethodType
from pathlib import Path
import torch


def load_frontend(checkpoint, train_last_blocks=0, backend='official'):
    digest = hashlib.sha256()
    with open(checkpoint, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != 'b60cf27ddbd0e51e9b59b03475c0d39d02d2e48ecf8dbb5866f04d46802b3c23':
        raise ValueError('checkpoint SHA256 differs from pinned official v0.5 weight')
    if not 0 <= train_last_blocks <= 24:
        raise ValueError('train_last_blocks must be 0..24')
    if backend not in ('official', 'masked_sdpa_experimental', 'dense_sdpa_experimental'):
        raise ValueError(backend)
    if backend != 'official':
        os.environ['XFORMERS_DISABLED'] = '1'
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'vendor/lingbot-depth'))
    from mdm.model.v2 import MDMModel
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = MDMModel(**payload['model_config'])
    model.load_state_dict(payload['model'], strict=True)
    if backend != 'official':
        model.enable_pytorch_native_sdpa()
    if backend == 'masked_sdpa_experimental':
        from mdm.model.dinov2_rgbd.layers.block import Block
        def independent_forward(block, x):
            # Keep the original selected depth tokens and their positions.
            # Per-sample tensor attention replaces only nested-list batching.
            # Eval parity versus official xformers remains unverified.
            return [Block.forward(block, sample) for sample in x] if isinstance(x, list) else Block.forward(block, x)
        for block in model.encoder.backbone.blocks:
            block.forward = MethodType(independent_forward, block)
    model.requires_grad_(False)
    if train_last_blocks:
        for block in model.encoder.backbone.blocks[-train_last_blocks:]:
            block.requires_grad_(True)
        model.encoder.output_projections.requires_grad_(True)
        # Upstream class-level checkpoint wrapper would bypass our instance forward.
        # Last-block tuning stays differentiable without checkpointing in this mode.
        if backend != 'masked_sdpa_experimental':
            model.encoder.enable_gradient_checkpointing()
    return model
