"""CPU-only weight/schema/parameter inspection; never a training acceptance."""
import json
from pathlib import Path
import torch
from gvf.frontend import load_frontend
from gvf.model import Perception

torch.set_num_threads(4)
model = load_frontend('weights/model.pt', backend='masked_sdpa_experimental')
counts = {}
for mode in ('full', 'current', 'history', 'memory'):
    network = Perception(model, mode)
    counts[mode] = {'total': sum(p.numel() for p in network.parameters()),
                    'trainable': sum(p.numel() for p in network.parameters() if p.requires_grad)}
report = {'torch': torch.__version__, 'cuda_build': torch.version.cuda,
          'device': str(next(model.parameters()).device), 'strict_load': True,
          'encoder_channels': model.encoder.dim_features,
          'encoder_blocks': len(model.encoder.backbone.blocks),
          'parameters': counts,
          'scope': 'CPU deserialization and construction only; no forward or training'}
Path('evidence/runtime.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
