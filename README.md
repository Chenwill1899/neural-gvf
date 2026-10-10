# Neural GVF perception

Independent workspace: `/home/ubuntu5090/cwl/neural-gvf`.
Scope: RGB-D perception and temporal spatial query memory. No navigation controller,
geometric target following, RealSense integration or Orin benchmark in this phase.

Status: real-data CPU optimization completed for current-frame and recurrent-memory modes. A real-frame update with the final frontend block unfrozen passed. GPU execution is blocked by driver-level BAR1/context allocation failure before model forward. See [first results](docs/first_results.md), [curves](evidence/loss-curves.png) and [held-out readouts](evidence/heldout-readouts.png). No GPU-training or generalization success is claimed.

## Architecture and inference boundary

`gvf/frontend.py` loads local checkpoint with weights_only=True and strict=True.
It invokes official differentiable forward_feat, never infer_feat; native SDPA
backend selection is explicit. Default preserves official depth masking and xformers; masked SDPA is an unverified experimental alternative preserving token selection. Dense SDPA disables masking and is a distribution-changing ablation. Frozen
front-end optimization and a separate final-block-unfrozen run are distinct
acceptance stages. Eval mode does not disable differentiation.

At 420×560, 1200 RGB patches are read out, but attention also sees depth tokens
and CLS; 1200 is not the transformer's total token count. The original decoder
uses F+CLS and multiscale aspect-ratio UV. F, CLS, feature grid and input dimensions
are retained. No claim that compressed slots preserve all spatial information.
Native depth decoding is available via upstream forward; decoder parity still
needs runtime verification before relying on cached features for depth decoding.

Four modes: `full` keeps 1024 feature channels and all spatial tokens plus CLS;
`current` compresses channels to 384; `history` keeps three sets of 384-D tokens;
`memory` recursively updates 256 nominal spatial and 8 nominal motion slots.
Memory has four cross-attention/self-attention/1536-FFN blocks, six heads and a
learned final gate. Slot type names are hypotheses, not proven disentanglement.
Ray/depth/validity/time features augment tokens. Query xyz is in the current
camera frame; pose truth never enters forward. State resets per trajectory.
No analytic ego-motion warp is currently claimed: turning correspondence must
be learned and tested. Full-sequence backprop is used for short acceptance clips.

Output is surface-band/free classification plus current-view evidence logits.
Surface/free BCE uses only known labels; unknown locations are never negative occupancy. Simulated pose and
object motion may generate labels and metrics only. No arbitrary far-ray free
labels. Use visible, occluded, turn/revisit and dynamic groups with balanced
positive/negative known queries, held-out trajectories and per-group metrics.

## Pending simulator interface

Read-only upstream: `/home/ubuntu5090/cwl/isaac-uav`.
`data_contract_v1.md` is now available; see docs/approval_and_contract.md for compatibility gaps. Provisional NPZ adapter:

* rgb uint8 [T,H,W,3], depth float meters [T,H,W], valid bool [T,H,W].
* Depth is camera optical-axis z, not Euclidean range; camera x right,y down,z forward.
* K float [T,3,3] in pixel-center coordinates (integer pixel centers).
* timestamp [T] seconds, strictly increasing; converted to relative seconds.
* query_xyz [T,Q,3] in current camera coordinates; occupancy [T,Q] binary;
  known [T,Q] evidence mask (different from geometric ground-truth availability).
* Needed for grouped evaluation: query category [T,Q] and independent trajectory IDs.
* Ground-truth pose/object states must remain outside model input fields.

The real-data adapter is now implemented in convert.py; see docs/label_contract.md for its supported supervision and remaining geometry fields. Do not synthesize
missing simulator evidence or treat random tensors as the requested real-data test.

## Acceptance run (after approval and GPU release)

Run train.py separately with --mode full/current/history/memory and identical
train/val trajectory files and seeds. Required arguments: --train, --val,
--weights, --output. Explicitly pass --device cuda only after GPU release.
Start with short clips and frozen frontend, then --train-last-blocks 1 to prove
nonzero finite frontend gradients. Reports store initial/final loss, gradient
norms, parameter delta, independent-trajectory query predictions and accuracy.
Per-category metrics and dataset validation await final simulator contract.
Small-sample optimization success is not generalization evidence. Fair architecture
comparison needs matched training budgets and repeated seeds beyond this first run.

Latest approval scope, mask semantics and contract reconciliation: [approval_and_contract.md](docs/approval_and_contract.md). Backend execution status is tracked in evidence/.

## Current navigation interface decision

Only three-dimensional goal-point navigation is in scope; the joystick branch is paused. Keep perception/spatial memory independent of the chosen goal, reusable across goals. A later GVF module will fuse relative goal position expressed in the robot frame. TartanAir future trajectory positions may support goal relabeling, but are not optimal collision-avoidance action labels. This decision does not add a navigation controller to the current perception work.
