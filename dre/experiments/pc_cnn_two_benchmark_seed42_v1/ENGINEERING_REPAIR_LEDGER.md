# Resume-only engineering repair

The frozen `PROTOCOL_LOCK.json`, data splits, model, loss, optimizer, seeds,
checkpoint-selection rule, and all completed checkpoints were left unchanged.

- During ictal fold2 RawCNN initialization, the Windows Python child exited
  with access violation `0xc0000005` and no completed fold2 epoch. A single
  same-command retry passed fold2 and resumed the queue; the underlying native
  cause was not determined. No unrelated GPU process was stopped.
- During ictal fold3 PC-CNN Stage B epoch10, PyTorch raised
  `CUDNN_STATUS_EXECUTION_FAILED`. The fold3 Stage B `last.pt` from epoch9
  and its selected best checkpoint remained intact. A same-command resume
  then exposed a deterministic loader bug: `torch.load(...,
  map_location=cuda)` moved the saved **CPU** RNG ByteTensor to CUDA, while
  `torch.set_rng_state` requires it on CPU.
- The only code repair was `.detach().cpu()` before restoring `torch_rng`
  in both RawCNN and PC-CNN trainers. The already-saved CPU/CUDA RNG bytes,
  model, optimizer, and selected checkpoint are preserved. This does not
  alter the numerical training protocol; it permits the intended exact
  epoch-boundary resume. Eleven local tests passed after the change.

No outcome-driven tuning or test access occurred during this repair.
