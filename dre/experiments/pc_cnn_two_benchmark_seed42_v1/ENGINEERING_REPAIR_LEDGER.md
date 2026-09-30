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

## Native checkpointing repair during fold5

Ictal fold3 Stage C had one silent native exit after epoch12, but its complete
epoch checkpoint was retained and a same-command retry finished that fold.
At ictal fold5 PC-CNN startup, Windows reported access violation
`0xc0000005`; Python's fault handler located it in the weakref-based
recomputation check of PyTorch's **non-reentrant** activation checkpoint
implementation, during backward. Fold5 RawCNN and folds1–4 selections remain
intact. No fold5 PC-CNN epoch was completed before this crash.

The backbone checkpoint call now uses `use_reentrant=True`. This changes only
the memory-saving recomputation implementation, not the model forward,
loss, optimizer, data, or selection protocol. An outcome-free synthetic test
compared checkpointed and uncheckpointed forward outputs and every trainable
parameter gradient in both frozen- and trainable-backbone configurations;
all matched within `atol=1e-6, rtol=1e-5` (four local topology tests passed).
The backbone BatchNorm state is frozen during PC-CNN training and the FiLM
input requires gradients, satisfying the reentrant checkpoint preconditions.
No previously selected checkpoint is regenerated or modified.

## Omni native-runtime fallback after epoch 6

The Omni RawCNN trainer saved complete epoch-boundary checkpoints through
epoch 6. Three subsequent attempts with the original PyTorch nightly
`2.12.0.dev20260327+cu128` exited natively in or near BatchNorm
(`c10_cuda.dll`, illegal instruction, and access violation); a fourth exited
without Python stderr before saving epoch 7. No official-test data was read.
The NVIDIA driver reported 616.56 and the GPU was an RTX 5090 with ample free
memory. No unrelated GPU process was terminated.

Disabling cuDNN was rejected: an official-model synthetic comparison changed
the maximum output by 0.0241 and maximum gradient by 11.86. An installed
stable PyTorch `2.11.0+cu128` environment was instead checked against the
nightly using the same pinned official CNN weights and synthetic input.
Forward outputs and BatchNorm buffers were exactly equal. Maximum gradient
drift was 0.0025654, comparable to the 0.0025597 drift obtained by replaying
the same reference in the original nightly runtime. The stable environment
loaded the epoch-6 model plus all 76 Adam optimizer slots, and a one-patient
TRAIN-only smoke step passed. This comparison used no patient outcomes.

The frozen protocol, split, model, optimizer, learning rates, seed, checkpoint
selection, and completed checkpoints were not edited. The remaining Omni
training is resumed with the stable runtime as an engineering workaround.
The PyTorch runtime change and its numerical audit are disclosed here rather
than represented as bitwise-identical training.

## Bounded native-failure supervisor for Omni PC-CNN

The stable environment completed Omni RawCNN 10/10 and selected epoch 7
using only inner validation. Omni PC-CNN Stage B completed and saved epoch 1,
then its process exited with native Windows access-violation code
`0xC0000005`; no Python exception or test access occurred. The Stage B
`last.pt` and selected checkpoint remain intact.

`resume_native_supervisor.py` launches the unmodified Omni PC trainer with
the same frozen arguments. It retries only known native Windows exit codes,
resuming from the last complete epoch and retaining separate private logs.
It stops on a Python-level error, three consecutive failures without a new
completed checkpoint, or the bounded attempt limit. It does not inspect
validation scores, change training parameters, or launch any test evaluation.

During Omni PC-CNN Stage C, the driver raised native breakpoint exception
`0x80000003` in `nvcuda64.dll` after epoch 2. Windows Application Error 1000
identified the faulting module and exception; trainer stderr was empty, and
the Stage C epoch-2 checkpoint was complete. The supervisor initially stopped
because this native code was absent from its allowlist. The allowlist now
includes `0x80000003`; bounded exact-checkpoint resume remains unchanged.
The supervisor also starts from the next unused attempt number so the earlier
private attempt logs cannot be overwritten when the supervisor itself resumes.
