# Execution audit

2026-10-25: fixed additional45-49 batch in response to the request for more training seeds.
Historical42-44 test outcomes were already viewed. No result-directed seed search,
optional stopping, seed exclusion or new main-model selection is permitted.
Old sources/results remain immutable; new formal training has NOT started.
Record actual source push, server admission, frozen artifacts, production,
seals, independent verification and release below as they occur.

## Actual pre-fit admission01 PASS

Source005db7f983fd3e298770fd0297256e7b002ad836 was pushed and git ls-remote
verified before server admission. Its actual committed-tree archive passed6CPU
tests,21 exact parent-function AST checks and unchanged TimeConv engine; all13
Python/JSON/PowerShell source files, including the local release checker, matched
the archive byte-for-byte. Archive SHA256:
190935c59e09285799d8d1dcaced839e9b50c0e6634a2137798c77e82cb32582.

The native pinned OpenSSH/SCP transport retained the existing Ed25519 identity,
strict host checking and in-memory credential helper. No authentication or
cryptographic checks were weakened. The new helper changes only the explicit
anonymous admission/final download allowlist. Existing old sources were not edited.

All5 new seeds45-49 actually passed real-server admission on2026-10-25,
16:56-17:03 Asia/Shanghai:6 CPU tests each,80 matched new initializers,
20 complete actual FIT-patient backward paths,all80patients and1471965valid
windows perseed, FIT-only preprocessing invariance, branch isolation and SVM
patient-fold calibration isolation. TimeConv direct/cache logits,loss,BN and
CPU/CUDA RNG were exact; max gradient difference9.313225746154785e-10 for
every new seed, within the unchanged original tolerances. Probe states discarded,
not formal fitted epochs. No new test inference or formal training occurred.

All60 historical42-44 cells passed strict source/dependency/roles/FIT-preprocessor/
checkpoint/validation/threshold/sealed-score hash gates. All48 historical neural
initializers were independently reconstructed and matched. Historical predictions
were read for binding checks, not recomputed or changed; their outcomes were
already viewed and remain disclosed. All12 new scientific sources remain frozen.

Authenticated anonymous admission archive32876bytes,21 payloads plus manifest,
SHA256 fef7bb0651da61a91d1d5835ccbc65cc6bc8fa963ab6ee4adecde2bf18e853e0.
Local source/hash/schema/allowlist checks and all6CPU tests passed BEFORE explicit
allowlist extraction. No patient IDs, predictions, signals, weights or logs exported.

## Actual production01 start

Real admission-evidence commit33fcd109ac40e0dd55f9cf8df9eaeeba624bc83c
was pushed and git ls-remote verified BEFORE formal fitting. The actual committed
tree again passed all6CPU tests,21 unchanged parent AST functions,12 scientific
source hashes and all21 admission payload hashes. Prefit release archive SHA:
067989f2f6ad23656d8acde77237c47ab51f79caffe5cd6ff3dc4c86f26aa8ab.

Production01 actually started2026-10-25T17:06:15+08:00. The sustained foreground
native --stream SSH owner is execsession76798, coordinator17356(parent19572),
initial Python worker6224/venvchild40804. Exact frozen command:
run_all.ps1 -RunTag production01 -SourceCommit
33fcd109ac40e0dd55f9cf8df9eaeeba624bc83c -StopAfter All.

At17:06:48+08,seed45/HUP-held-out Raw+Feature was training epoch8,with epochs
about3.3seconds each, stderr empty and atomic LAST checkpoint advancing. This
head timing is NOT an ETA for TimeConv or the whole batch. No new test inference
or performance metrics have been inspected. Keep the owner alive; never duplicate
active workers or interpret a transport reset alone as a trainer failure.

All12 scientific source files remain identical to real admission. Completion
still requires100 fresh main fits, all160 choices frozen before new test,all160
scores sealed before metrics, independent3200row/28800endpoint/405nominal/
8family verification and final anonymous release with GitHub remote-tip check.
