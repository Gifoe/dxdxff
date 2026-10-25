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

## Production01 host-memory failure and exact production02 resume

Production01 exited1 at2026-10-26T02:14:21+08:00 during the final fresh
main fit: seed49, pediatric held out, original TimeConvCNN. At that point,
99/100 fresh main selections were complete; seeds45-48 each had20 selections
and were internally frozen, and seed49 had19. No global all160-choice freeze
or new test inference had occurred.

The actual exception was MemoryError in the original RawStore SHA streaming
reader while allocating an8MiB read block. It was not a hash mismatch, a
scientific input-gate failure or a CUDA out-of-memory exception. Windows System
Event2004 at02:14:15 reported low virtual memory and named the own training
processes. This establishes host virtual-memory exhaustion; it does not identify
a specific leak or justify changing the frozen scientific implementation.

Read-only inspection confirmed the old owner, coordinator and workers exited.
The latest atomic LAST checkpoint was epoch6: finite model tensors,68 Adam
states, complete Python/NumPy/CPU/CUDA RNG state, history6, best epoch5 and
stale1. Its input binding, all12 scientific source hashes and58 parent dependency
hashes matched. Its181637763bytes had SHA256:
9768819e77b6b6615045865f4f1abbb0448f6403db8d95e3e188b0bf8a1939a1.
Byte-verified checkpoint, configuration/status and old log copies, together with
hashes of all99 completed main-cell records, were retained only on the trusted
server in a separate engineering-private failure snapshot. No patient records,
model states or runtime logs were exported or committed.

After confirming the coordinator lock was free, exactly one foreground resume
was launched with a new tag and the identical pre-fit SourceCommit:
run_all.ps1 -RunTag production02 -SourceCommit
33fcd109ac40e0dd55f9cf8df9eaeeba624bc83c -StopAfter All.
Foreground native owner execsession79030, coordinator27384(parent29176),
Python38688/venvchild27392. The99 completed selections were retained. The final
fit restored epoch6 model/Adam/RNG, replaying its unfinished epoch7 under the
unchanged patient order, precision, loss, optimizer, validation rules and budget.
No completed epoch was discarded. No cache, hash-reader, scientific source,
data, feature, OS/pagefile setting or unrelated job was changed.

At2026-10-26T02:31:31+08:00, resumed epochs7 and8 had completed, the atomic
LAST checkpoint advanced at02:31:06, and epoch9 had reached10/56 training
patients with current stderr empty. This is recovery evidence, not completion
or performance evidence. Keep production02's owner alive and monitor its own
logs; production01's preserved error is historical. All original freeze, seal,
independent-verification and anonymous-release gates still apply.
