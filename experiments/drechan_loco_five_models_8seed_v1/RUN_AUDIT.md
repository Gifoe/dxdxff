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

## Actual all160 completion, score seal, independent verification and export

Production02's exact resume started2026-10-26T02:26:23+08:00. The final
seed49/pediatric TimeConv cell completed15 epochs, selecting epoch9 by the
unchanged validation rule. All100 fresh main fits completed, including80 neural
cells and20 final SVMs, plus60 FIT-only OOF auxiliary fits and20 calibrators.
Fresh complete neural epochs total1026; all historical/fresh records total1651.
The partial unfinished epoch replay is not an additional completed budget epoch.

Server public phase records were written in order (Asia/Shanghai):

- 02:50:18: ALL160_FROZEN_BEFORE_NEW_TEST,160 choices.
- 03:01:45: ALL160_SEALED_BEFORE_METRICS,160 score cells.
- 03:02:18: ALL5_MODELS_8SEED_REPORTED,3200 patient-seed-model rows.
- 03:02:54: COMPLETE_VERIFIED_LOCO_5_MODELS_4_SOURCES_8SEED.
- 03:03:15: final anonymous package created; foreground owner subsequently exit0.

All current production02 stage stderr files were empty. The independent verifier
reconstructed all3200 patient rows/28800 endpoint values and405 nominal plus8
main-family intervals. It verified160 cells,60 strict historical reuses and100
fresh fits; maximum aggregate numeric error1.1102230246251565e-16, tolerance
unchanged1e-12. Original80patients,7635channels,256seizures and1471965 valid
windows remained intact; pure-feature arms intentionally read no Raw windows.

All160-choice freeze SHA256:
378be5240a32854a22cac8db30b4b9a6649548f5f70e342f4ee8af63f8334a9a.
All160-score seal SHA256:
a3b70e784e2082f22492537eb10be961f26ad6208472cf5efdb8eb587e55e4ec.

Only the explicit anonymous final archive was fetched through the unchanged
native pinned transport. Whole SHA256 matched both server package output and
local transfer check:
c879e83e667f4e1f02c6f45218667587000bc004229b00cb1f605ef1696e7aa1,
112318bytes,42 anonymous payloads plus manifest. Before any extraction, local
release_check.py --package passed all6 CPU tests, exact allowlist/schema/counts,
all payload hashes, all12 scientific source hashes and independent PASS gates.
Only then were those43 entries extracted into a new public directory; all42
payload hashes were rechecked. No private runtime artifact was downloaded.

The release-only verifier also checks63 existing/final admission/payload hashes,
unchanged12 source hashes, byte-identical copied admission proofs, all60 old
cell bindings,75 old aggregate rows and all40 seed/arm patient-channel totals.
All24 Raw-reading seed/arm groups retain1471965 valid windows; the16 feature-only
groups correctly report zero Raw consumption. The old42-44 sources/results are
not edited or recomputed. The additional release-only check treats identical
historical SVM parameter-count strings nan as unchanged, not a numeric difference.

README and FINDINGS disclose all42-49 outcomes, including all9 metrics and both
per-source F1 tables. Full leads overall Macro-F1 but not every source or AUROC;
Feature-only/SVM primary difference intervals include zero. No seed or source
was excluded and no favorable stopping/threshold tuning was introduced.

Release commands (also used from a real staged/committed Git-tree archive):

```powershell
python experiments/drechan_loco_five_models_8seed_v1/release_check.py --package <authenticated-public-archive.zip>
python tools/drechan_loco_3seed_release_tests_v1/verify_parent_math.py
python tools/drechan_loco_8seed_release_tests_v1/verify_public.py
```

The archived tree must include prior3seed common/pipeline/svm/timeconv_engine,
original seed42 parent functions and original m8_raw_backbones_pr_seed42_v1/
evaluate.py, with no worktree-only import fallback. The archive payload hashes
are checked again so Git newline normalization cannot silently change public or
nested admission bytes. Only this experiment's release/doc files are staged;
the unrelated untracked patient-context experiment remains untouched.

## Actual Git-index release gate PASS

An actual git write-tree snapshot88f0a34121acee2cf1dc84abc1d4c21af1f16b70
was exported with git archive and extracted into a separate new directory.
That archive included all explicitly required original/3seed dependencies,
not worktree-only fallback files. From the archive, all6 CPU tests passed,
the21 original-parent AST comparisons and exact TimeConv engine passed, the
authenticated final package/source/count verifier passed, and the additional
public verifier passed all42 final/21 admission hashes,12 frozen sources,
60 old bindings,75 old aggregate rows and40 seed/arm coverage groups. A Git
diff against pre-fit33fcd confirmed original42/3seed sources/results and the
old admission files unchanged. Staging contains only the explicit47 release
and documentation files. A final committed-tree archive will repeat these
same gates before push; remote ls-remote must match that final commit.
