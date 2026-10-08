# E3 execution amendments and engineering changes

## Authorized scientific amendment

The user approved excluding exactly the one seizure whose required preictal crop contains 1,000 padded samples per channel. All 80 patients, 7,635 channels, frozen fold roles, model windows, labels and hyperparameters remain unchanged. The amended model input contains 255 seizures. The exact private exclusion ledger is not published.

Only E3 training and validation are authorized under the source-only provenance gate. No outer held-out evaluation is performed.

## Execution changes

- Patient tensors are loaded once for each fold's FIT+VAL membership, retaining the supplied loader's exact tensor values. The access gate rejects other patients.
- Complete epochs save model, optimizer, best validation state and Python/NumPy/PyTorch/CUDA RNG states atomically. Resume rejects changes to source, data or protocol hashes.
- The amended exporter and runner use explicit UTF-8. The first amended export used the Windows default encoding; the runner stopped before training. That export was retained under a backup directory and regenerated from unchanged source caches. No completed training epoch was discarded.
- A NumPy `_multiarray_umath.cp310-win_amd64.pyd` native access violation (`0xc0000005`) ended attempt 02 after fold 1 supervised epoch 30 had been saved. Attempt 03 resumed the same code, protocol and checkpoints; fold 1 selection completed before fold 2 began. No model, threshold rule, data or seed was changed in response.
- The same native module/exception ended attempt 03 after fold 4 supervised epoch 30. Folds 1-3 had finalized and fold 4's complete training state was retained. Attempt 04 again uses the unchanged command and binding, checking existing selected weights before continuing. All attempt logs are preserved privately.
- Attempt 04 completed all fold 5 training before another native metrics crash. Attempt 05 enabled `PYTHONFAULTHANDLER` and located the failure in NumPy reductions called by sklearn F1 threshold-grid evaluation. A model-free synthetic test also crashed in NumPy-backed AUROC calculation. Thus a model-free reproduction exists; the root hardware/library cause has not been established.
- Disabling NumPy AVX2/FMA3 dispatch did **not** resolve the synthetic crash and was rejected. `resume_numpy_safe.cmd` is a rejected diagnostic, not the adopted execution route.
- A new isolated `metrics_numpy126_env` overlays NumPy 1.26.4 on the original environment's unchanged PyTorch 2.8.0+cu128, sklearn 1.7.2 and SciPy 1.15.3. The original environment was not modified. All 125 SSL and 150 supervised epochs had already completed before this change; the isolated environment only resumes finalization.
- The unstable server NumPy 2.2.6 could not finish the model-free stress reference, so that reference was produced locally with the same NumPy 2.2.6 / sklearn 1.7.2 / SciPy 1.15.3 versions. The isolated server matched all 300 metric points, 12 full threshold selections and synthetic random sampling exactly. A real fold-5 validation input had identical bytes and frozen-checkpoint GPU logits across the original and isolated environments (maximum difference 0). No prediction or threshold-selection rule was changed.

## Numerical checks

The server's 13 package/amendment tests pass. Synthetic CPU checks compare the original supplied training loops with interrupted/resumed execution: cached inputs, SSL parameters, supervised parameters, selected epoch, threshold and metrics match exactly. Changed bindings and held-out patient access are rejected. These tests do not establish scientific performance or bitwise reproducibility of GPU training.

Private logs, patient records, exports and checkpoints remain on the server.
