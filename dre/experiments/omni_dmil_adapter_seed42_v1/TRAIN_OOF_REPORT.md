# D-MIL TRAIN OOF report

| Model | Params | AUROC | AP | Macro-F1 | patient-equal AP | MRR | Top1 | Delta AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V0_MEAN | 0 | 0.957155 | 0.829912 | 0.870792 | 0.871055 | 0.949253 | 0.928000 | +0.000000 |
| V1_TAIL_EXCESS | 1 | 0.956120 | 0.820689 | 0.866437 | 0.862990 | 0.938870 | 0.912000 | -0.001035 |
| V2_Q90_EXCESS | 1 | 0.956171 | 0.821867 | 0.866437 | 0.864582 | 0.938853 | 0.912000 | -0.000984 |
| V3_HETEROGENEITY | 1 | 0.956295 | 0.823954 | 0.866437 | 0.866934 | 0.943270 | 0.920000 | -0.000860 |
| V4_FULL_DMIL | 3 | 0.956201 | 0.822382 | 0.866437 | 0.865855 | 0.938870 | 0.912000 | -0.000954 |

Terminal: **`STOP_DMIL_TRAIN_GATE_FAILED`**

Full D-MIL delta AUROC: `-0.000954`; delta AP: `-0.007530`; nonnegative folds: `1/5`.

No official TEST data were read by this TRAIN-only run.

## Parameter signs

- `beta_T` median `+0.344322`, positive/negative/zero folds `5/0/0`.
- `beta_Q` median `+0.078531`, positive/negative/zero folds `5/0/0`.
- `beta_S` median `+0.184481`, positive/negative/zero folds `5/0/0`.
