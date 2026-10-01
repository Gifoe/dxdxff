# Frozen layer mapping

All tensors come from one evaluation-mode forward pass of the SHA-256-locked official CNN. No hook changes state and no BatchNorm statistics are updated.

| Audit stage | Exact source operation | Per-segment vector |
|---|---|---:|
| R0 | output of `NeuralCNNPreProcessing.__call__` after `normalize_img` | 453 |
| R1 | `feature_extractor[1](feature_extractor[0](R0_image))`; spatial mean and population std for each of 16 channels | 32 |
| R2 | `feature_extractor[3](feature_extractor[2](R1_tensor))`; spatial mean and population std for each of 32 channels | 64 |
| R3 | flattened output of `cnn.avgpool` after ResNet18 `layer4` | 512 |
| R4 | `cnn.fc(R3)`, the 32D representation exported by the prior frozen embedding pipeline | 32 |
| R5 | `fc_out(bn1(relu1(fc1(bn(relu(fc(R4)))))))` | 1 |

R0 contains five global statistics, the exact 224-value frequency marginal, and a fixed 224-bin temporal marginal. Its quantiles use a 256-bin histogram on the locked 0–255 normalized input, with at most 0.5 intensity-unit discretization error.

## Prompt/model mismatch for the task direction

The prompt assumes that the final classifier has a weight `w in R^32` directly on R4. That layer does not exist in the frozen model. The real classifier after R4 is nonlinear: `Linear(32,32) -> LeakyReLU -> BatchNorm -> Linear(32,16) -> LeakyReLU -> BatchNorm -> Linear(16,1)`. Treating any 32D matrix as the final classifier weight would fabricate a direction that the model does not contain.

The task-direction audit therefore additionally extracts `P16`, the true 16D input to `fc_out`, and uses the actual `fc_out.weight in R^16`. R4 remains the required 32D audit representation for every other analysis. Task-direction outputs explicitly identify their layer as `P16_TRUE_CLASSIFIER_INPUT`.
