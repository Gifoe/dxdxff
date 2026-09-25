# P2-Q10-NPAM Unified Implementation Report

本目录已统一到固定 outer-CV、成功阳性的单一实现。

## 已实现

1. Task 2 标签固定为 success=1、failure=0，Engel I/II–IV 映射一致。
2. BCE 只训练 `outcome_logit_success`；失败概率和指标为派生量。
3. PAM 失败风险内部命名、Q10 方向和掩码统一，真实 `q10_adjustment` 参与残差正则。
4. 原始图固定三阶段，只有两组相邻差分，图覆盖分母为有效发作数×3。
5. `data_exclusions.csv` 在所有数据路径前置应用，并输出逐规则删除审计。
6. runner 仅有 `quick|outer_cv`，不包含 inner fold、阈值搜索、patience、validation 或 early stopping 参数/调用。
7. M0–M4 固定 A=30；M5 固定 A=30、B=15；最终保存 `final_npam.pt`。
8. threshold 固定 0.5；bootstrap 离散指标复用已存 `outcome_pred_05`。
9. strict fold/P2 leakage/checkpoint fold metadata/patient OOF 门禁已实现。
10. 云端真实 repo、Python、28D feature、raw、fold、P2 与 adapter audit 地址已写入配置。

## 验证边界

本地 pytest 和有界 smoke 只验证实现路径，不构成论文性能结果。只有服务器上严格跑满固定五折、通过全部 P2 泄漏与 metadata 门禁的结果才可标记 paper-valid；否则报告明确写入 `OUTER_CV_NOT_PAPER_VALID`，未跑满写入 `FULL_OUTER_CV_NOT_RUN`，quick 写入 `QUICK_SCREENING_NOT_PAPER_VALID`。

## 本地验证记录

- strict input audit：149 人（success 90 / failure 59）、474 feature/raw runs 完全对齐、五折 149 人精确覆盖、5 个 P2 checkpoint embedding export 与 fold metadata 均通过。
- bounded strict graph：包含 13 人、36 次发作、108 个三阶段图；108/108 有效。
- `lzu:tuoyongxiang/SZ2`：feature/raw 各删除 2 条，剩余 1 次发作，患者保留；图缓存无任何 SZ2。
- quick：`M0_PAM`、`M2_PHASE_NETWORK`、`M5_LIMITED_FINETUNE` 均完成单折 1-epoch 路径；M5 的 Stage A/Stage B 均执行。
- quick OOF：成功/失败概率逐行互补，阈值文件固定 0.5，真实 Q10 adjustment 非零，最终 checkpoint metadata 完整。
- 本地旧 149 人 Task 2 folds 与 sensitivity80 P2 ledger 不对齐，因此 strict dry gate 正确拒绝。远程配置不再使用该本地 folds，而使用服务器的 `fixed_outer_folds_v3_qbc.csv`（80 名 sensitivity80 成功患者 + 失败患者）；最终是否 paper-valid 由远程逐折患者交集门禁决定。
- 远程单类别 fold 报错修复：新增 `prepare_sensitivity80_plus_failures.py` 和 PowerShell ensure 步骤。80 名成功患者的折号来自 P2 test ledger，失败患者使用固定 Task 2 折号，生成显式组合 cohort/folds。验证得到 139 人（80 success / 59 failure），五折均含两类；M0 quick 与 fold-1 strict outer gate 均通过。
