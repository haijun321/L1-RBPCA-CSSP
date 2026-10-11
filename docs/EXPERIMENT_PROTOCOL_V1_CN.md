# CSSP v1.0.0 正式实验协议（发布前冻结版）

本文档规定从 `0.1.1-review-candidate` 产生 CSSP 论文正式结果时必须遵守的实验协议。`results/legacy/` 中的历史结果不属于本协议，不能直接作为论文主表或主图的统计证据。

## 1. 版本与方法冻结

正式运行前应冻结以下文件并记录 Git commit：

- `config/final_cssp_v1.m`：所有路径、参数、随机种子和输出目录；
- `config/method_manifest.csv`：论文方法名与 MATLAB 函数的唯一映射；
- `src/algorithms/`：L1-RBPCA、L1-Greedy、L1-NonGreedy；
- `src/baselines/`：所有 L2 对照方法；
- `experiments/run_experiment_v1.m`：唯一正式运行入口。

正式结果中不允许出现未在 `method_manifest.csv` 中登记的方法名、旧脚本自动生成的结果或手工修改的汇总数字。

## 2. 算法正确性门槛

L1-RBPCA 的每一个分量 `j` 必须使用当前的 `U(:,j)` 和 `V(:,j)` 进行成对交替更新，并在提取后移除先前分量。代码不得将第一个分量的投影缓存后复用于后续分量。零点处理和接近零分母处理必须是确定性的，并记录在 `info.zero_count` 与 `info.denominator_guard_count` 中。

每次调用算法必须返回：目标函数轨迹、迭代次数、收敛状态、停止原因、运行时间和异常信息。算法错误不得通过空 `try/catch` 变成 `NaN` 后继续汇总。

正式运行前必须通过：

1. 输出维度检查；
2. `U`、`V` 正交性检查；
3. 目标函数轨迹和停止准则检查；
4. 固定输入、固定 seed 的重复运行一致性检查；
5. 零分母、全零样本、`q=1` 和接近满秩情形的边界测试。

## 3. 数据、划分和污染

最终稿件中必须明确最终数据集列表。建议使用 AUSLAN、YALE、JAPAN、PIE、XM2VTS、ORL、NATOPS、FingerMovements、Heartbeat 和 AR14；任何无法公开获得、无法核验标签或许可不清的数据集应从主结果中删除，或移入补充实验并在 README 中说明。

每个数据集需要有 `data_manifest.csv`，记录来源、引用、许可证、MAT 变量名、样本数、矩阵尺寸、类别数和 SHA256。

每个重复采用独立的分层外层 70/30 train/test split。测试集始终保持干净。若需要验证集，在外层训练集内部使用固定的分层 inner split 或 5-fold validation；不得根据外层测试集选择维度、阈值或超参数。

污染协议固定为：

| 场景 | 污染比例 |
|---|---:|
| Clean | 0.00 |
| Mild | 0.20 |
| Moderate | 0.40 |
| Severe | 0.60 |
| Extreme | 0.80 |

污染类型、幅度、边界处理、是否允许同一样本多次污染必须在配置文件和 Methods 中写明。每个重复先生成一份训练污染 mask，所有方法共享相同的 split 和 mask；禁止在旧污染数据上继续叠加新污染。

## 4. 重复次数和维度选择

每个 `dataset × scenario × repeat × method` 运行 10 次完整 pipeline。每次重复重新生成分层划分、训练污染和算法初始化。开发期 3 次或 5 次运行只能标记为 development/legacy，不能用于正式均值、标准差或显著性检验。

主结果使用一种预先声明的、无测试泄漏的 q 选择规则。推荐外层 70/30 划分后，在外层训练集内用 inner validation 选择候选 `q=1:qMax` 中验证 L1 重构误差最小的 q；并列时取最小 q。锁定 q 后，应使用完整外层训练集（按同一污染协议重新拟合）得到最终模型，最后只在外层干净测试集报告一次结果。不得直接用 inner-train 模型评估外层测试集。

如果正文保留 CAP/knee 作为贡献，CAP 只能使用训练目标曲线计算，并给出目标函数的正式定义、候选 q 范围、并列处理和失败回退规则。测试集最优 q 只能作为附录诊断，不能进入主表。

## 5. 指标与统计

正式主表至少报告：

- 干净测试集归一化 L1 重构误差；
- 归一化 Frobenius 重构误差（若正文使用）；
- 欧氏 1-NN 与白化 Mahalanobis 1-NN 识别率；
- 选择的 q、运行时间、迭代次数、收敛率和失败率。

若报告子空间稳健性，必须定义 clean-reference 子空间以及 principal-angle 或 projection-distance 的计算方式；没有真值时不得写成“准确恢复”。

均值和标准差必须跨 10 个完整重复计算。q 曲线上不同 q 点的标准差不得标为重复实验标准差。建议同时报告 paired bootstrap 95% CI、median/IQR 和效应量。

预注册的配对检验为：同一重复划分下，L1-RBPCA 与各主要基线进行 paired Wilcoxon signed-rank test，报告中位数差、95% bootstrap CI 和 Cliff's delta；多重比较使用 Holm 校正。所有统计结果均从 raw long-format 表自动生成。

## 6. 原始日志和文件结构

每次运行至少写入以下字段：

```text
dataset,scenario,repeat,split_seed,corruption_seed,init_seed,
method,p1,p2,n,q_max,q_selected,objective_train,
nre_test,fro_test,accuracy_euclidean,accuracy_mahalanobis,
subspace_distance,runtime_sec,iterations,converged,stop_reason,
status,error_message,matlab_version,git_commit,config_hash,
split_hash,mask_hash,timestamp
```

同时保存：

- `raw/runs.csv`：逐方法、逐重复记录；
- `raw/splits/<dataset>_repXX.mat`：训练/测试索引；
- `raw/masks/<dataset>_<scenario>_repXX.mat`：污染 mask 和参数；
- `raw/seeds.csv`、`raw/config_final.json`、`raw/environment.txt`；
- `summary/`：由脚本从 raw 自动聚合的结果；
- `provenance/`：Git commit、配置哈希、输出 SHA256。

## 7. 推荐运行顺序

```matlab
matlab -batch "addpath(genpath('.')); cfg=final_cssp_v1; run_selftest(cfg);"
matlab -batch "addpath(genpath('.')); cfg=final_cssp_v1; run_experiment_v1(cfg,'dataset','auslan','repeatRange',1:10);"
matlab -batch "addpath(genpath('.')); aggregate_v1_results('results/v1.0.0/raw','results/v1.0.0/summary');"
matlab -batch "addpath(genpath('.')); make_v1_figures('results/v1.0.0/summary','figures/v1.0.0');"
matlab -batch "addpath(genpath('.')); verify_release_v1('results/v1.0.0');"
```

脚本应支持按数据集和重复区间断点续跑，并检查期望行数：

```text
number_of_datasets × 5 scenarios × 10 repeats × number_of_methods
```

## 8. v1.0.0 发布门槛

只有在以下条件全部满足后，才将版本从 `0.1.1-review-candidate` 改为 `v1.0.0`：

1. 正文主表和主图全部由 `results/v1.0.0/summary` 自动生成；
2. 每个计划组合均有 10 次完整重复，失败组合有显式记录；
3. 主结果没有测试集选 q 或测试集调参；
4. 没有硬编码 `H:/`、`C:/` 等本机路径；
5. 从 clean clone 可以运行 selftest 和一个小型示例；
6. README 包含 MATLAB 版本、工具箱、数据来源/许可证、完整命令、预计运行时间和稿件表图映射；
7. 发布包包含 `CITATION.cff`、`CHANGELOG.md`、`figure_table_manifest.csv` 和 SHA256 清单。

