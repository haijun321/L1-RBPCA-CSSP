# 投稿版（2）GitHub 发布审计报告

## 结论

本仓库是根据提交的 `投稿版 (2).zip` 整理出的 **review candidate / 历史实验快照**。它可以作为代码审计和历史结果归档上传 GitHub，但不能在当前状态下宣称“完整复现 CSSP 稿件中的所有表格和图”。

## 立即需要修正的 P0 问题

1. `TPCA_RBPCA.m` 与 `TPCA_Greed_test.m` 对第 2 个及后续分量复用了第 1 个分量的投影数据，和论文中双边正交补空间的逐分量提取叙述不一致。
2. 论文若使用 dead-zone sign、零向量确定性回退和成对交替优化，当前函数未完整实现这些机制。
3. `demo_recognitionrate0704.m` 含机器绝对路径，调用缺失的 `dimensionality_reduction_and_classification`/`NewGreedy`，且四参数调用 `add_outliers`（函数需要五个参数）；其所谓 `num_repeats=10` 的循环主体没有执行实验。
4. `demo_select_dim_robust_outliers_0419.m` 的 train/validation/test 划分在 repeat 循环外；3 次重复只改变部分随机初始化，不能作为独立重复实验。脚本中的块大小标签和归档 CSV 也不一致。
5. `result_2601` 没有生成脚本、split、seed 或 repeat 元数据，且 `MeanBestAcc/BestQ` 是测试曲线上选最优维度，属于 test oracle。
6. `result3` 的 `ErrorStdDev` 是 q 曲线波动，不是重复实验标准差。
7. 数据、中心化、异常注入幅值/边界、评价口径和方法映射均未形成一个统一配置。

## 当前结果可如何写

- robust 目录：最多写为 3 次、共享一次划分的内部诊断；不能写成 10 次独立重复的最终证据。
- result_2601：写为未审计的历史聚合，并明确最优维度使用了测试曲线；不能作为公平主表。
- result3 和重构图：仅作历史图像资产；不能当作均值±标准差的正式结果。

## 达到 v1.0.0 的必要实验协议

1. 给每个数据集提供无绝对路径 loader，并记录变量名、形状、类别和许可证。
2. 固定一个公开配置：例如 train/test=70/30；若使用 validation，则公开 train/validation/test 的实际比例。
3. 每个 repeat 重新分层划分，并保存 split indices；固定 seeds（例如 `baseSeed + repeat`）。
4. 异常只注入训练观测集；测试集保持 clean，并记录类型、比例、块大小、幅值和实际注入面积。
5. 用 validation-selected q 作为主结果；test-oracle q 只能作为补充诊断。
6. 至少 10 次完整独立重复，导出每次的 raw CSV/JSON、均值、标准差、95% CI 和配置快照。
7. 修复并单元测试 L1-RBPCA、多分量提取、零向量回退、1-NN 距离和中心化/重构口径。
8. 给每张论文图/表写复现命令，并在结果中写入 git commit、MATLAB 版本和数据 checksum。

## 发布前门槛

只有完成上述 P0/P1 项并重新导出结果后，才能把版本从 `0.1.1-review-candidate` 改为论文关联的 `v1.0.0`。
