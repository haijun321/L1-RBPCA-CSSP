# 论文最终发表版补充清单

## 当前判断

当前仓库可以作为 `0.1.1-review-candidate` 上传，但还不能标记为论文关联的 `v1.0.0`。原因不是文件数量不够，而是论文中的算法、实验协议和结果文件还没有形成可审计的一一对应关系。

CSSP/CSDA 目标期刊关注计算方法、算法验证和稳健程序。最终版本应让审稿人从一个干净目录开始，能够知道：使用了什么数据、怎样划分、怎样污染、使用什么随机种子、怎样选择维度、哪条命令生成哪张图表，以及结果是否来自独立重复实验。期刊范围本身涵盖统计软件/算法的开发与验证以及稳健程序，因此代码和实验验证必须与方法主张同样完整。

## P0：必须补齐后才能称为论文最终版

### 1. 冻结算法与论文定义

- 明确最终方法名称只使用 `L1-RBPCA`，并建立论文符号、函数名、输出字段的对应表。
- 修复 `TPCA_RBPCA` 和 `TPCA_Greed_test` 对后续分量复用第一个分量投影的问题。
- 明确每个分量的双侧正交补投影、极性交替更新、零聚合向量回退、收敛判据和最大迭代次数。
- 对固定极性子问题只声称条件最优或单调上升；不要声称原始非凸 L1 问题的全局最优。
- 对 `Non_Greedy`、L2 基线和论文中的 UMPCA/TROD/PARAFAC 建立一一对应关系。若仓库没有实现，就不能在正文中把它们写成已复现基线。

### 2. 建立唯一的论文实验配置

每个正式实验必须由一个版本化配置文件启动，至少记录：

- 数据集、变量名、形状、类别数和数据版本/校验和；
- 中心化、归一化和样本排列方式；
- train/validation/test 比例和分层规则；
- 每个 repeat 的 seed 和保存的 split indices；
- q 候选范围、停止容差、最大迭代次数、初始化方式；
- 异常类型、污染比例、块大小、幅值、边界处理和实际污染面积；
- 训练集是否污染、测试集是否保持 clean；
- MATLAB 版本、Toolbox、线程数、CPU/内存和运行时间。

### 3. 重新生成论文结果

正式主结果必须从同一个 tagged commit 重新生成：

- 至少 10 次完整独立重复；每次重新分层划分，不能只改变初始化；
- q 由 validation-selected 规则确定；test-oracle 结果只能放补充材料；
- 导出每个 repeat 的 raw CSV/JSON，而不是只保留一行均值；
- 同时报告 mean、SD、95% CI、有效重复数和失败运行数；
- 采用 paired seed/split 比较，并在需要时报告 Wilcoxon/Friedman + Holm 校正；
- 所有表格和图形由脚本从 raw 结果重新导出；不手工复制旧 Excel 或旧 PNG。

### 4. 补齐论文主张所需的实验

至少应覆盖以下证据链：

| 论文主张 | 必须提供的证据 |
|---|---|
| L1-RBPCA 的当前分量目标更稳定 | 与 Greedy 使用相同历史状态、相同初始化和相同 q 的 matched-history 曲线 |
| 顺序提取具有可解释的正交结构 | 正交误差、当前分量目标、累计目标和边际增益 |
| q 选择不是测试集调参 | validation-selected q、测试集一次性评估、test-oracle 仅作诊断 |
| L1 对结构污染更稳健 | clean/mild/moderate/severe/extreme 的重构误差和分类结果 |
| 方法改进来自具体机制 | SVD 更新、极性处理、正交补、knee/CAP 规则的消融 |
| 方法具有实际计算可用性 | 运行时间、迭代次数、内存/复杂度和失败率 |

## P1：正式投稿前应加入

- `docs/paper_table_figure_map.csv`：每张表/图对应的脚本、配置、输入结果、输出文件和 commit；
- `docs/method_map.csv`：论文方法名、仓库函数名、引用、许可证和是否实际运行；
- `docs/environment_capture.md`：MATLAB、Toolbox、操作系统、硬件和线程设置；
- `docs/raw_result_schema.md`：raw 结果字段定义和缺失值规则；
- `data/dataset_manifest.csv`：数据来源、下载地址、许可证、变量名、形状和 checksum；
- `results/final/`：仅存放经过审计的论文结果；历史结果继续保留在 `results/legacy/`；
- `supplementary/`：算法伪代码、完整超参数、每个数据集的逐次结果、额外图表、失败案例和敏感性分析；
- 至少一个从干净环境执行的 smoke test 和一个小规模 end-to-end run；
- 结果目录中的 `run_manifest.json`，记录 commit、配置、数据 checksum、seed、开始时间和 MATLAB 版本。

## P2：发表后维护内容

- GitHub Release 标签 `v1.0.0`；
- Zenodo DOI 或其他长期存档 DOI；
- `CITATION.cff` 中更新 DOI 和正式版本；
- Release notes 说明哪些数据因许可证没有随仓库分发；
- 第三方代码、基线和数据集的完整引用及许可证；
- 可选的 GitHub Actions/MATLAB CI，仅用于函数级测试，不替代正式实验。

## 不应加入最终主结果的内容

- 没有生成脚本和 seed/split 元数据的 `result_2601`；
- 将 q 曲线波动误写成重复实验 SD 的 `result3`；
- 与当前脚本块大小、污染比例或数据集标签不一致的历史图片；
- 测试集扫描得到的 `BestAcc/BestQ`，除非明确标记为 test-oracle 诊断；
- 未确认实现和引用的 UMPCA/TROD/PARAFAC 结果；
- 任何机器绝对路径、个人电脑日志或未授权数据。
