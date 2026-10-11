# v1.0.0 发布前检查表

## A. 代码

- [ ] `TPCA_RBPCA`、`TPCA_Greed_test`、`Non_Greedy` 的多分量输出经过维度、正交性、有限值和退化样本测试。
- [ ] 零聚合向量、零符号、近满秩 q 和 `eigs` 边界均有确定性处理。
- [ ] 所有基线均有来源、版本、引用和许可证。
- [ ] `src/`、`config/` 和正式 `experiments/` 入口不存在 `H:\\`、`C:\\` 等绝对路径；`experiments/legacy/` 中的历史绝对路径必须继续标记为不可直接运行。
- [ ] 所有正式入口从配置文件读取参数，不依赖脚本内部隐藏变量。

## B. 数据与协议

- [ ] 每个数据集有 loader、变量名、维度、类别数、预处理和许可证说明。
- [ ] train/validation/test 的比例和分层规则已冻结。
- [ ] 每个 repeat 都有独立 seed、split index 和污染 mask。
- [ ] 污染只作用于训练观测集，测试集保持 clean，除非正文明确另设实验。
- [ ] q 使用 validation-selected 规则；test-oracle 已与主结果分离。

## C. 统计结果

- [ ] 至少 10 次完整重复，保存逐次 raw 结果。
- [ ] 汇总表包含 `n_valid`、均值、SD、95% CI 和失败运行数。
- [ ] 同一 seed/split 的方法比较使用 paired 统计。
- [ ] 图表没有把 across-q 波动写成 across-repeat SD。
- [ ] 所有最终数字均能由脚本从 raw CSV 重新生成。

## D. 论文映射

- [ ] 每张图和表都有唯一配置文件及复现命令。
- [ ] 正文方法名与仓库函数名完全一致。
- [ ] 所有理论结论标明适用条件，不把条件单调性写成全局最优。
- [ ] 讨论部分写明数据缺失、基线依赖、计算成本和失败边界。
- [ ] Data availability、Code availability、Funding、Competing interests、Author contributions 已填写。

## E. 发布

- [ ] `README.md` 已切换为 `v1.0.0` 的真实可复现声明。
- [ ] `VERSION`、`CHANGELOG`、`CITATION.cff` 和 Git tag 一致。
- [ ] ZIP/Release 包通过完整性检查并记录 SHA256。
- [ ] DOI、数据许可证和第三方许可证已补齐。
