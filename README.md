# L1-RBPCA — CSSP review candidate

> **Status: `0.1.1-review-candidate`**
>
> This repository is an organized snapshot of the supplied MATLAB code and historical outputs. It is suitable for code review and archival upload. It is **not yet a complete, paper-grade reproduction package**. Existing result files are deliberately kept under `results/legacy/` and are not advertised as final CSSP evidence.

## What is included

- MATLAB source functions organized under `src/`;
- original experiment drivers under `experiments/legacy/`;
- all supplied historical figures/tables under `results/legacy/`;
- a reproducibility configuration template in `config/default_config.m`;
- a synthetic smoke test in `experiments/run_selftest.m`;
- data and third-party-method contracts;
- a Chinese release audit in `docs/RELEASE_AUDIT_CN.md`;
- a frozen protocol proposal in `docs/EXPERIMENT_PROTOCOL_V1_CN.md` and `config/final_cssp_v1.m`;
- a static release checker in `tools/preflight_release.py`;
- publication-release templates in `docs/PUBLICATION_SUPPLEMENT_PLAN_CN.md`, `docs/RELEASE_CHECKLIST_CN.md`, and `docs/raw_result_schema.md`.

## Quick start

Use MATLAB R2020b or newer. The helper functions also require Statistics and Machine Learning Toolbox for some legacy recognition paths. The smoke test does not need the external datasets.

```matlab
repoRoot = pwd;
addpath(genpath(fullfile(repoRoot, 'src')));
addpath(fullfile(repoRoot, 'config'));
addpath(fullfile(repoRoot, 'experiments'));
report = run_selftest();
```

`report` is a dimension/finite-value smoke test, not a validation of the manuscript results. The expected behavior is a printed table with one row per function and an explicit PASS/FAIL status. Some legacy functions may fail on the synthetic call; such a failure is reported rather than hidden.

For a static candidate check (does not execute MATLAB):

```bash
python tools/preflight_release.py
python tools/preflight_release.py --final  # expected to fail until paper/raw v1.0 artifacts exist
```

To inspect the release configuration:

```matlab
cfg = default_config();
disp(cfg);
```

`reproduce_all()` runs the smoke test and explains which external data and metadata are still missing. It does not run legacy scripts automatically.

## Current source-to-paper caveat

The archive contains functions named `TPCA_RBPCA`, `TPCA_Greed_test`, and `Non_Greedy`, but the current implementations do not yet establish that the manuscript's full pairwise bilateral orthogonal-complement, dead-zone sign, and deterministic fallback algorithm is implemented for every extracted component. The P0 algorithm/protocol alignment must be completed before using this repository to support a final claim.

The code also contains baseline names (`BPCA`, `GLRAM`, `projected_estimator`) that are not a substitute for UMPCA/TROD/PARAFAC unless their implementations and citations are added. Keep the method names in the paper, config, scripts, and tables identical.

## Data

Original datasets are not included. See [`data/README.md`](data/README.md) for the legacy filenames and the required loader contract. Do not commit restricted datasets without permission. Every final run must record the data checksum and tensor layout `(rows, columns, samples)`.

## Legacy result protocol

| Result family | Reported setup | Repeat evidence | Status |
|---|---|---:|---|
| `robust_outliers_noPCA_kneeAvg` | train 0.70, validation 0.10, test 0.20 in the script; impulse; old block-size tag 13 | 3, but one shared split and no raw per-repeat output | diagnostic only |
| `result_2601` | impulse; old blocks 13/5/50; BestAcc and BestQ | repeat count not recorded; possible legacy aggregation | unverified/test-oracle |
| `result3` | old q scans and mixed ratios/block sizes | no explicit independent repeats; `ErrorStdDev` is across-q variation | legacy archive |
| reconstruction figures | historical PEMS/AUSLAN outputs | single or unknown runs; configuration differs from current script | visual archive |

These outputs cannot be merged into one mean±standard-deviation table. In particular, do not describe `ErrorStdDev` as repeat SD, and do not use test-selected q as the primary comparison.

## Paper-ready protocol still required

The planned paper-linked protocol in `config/final_cssp_v1.m` uses an outer stratified 70/30 train/test split and an inner validation split inside the outer training data. Before a `v1.0.0` release, independently rerun the full pipeline at least 10 times; save seeds and split indices; contaminate training observations only; keep the outer test set clean; select q using inner validation; report test metrics once at that q; export raw per-repeat records plus aggregate mean, SD, CI, configuration, commit, and data checksum.

## Repository map

```text
src/                  MATLAB functions (current snapshot)
experiments/legacy/    Original drivers; do not assume clean-clone execution
experiments/run_selftest.m
experiments/reproduce_all.m
config/default_config.m
data/                  Data contract; datasets omitted
results/legacy/       Supplied historical outputs, clearly labeled
results/summary/      Machine-readable audit summaries
docs/                 Release audit and publication checklist
tests/                 Smoke-test wrapper
archive/              Source and nested archive traceability
docs/                 Publication protocol, schema, audit, and release checklists
paper/                Placeholder for author-owned manuscript source
supplementary/        Placeholder and content checklist for the supplement
environment/          Environment-record placeholder
results/final/        Reserved for newly regenerated paper-linked results
```

## Citation and license

See [`CITATION.cff`](CITATION.cff) and [`LICENSE`](LICENSE). Dataset and third-party licenses remain separate. Replace the provisional repository URL in `CITATION.cff` after creating the public GitHub repository.
