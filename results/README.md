# Result archive and status

The files in this directory preserve the supplied outputs for auditability. They are **not** certified CSSP main-table results. No result is promoted to `final_results` in this release.

| Tree | What is present | Audit status | Permitted use |
|---|---|---|---|
| `legacy/robust_outliers_noPCA_kneeAvg` | 12 one-row CSV summaries and figures for AUSLAN/BCI/JAPAN/PEMS | `verified_small_run` only: 3 reported repeats, one shared split, no raw per-repeat records; block-size tags conflict with current script | Internal diagnostic only |
| `legacy/result_2601` | Tsummary CSV/MAT and figures for AUSLAN/JAPAN/Wafer | `legacy_oracle_unverified`: generating script, seed, split, and repeat count absent; BestAcc/BestQ are test-curve selections; L1RBPCA and TPCA-L1-Greedy are identical in many rows | Appendix/history only |
| `legacy/result3` | Old XLSX/PNG/FIG recognition outputs | `legacy_unverified`: `ErrorStdDev` is across-q curve variation, not repeated-run SD; labels and settings differ from current scripts | Archive only |
| `legacy/PEMS重构及如果图` | Historical reconstruction images/PDF/FIG and nested archive | `legacy_unverified`: filenames/configuration differ from current reconstruction script (e.g., block 123 vs 129) | Visual history only |

For a paper-ready release, every row must be regenerated from a tagged commit with raw per-repeat output, split indices, seed, data checksum, configuration, and a command that recreates the table/figure.
