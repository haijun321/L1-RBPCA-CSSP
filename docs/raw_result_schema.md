# Raw result schema for the paper-linked release

Every complete run should write one row per `(dataset, corruption, repeat, method, q)` before any aggregation. The following fields are required.

| Field | Meaning |
|---|---|
| `dataset` | Canonical dataset identifier |
| `data_checksum` | SHA256 or provider checksum of the input file |
| `commit` | Git commit used for the run |
| `matlab_version` | MATLAB release |
| `repeat` | 1-based independent repeat index |
| `seed` | Exact random seed |
| `split_id` | Identifier of saved train/validation/test indices |
| `method` | Exact paper method name |
| `q_row`, `q_col` | Candidate bilateral dimensions |
| `q_selected_by` | `validation`, `knee`, or `test_oracle` |
| `corruption_type` | `clean`, `stripe`, `block`, `impulse`, or declared type |
| `corruption_rate` | Nominal contamination fraction |
| `block_size` | Requested block size |
| `actual_corrupted_fraction` | Fraction after boundary handling |
| `train_only_corruption` | Boolean |
| `objective_train` | Training objective at the selected q |
| `reconstruction_test` | Clean test reconstruction metric |
| `classification_test` | Test classification metric, if applicable |
| `runtime_seconds` | End-to-end method runtime |
| `iterations` | Iteration count |
| `converged` | Boolean |
| `status` | `success`, `failed`, or `invalid` |
| `failure_reason` | Nonempty for failed/invalid runs |

Undefined ratios must be recorded as `NaN` with a documented reason. Do not silently convert failed runs to zero.
