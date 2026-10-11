# Data contract

The submitted archive does **not** contain the original datasets. Do not commit restricted or third-party data to this repository.

The legacy scripts refer to external MATLAB files including (but not limited to):

- `auslan.mat`
- `bci.mat`
- `japan.mat`
- `PEMSsorttrans.mat`
- `yaledata_norm_64x64.mat`
- ORL, XM2VTS, NATOPS, FingerMovements, Heartbeat, AR14 and related files mentioned in the manuscript

Before a reproducible release, add one documented loader per dataset. Each loader must record the exact variable name, tensor shape `(rows, columns, samples)`, class labels, normalization/centering, and the dataset license. A local data root should be supplied through `cfg.dataRoot`; no absolute `H:\` or `C:\` paths are permitted.

The release candidate therefore runs only a synthetic smoke test. It does not silently download data or fabricate missing results.
