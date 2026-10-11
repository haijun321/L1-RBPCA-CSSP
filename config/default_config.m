function cfg = default_config()
%DEFAULT_CONFIG Configuration template for the L1-RBPCA release candidate.
% This is a planning/configuration contract; it does not make the legacy
% result archive paper-ready.

cfg.releaseStatus = '0.1.1-review-candidate';
thisFile = mfilename('fullpath');
if isempty(thisFile)
    repoRoot = pwd;
else
    repoRoot = fileparts(fileparts(thisFile));
end
cfg.repoRoot = repoRoot;
cfg.matlab.minVersion = 'R2020b';
cfg.dataRoot = fullfile(repoRoot, 'data');
cfg.outputRoot = fullfile(repoRoot, 'runs');
cfg.baseSeed = 1000;
cfg.nRepeats = 10;                 % required for the paper-ready rerun
cfg.split.trainRatio = 0.70;
cfg.split.validationRatio = 0.00;  % outer validation is nested inside training
cfg.split.testRatio = 0.30;
cfg.split.innerValidationRatio = 0.20;
cfg.split.description = 'Outer stratified 70/30; inner validation only';
cfg.split.stratified = true;
cfg.corruption.trainOnly = true;
cfg.corruption.types = {'clean','mild','moderate','severe','extreme'};
cfg.corruption.ratios = [0.00 0.20 0.40 0.60 0.80];
cfg.corruption.blockSize = [];       % dataset-specific, must be logged
cfg.corruption.amplitude = [];       % must be logged per dataset/type
cfg.selection.main = 'validation-selected';
cfg.selection.diagnostic = 'test-oracle';
cfg.methods = {'L1-RBPCA','TPCA-L1-Greedy','NonGreedy','BPCA','GLRAM','PE'};
cfg.requiredArtifacts = {'raw_per_repeat.csv','summary.csv','config.json','split_indices.mat','run_manifest.txt'};
end
