function cfg = final_cssp_v1()
%FINAL_CSSP_V1 Planning contract for the paper-linked rerun.
% This function intentionally stops short of claiming a completed rerun:
% dataset loaders, corrected algorithms, and formal run drivers are required.

cfg = default_config();
cfg.releaseStatus = 'planned-v1.0.0';
cfg.nRepeats = 10;
cfg.split.trainRatio = 0.70;
cfg.split.validationRatio = 0.00;
cfg.split.testRatio = 0.30;
cfg.split.innerValidationRatio = 0.20;
cfg.split.description = 'Outer stratified 70/30; validation is inside outer training data';
cfg.qSelection = 'inner-validation-selected';
cfg.mainResultSet = 'clean-outer-test';
cfg.testOracleSet = 'diagnostic-only';
cfg.scenarios = {'Clean','Mild','Moderate','Severe','Extreme'};
cfg.requiredFinalEntrypoints = {'run_experiment_v1','aggregate_v1_results','make_v1_figures','verify_release_v1'};
end
