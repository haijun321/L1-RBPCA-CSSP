function report = reproduce_all()
%REPRODUCE_ALL Safe entry point for the release candidate.
% It does not execute legacy scripts with machine-specific paths.
repoRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(repoRoot, 'src')));
addpath(fullfile(repoRoot, 'experiments'));
report.smoke = run_selftest();
report.dataRoot = fullfile(repoRoot, 'data');
report.externalDataPresent = ~isempty(dir(fullfile(report.dataRoot, '*.mat')));
report.message = ['Historical drivers are archived under experiments/legacy. ' ...
    'A paper-grade reproduction still requires dataset loaders, split indices, ' ...
    'seeds, a tagged algorithm implementation, and a canonical rerun.'];
disp(report.message);
end
