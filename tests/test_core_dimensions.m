function report = test_core_dimensions()
%TEST_CORE_DIMENSIONS Wrapper for the release-candidate smoke test.
repoRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repoRoot, 'experiments'));
report = run_selftest();
end
