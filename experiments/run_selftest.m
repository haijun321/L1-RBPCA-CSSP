function report = run_selftest()
%RUN_SELFTEST Small synthetic smoke test for the organized source snapshot.
% It checks dimensions and finite outputs only; it is not a paper experiment.

repoRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(repoRoot, 'src')));
rng(7, 'twister');
X = randn(8, 6, 12);
q = [2 2];
mo.U = orth(randn(6, q(2)));
mo.V = orth(randn(8, q(1)));
tol = 1e-6;
rows = {'TPCA_RBPCA'; 'TPCA_Greed_test'; 'Non_Greedy'; 'BPCA'; 'GLRAM'; 'projected_estimator'; 'reconstruct'};
status = false(numel(rows),1);
message = cell(numel(rows),1);

try
    m = TPCA_RBPCA(mo, X, q, tol);
    status(1) = isfield(m,'U') && isfield(m,'V') && all(isfinite(m.U(:))) && all(isfinite(m.V(:)));
    message{1} = 'finite U/V and fields present';
catch ME
    message{1} = ME.message;
end
try
    m = TPCA_Greed_test(mo, X, q, tol);
    status(2) = isfield(m,'U') && isfield(m,'V') && all(isfinite(m.U(:))) && all(isfinite(m.V(:)));
    message{2} = 'finite U/V and fields present';
catch ME
    message{2} = ME.message;
end
try
    m = Non_Greedy(mo, X, q, tol);
    status(3) = isfield(m,'U') && isfield(m,'V') && all(isfinite(m.U(:))) && all(isfinite(m.V(:)));
    message{3} = 'finite U/V and fields present';
catch ME
    message{3} = ME.message;
end
try
    [R,C] = BPCA(X, 2, 2);
    status(4) = isequal(size(R),[8 2]) && isequal(size(C),[6 2]) && all(isfinite(R(:))) && all(isfinite(C(:)));
    message{4} = 'expected BPCA dimensions';
catch ME
    message{4} = ME.message;
end
try
    opts = zeros(1,14); opts(14)=2;
    [~, optsOut] = GLRAM(mo, X, 2, 2, opts); %#ok<ASGLU>
    status(5) = true; message{5} = 'GLRAM returned';
catch ME
    message{5} = ME.message;
end
try
    [Rt,Ct] = projected_estimator(mo, X, 2, 2);
    status(6) = isequal(size(Rt),[8 2]) && isequal(size(Ct),[6 2]);
    message{6} = 'expected projected-estimator dimensions';
catch ME
    message{6} = ME.message;
end
try
    Y = reconstruct(X, mo.V, mo.U);
    status(7) = isequal(size(Y),size(X)) && all(isfinite(Y(:)));
    message{7} = 'expected reconstruction dimensions';
catch ME
    message{7} = ME.message;
end
report = table(rows, status, message, 'VariableNames', {'Function','Pass','Message'});
disp(report);
if ~all(status)
    fprintf('Smoke test completed with one or more reported failures. Review the message column; failures are not hidden.\n');
else
    fprintf('Smoke test passed for all checked calls. This does not validate paper results.\n');
end
end
