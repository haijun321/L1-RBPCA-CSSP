function [model, audit] = TPCA_RBPCA(mo, data, q, tol, opts)
%TPCA_RBPCA Paper Algorithm 1: constrained paired sequential L1 extraction.
%
% [model,audit] = TPCA_RBPCA(mo, data, [q q], tol)
% [model,audit] = TPCA_RBPCA(mo, data, [q q], tol, opts)
%
% INPUTS
%   mo.V : h-by-q matrix of INITIAL left directions (paper L).
%   mo.U : w-by-q matrix of INITIAL right directions (paper R).
%   data : h-by-w-by-n real, finite, matrix-valued samples.
%   q    : scalar q or [q q]; asymmetric dimensions are NOT supported.
%   tol  : positive relative objective tolerance.
%   opts.maxIter     : maximum INNER iterations per direction pair (10000).
%   opts.eps0        : positive denominator safeguard (1e-12).
%   opts.usePageTimes: use pagemtimes if available (true).
%
% OUTPUTS
%   model.V: h-by-q learned left directions, paper L.
%   model.U: w-by-q learned right directions, paper R.
%   model.algorithm_id, model.audit: implementation identity and log.
%   audit.pair_objective(k): Phi_k(l_k,r_k) on all data samples.
%   audit.full_objective(k): COMPLETE entrywise L1 J(k), including crosses.
%   audit.phi_history{k}: Phi_k after each completed LEFT/RIGHT sweep,
%       starting with the feasible initialization.
%   audit.zero_*: exact-zero diagnostics (NOT a near-zero guarantee).
%
% ALGORITHM CONTRACT
%   * Within EACH k, alternate left and right conditional updates.
%   * Previously learned directions are fixed; aggregate is projected
%     onto the full corresponding historical orthogonal complement.
%   * Polarity is +1 at z>0 and -1 at z<=0, including exact zero.
%   * Exact zero projected aggregate: RETAIN current feasible direction.
%   * Nonzero projected vector: obtain oriented first SVD direction,
%     algebraically equivalent to projected-vector normalization.
%   * Stop on the CURRENT PAIR Phi_k, not the full J(k).
%   * No random perturbation is injected by this function.
%
% IMPORTANT FOR REPRODUCIBILITY
%   This algorithm DIFFERS from the archived, two-pass TPCA_RBPCA.m.
%   Historical numerical results MUST NOT be attributed to this version
%   without new runs and provenance. MATLAB R2020b+ is recommended.
%
% See docs/ALGORITHM1_ALIGNMENT.md and tests/test_TPCA_RBPCA_Algorithm1.m.

if nargin < 5 || isempty(opts), opts = struct(); end
if nargin < 4 || isempty(tol)
    error('L1RBPCA:MissingTol','Pass the explicitly chosen positive tol.');
end
if ~isstruct(mo) || ~isfield(mo,'V') || ~isfield(mo,'U')
    error('L1RBPCA:BadInitialModel','mo must contain initial V and U.');
end
if ~isnumeric(data) || ~isreal(data) || ~isfloat(data) || ...
        ndims(data)>3 || any(~isfinite(data(:)))
    error('L1RBPCA:BadData','data must be a finite real float h-by-w-by-n array.');
end
[h,w,n] = size(data);
if h<1 || w<1 || n<1
    error('L1RBPCA:EmptyData','Matrix dimensions and sample count must be positive.');
end
if isscalar(q), q = [q q]; end
validateattributes(q, {'numeric'}, {'vector','numel',2,'integer','positive','finite'},mfilename,'q');
if q(1) ~= q(2)
    error('L1RBPCA:AsymmetricQ','Algorithm 1 requires q_r=q_c.');
end
K = double(q(1));
if K>min(h,w)
    error('L1RBPCA:TooManyDirections','q exceeds min(h,w).');
end
validateattributes(tol,{'numeric'},{'scalar','positive','finite','real'},mfilename,'tol');
if ~isequal(size(mo.V),[h K]) || ~isequal(size(mo.U),[w K]) || ...
        ~isreal(mo.V) || ~isreal(mo.U) || ...
        any(~isfinite(mo.V(:))) || any(~isfinite(mo.U(:)))
    error('L1RBPCA:BadInitialization', ...
        'mo.V must be h-by-q and mo.U must be w-by-q; all real and finite.');
end
if ~isstruct(opts), error('L1RBPCA:BadOptions','opts must be a struct.'); end
if ~isfield(opts,'maxIter'),      opts.maxIter = 10000; end
if ~isfield(opts,'eps0'),         opts.eps0 = 1e-12; end
if ~isfield(opts,'usePageTimes'), opts.usePageTimes = true; end
validateattributes(opts.maxIter,{'numeric'}, ...
    {'scalar','integer','positive','finite'},mfilename,'opts.maxIter');
validateattributes(opts.eps0,{'numeric'}, ...
    {'scalar','positive','finite'},mfilename,'opts.eps0');
validateattributes(opts.usePageTimes,{'logical','numeric'}, ...
    {'scalar'},mfilename,'opts.usePageTimes');
usePageTimes = logical(opts.usePageTimes) && ...
    (exist('pagemtimes','builtin') == 5 || exist('pagemtimes','file') == 2);

% Reshaping does not change the sample layout: each page is h-by-w.
% Right-projected samples X_i' * l can be obtained by one BLAS multiply.
Xflat = reshape(data,h,w*n);
Vinitial = cast(mo.V,'like',data);
Uinitial = cast(mo.U,'like',data);
L = zeros(h,K,'like',data);
R = zeros(w,K,'like',data);

clockStart = tic;
audit = struct();
audit.algorithm_id = 'L1RBPCA_ALGORITHM1_PAIRWISE_V1_20261015';
audit.update_order = 'k pairwise; left then right within inner iteration';
audit.polarity_rule = 'z>0:+1, z<=0:-1';
audit.maxIter = opts.maxIter;
audit.eps0 = opts.eps0;
audit.tol = tol;
audit.usePageTimes = usePageTimes;
audit.pair_objective = zeros(K,1);
audit.full_objective = zeros(K,1);
audit.cross_increment = zeros(K,1);
audit.phi_history = cell(K,1);
audit.iterations = zeros(K,1);
audit.stop_reason = cell(K,1);
audit.zero_projections_left = zeros(K,1);
audit.zero_projections_right = zeros(K,1);
audit.zero_aggregates_left = zeros(K,1);
audit.zero_aggregates_right = zeros(K,1);
audit.init_fallback_left = false(K,1);
audit.init_fallback_right = false(K,1);
audit.monotone_violations = zeros(K,1);
audit.orth_error_left = zeros(K,1);
audit.orth_error_right = zeros(K,1);

for k = 1:K
    Hleft  = L(:,1:k-1);
    Hright = R(:,1:k-1);
    [lk, fL] = feasible_initial(Vinitial(:,k),Hleft);
    [rk, fR] = feasible_initial(Uinitial(:,k),Hright);
    audit.init_fallback_left(k) = fL;
    audit.init_fallback_right(k) = fR;

    Yleft = left_projections(data,rk,h,w,n,usePageTimes);
    phiOld = sum(abs(lk' * Yleft));
    phiTrace = zeros(opts.maxIter+1,1);
    phiTrace(1) = phiOld;
    reason = 'maxIter';

    for t = 1:opts.maxIter
        % (1) Left: sign- at the CURRENT (l_k,r_k), fixed r_k.
        scoresLeft = lk' * Yleft;
        p = ones(1,n,'like',data);
        p(scoresLeft<=0) = -1; % sign_-(0) = -1, no random perturbation
        audit.zero_projections_left(k) = audit.zero_projections_left(k) + ...
            sum(scoresLeft==0);
        gLeft = Yleft * p.';
        nLeft = project_history(gLeft,Hleft);
        if norm(nLeft,2)>0
            lkNew = oriented_svd_direction(nLeft,Hleft);
        else
            lkNew = lk; % exact zero aggregate: preserve feasibility
            audit.zero_aggregates_left(k) = audit.zero_aggregates_left(k)+1;
        end

        % (2) Right: use NEW left direction; fixed old r_k for polarity.
        Yright = right_projections(Xflat,lkNew,w,n);
        scoresRight = rk' * Yright;
        s = ones(1,n,'like',data);
        s(scoresRight<=0) = -1;
        audit.zero_projections_right(k) = audit.zero_projections_right(k) + ...
            sum(scoresRight==0);
        gRight = Yright * s.';
        nRight = project_history(gRight,Hright);
        if norm(nRight,2)>0
            rkNew = oriented_svd_direction(nRight,Hright);
        else
            rkNew = rk; % exact zero aggregate: preserve feasibility
            audit.zero_aggregates_right(k) = audit.zero_aggregates_right(k)+1;
        end

        % (3) Only current-pair Phi_k is used to stop the inner alternation.
        phiNew = sum(abs(rkNew' * Yright));
        if ~isfinite(phiNew)
            error('L1RBPCA:NonFiniteObjective','Nonfinite Phi_k at k=%d,t=%d.',k,t);
        end
        numericalSlack = 1e-10 * max(1,abs(phiOld));
        if phiNew < phiOld-numericalSlack
            audit.monotone_violations(k) = audit.monotone_violations(k)+1;
        end
        delta = abs(phiNew-phiOld)/max(abs(phiOld),opts.eps0);
        phiTrace(t+1) = phiNew;
        lk = lkNew;
        rk = rkNew;
        phiOld = phiNew;
        audit.iterations(k) = t;
        if delta <= tol
            reason = 'relativePhi';
            break;
        end
        Yleft = left_projections(data,rk,h,w,n,usePageTimes);
    end

    L(:,k) = lk;
    R(:,k) = rk;
    audit.stop_reason{k} = reason;
    audit.phi_history{k} = phiTrace(1:audit.iterations(k)+1);
    audit.pair_objective(k) = phiOld;
    audit.orth_error_left(k) = norm(L(:,1:k)'*L(:,1:k)-eye(k),'fro');
    audit.orth_error_right(k)= norm(R(:,1:k)'*R(:,1:k)-eye(k),'fro');

    % Complete entrywise-L1 J(k), including BOTH sets of cross terms.
    YlastL = left_projections(data,rk,h,w,n,usePageTimes);
    YlastR = right_projections(Xflat,lk,w,n);
    increment = sum(abs(lk' * YlastL));
    cross = 0;
    if k>1
        A = Hleft' * YlastL;   % old left directions vs NEW right
        B = Hright' * YlastR;  % NEW left vs old right directions
        cross = sum(abs(A(:))) + sum(abs(B(:)));
    end
    audit.cross_increment(k) = cross;
    if k==1
        audit.full_objective(k) = increment;
    else
        audit.full_objective(k) = audit.full_objective(k-1)+increment+cross;
    end
end

model = mo;
model.V = L;  % left projection matrix, paper L
model.U = R;  % right projection matrix, paper R
model.algorithm_id = audit.algorithm_id;
audit.runtime_seconds = toc(clockStart);
model.audit = audit;
end

function Y = left_projections(X,r,h,w,n,usePageTimes)
% Columns of Y are X(:,:,i)*r. No per-component reuse of stale projections.
if usePageTimes
    Rpages = repmat(reshape(r,w,1),1,1,n);
    Y = reshape(pagemtimes(X,Rpages),h,n);
else
    Y = zeros(h,n,'like',X);
    for i=1:n
        Y(:,i) = X(:,:,i)*r;
    end
end
end

function Y = right_projections(Xflat,l,w,n)
% Columns of Y are X(:,:,i)'*l; Xflat=reshape(X,h,w*n).
Y = reshape(l'*Xflat,w,n);
end

function z = project_history(z,H)
% Two passes reduce loss of orthogonality in finite precision.
if ~isempty(H)
    z = z-H*(H'*z);
    z = z-H*(H'*z);
end
end

function [v,usedFallback] = feasible_initial(v0,H)
% The user supplies the initial vector. Only rank-deficient projections
% trigger a deterministic feasible canonical-basis fallback.
v = project_history(v0,H);
threshold = 100*eps(class(v))*max(1,norm(v0,2));
usedFallback = norm(v,2)<=threshold;
if usedFallback
    d = numel(v0);
    bestNorm = -Inf;
    best = zeros(d,1,'like',v0);
    for j=1:d
        ej = zeros(d,1,'like',v0);
        ej(j) = 1;
        candidate = project_history(ej,H);
        candidateNorm = norm(candidate,2);
        if candidateNorm>bestNorm
            bestNorm = candidateNorm;
            best = candidate;
        end
    end
    if bestNorm <= 100*eps(class(v0))
        error('L1RBPCA:NoFeasibleInit','No remaining feasible direction.');
    end
    v = best;
end
v = v/norm(v,2);
end

function v = oriented_svd_direction(g,H)
% For a NONZERO vector g, economy-SVD U(:,1)*V(1,1)=g/norm(g),
% subject to floating-point error. Reproject for numerical feasibility.
[U,~,V] = svd(g,'econ');
v = U(:,1)*V(1,1);
v = project_history(v,H);
nv = norm(v,2);
if ~(nv>0 && isfinite(nv))
    error('L1RBPCA:SVDFeasibility','Nonzero aggregate yielded invalid SVD direction.');
end
v = v/nv;
end
