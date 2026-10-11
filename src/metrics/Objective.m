function[OB] = Objective(X,bp)
[dc,dr,N] = size(X);XX = X(:,:);
[~,q(1)] = size(bp.V);[~,q(2)] = size(bp.U);
OB = sum(sum(abs(reshape(permute(reshape(bp.V'*XX,[q(1),dr,N]),[1,3,2]),[q(1)*N,dr])*bp.U)));
end