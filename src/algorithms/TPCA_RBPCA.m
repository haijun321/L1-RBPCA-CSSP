function  [mo1]=TPCA_RBPCA(mo,data,q,tol)
%L1-RBPCA current snapshot; convergence is judged by the objective change
[r,d,N] = size(data);
qr=q(1);qc=q(2);maxIter = 10000;         % 最大迭代次数，用来防止在极端情况下死循环
for j = 1:qr
    % while true
    %%   Compute v while u is Fixed
    if j ==1
        C_arrayfun = arrayfun(@(i) data(:,:,i)*mo.U(:,j), 1:size(data,3), 'UniformOutput', false);
        Cu = cat(3, C_arrayfun{:});
        Xu = reshape(Cu,[r,N]);
    end
    V0 = mo.V(:, j);
    eoldr = sum(abs(V0'*Xu));
    iterCountr = 0;
    while true
        iterCountr = iterCountr + 1;
        p = sign(V0'*Xu);
        % p(p==0) = -1;
        Mu = Xu*p';
        [L,~,R] = svd(Mu);
        I1 = eye(1);I2 = zeros(r-1,1);I =[I1;I2];
        v = L*I*R';
        % v=v/norm(v,2);
        er = sum(abs(v'*Xu));
        if (iterCountr > 1 && abs(er - eoldr) / abs(er) < tol)
            break;
        end
        % 防止死循环：若迭代次数超过最大限制则退出循环
        if iterCountr > maxIter
            % warning('Component %d: Maximum iterations reached without full convergence.', j);
            break;
        end
        V0=v;eoldr = er;
    end
    % data update
    mo1.V(:,j) = v;
    Xu = Xu-mo1.V(:,j)*mo1.V(:,j)'*Xu;
    % eval(['load Xu_' int2str(j) '.mat;']);
end
%%  Compute u while v is Fixed
for m=1:qc
    if  m== 1
        C_arrayfun = arrayfun(@(i) (mo1.V(:,m)'*data(:,:,i))' , 1:size(data,3), 'UniformOutput', false);
        Cv = cat(3, C_arrayfun{:});
        Xv = reshape(Cv,[d,N]);
    end
    iterCountc =0;
    U0=mo.U(:,m);eoldc=sum(abs(U0'*Xv));
    while true
        iterCountc = iterCountc + 1;
        p = sign(Xv'*U0);
        % p(p==0) = -1;
        Mv = Xv*p;
        [L,~,R] = svd(Mv);
        I1 = eye(1);I2 = zeros(d-1,1);I =[I1;I2];
        u = L*I*R';
        % u=u/norm(u,2);
        ec = sum(abs(u'*Xv));
        if (iterCountc > 1 && abs(ec - eoldc) / abs(ec) < tol)
            break;
        end
        % 防止死循环：若迭代次数超过最大限制则退出循环
        if iterCountc > maxIter
            % warning('Component %d: Maximum iterations reached without full convergence.', m);
            break;
        end
        U0=u;eoldc = ec;
    end
    % data update
    mo1.U(:,m) =u;
    % Xv = Xv-mo.U(:,j)*mo.U(:,j)'*Xv;
    Xv = (Xv'-Xv'*mo1.U(:,m)*mo1.U(:,m)')';
end
end