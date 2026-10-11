function  [mo1]=TPCA_Greed_test(mo,data,q,tol)
[r,d,N] = size(data);maxIter = 10000;                     % 最大迭代次数，用来防止在极端情况下死循环
%%   Compute v while u is Fixed
qr=q(1);qc=q(2);
for j = 1:qr
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
        v = Xu*p'/norm(Xu*p',2);
        er = sum(abs(v'*Xu));
        if sum(p==0)>0
            detar = -1e-3 + (1e-3 + 1e-3) *rand(r, 1);
            v = (v + detar)/norm(v + detar);
            eoldr = er;
        end
        % 收敛条件满足，退出 while 循环
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
    mo1.V(:,j) = v;
    % data update
    Xu = Xu-mo1.V(:,j)*mo1.V(:,j)'*Xu;
end
% eval(['save Xu_' int2str(j) '.mat' ' Xu']);
%%  Compute u while v is Fixed
% iterCountc = 0;
for m=1:qc
    if m== 1
        C_arrayfun = arrayfun(@(i) (mo1.V(:,m)'*data(:,:,i))' , 1:size(data,3), 'UniformOutput', false);
        Cv = cat(3, C_arrayfun{:});
        Xv = reshape(Cv,[d,N]);
    end
    iterCountc =0;
    U0=mo.U(:,m);eoldc=sum(abs(U0'*Xv));
    while true
        iterCountc = iterCountc + 1;
        p = sign(U0'*Xv);
        u = Xv*p'/norm(Xv*p',2);
        ec = sum(abs(u'*Xv));
        if sum(p==0)>0
            detad = -1e-6 + (1e-3 + 1e-6) *rand(d, 1);
            u = (u + detad)/norm(u + detad);
            eoldc = ec;
        end
        % 收敛条件满足，退出 while 循环
        if (iterCountc > 1 && abs(ec - eoldc) / abs(ec) < tol)
            break;
        end
        % 防止死循环：若迭代次数超过最大限制则退出循环
        if iterCountc > maxIter
            % warning('Component %d: Maximum iterations reached without full convergence.', m);
            break;
        end
        U0=u; eoldc = ec;
    end
    mo1.U(:,m) =u;

    % data update
    Xv = (Xv'-Xv'*mo1.U(:,m)*mo1.U(:,m)')';
    % eval(['save Xv_' int2str(j) '.mat Xv']);
end
