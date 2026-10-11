function [mo1] = Non_Greedy(mo, data, q, tol)
% Non_Greedy 该算法是用目标函数作为收敛性判断的条件，用于非贪婪算法（L1范数最大化）
%   输入：
%       mo      -- 一个结构体，至少包含 U 和 V 两个矩阵字段：mo.U, mo.V
%       data    -- 尺寸为 (r x d x N) 的三维数据张量
%       q       -- 提取的成分数
%       tol     -- 收敛阈值
%   输出：
%       mo1     -- 更新后的结构体，包含新的 U, V
% 初始化
[r, d, N] = size(data);
maxIter   = 100;           % 防止死循环的最大迭代次数
mo1       = mo;            % 拷贝原始 mo 作为输出的基准
% 迭代计数
iterCount = 0;
%----------------------
%  外层循环：交替更新U、V
%----------------------
while true
    iterCount = iterCount + 1;
    %=========================
    %     1) 固定U, 更新V
    %=========================
    % Y1 (r x qN)：将数据在列维度乘以 mo1.U
    %   data(:,:,i) * mo1.U => (r x d)*(d x q) => (r x q)
    %   然后将 N 个 (r x q) 拼成 (r x q*N)
    C_arrayfun = arrayfun(@(i) data(:,:,i)* mo1.U, ...
        1:N, 'UniformOutput', false);
    Y1 = reshape(cat(3, C_arrayfun{:}), [r,q*N]);
    % 记录旧目标值 eoldr，用于收敛判断
    eoldr = sum(sum(abs(mo1.V'* Y1)));
    iterCountr = 0;
    while true
        iterCountr = iterCountr + 1;
        % 计算符号矩阵 P (q x qN)
        P0 = sign(mo1.V' * Y1);
        P  = reshape(P0, [q, q*N]);
        % M = Y1 * P' => (r x qN)*(qN x q) = (r x q)
        M = Y1 * P';
        % SVD 分解更新: M = L * S * R'
        [L, ~, R] = svd(M, 'econ');
        % 得到新的 v (r x q) => (r x q)
        v  = L * R';
        % 计算新目标值 er
        er = sum(sum(abs(v'*Y1)));
        %---------------------
        %   收敛判断 (V 部分)
        %---------------------
        if (iterCountr >1 && abs(er - eoldr)/abs(er) < tol)
            break;
        end
        if iterCountr > maxIter
            % warning('V 更新：超过最大迭代次数，提前退出。')
            break;
        end
        % 更新 V
        mo1.V = v;
        % 记录 eoldr 以备下一次循环
        eoldr = er;
    end  % while 更新 V
    %=========================
    %     2) 固定V, 更新U
    %=========================
    % Y2 (d x qN)：将数据在行维度乘以 mo1.V'
    %   mo1.V' * data(:,:,i) => (q x r)*(r x d) => (q x d)，再转置 (d x q)
    %   拼成 (d x q*N)
    C_arrayfun = arrayfun(@(i) (mo1.V' * data(:,:,i))', ...
        1:N, 'UniformOutput', false);
    Y2 = reshape(cat(3, C_arrayfun{:}), [d, q*N]);
    % 记录旧目标值 eoldc，用于收敛判断
    eoldc = sum(sum(abs(mo1.U' * Y2)));
    iterCountc = 0;
    while true
        iterCountc = iterCountc + 1;
        % 计算符号矩阵 S (q x qN)
        S0 = sign(mo1.U'*Y2);
        S  = reshape(S0, [q, q*N]);
        % M = Y2 * S' => (d x qN)*(qN x q) = (d x q)
        M = Y2 * S';
        % SVD 分解更新: M = L * ~ * R'
        [L, ~, R] = svd(M, 'econ');
        % 得到新的 u (d x q)
        u  = L*R';
        % 计算新目标值 ec (注意这里用 Y2 而非 Y1)
        ec = sum(sum(abs(u' * Y2)));
        %---------------------
        %   收敛判断 (U 部分)
        %---------------------
        if (iterCountc > 1 && abs(ec-eoldc)/abs(ec) < tol)
            break;
        end
        if iterCountc > maxIter
            % warning('U 更新：超过最大迭代次数，提前退出。')
            break;
        end
        % 更新 U
        mo1.U = u;
        % 记录 eoldc 以备下一次循环
        eoldc = ec;
    end  % while 更新 U
    %==============================
    % 3) 外层收敛检查(可选/增强功能)
    %    判断总体目标函数是否变化很小
    %==============================
     % 用更新后的 mo1.V, mo1.U 重新计算目标函数:
    outerObjective = 0;
    for i = 1:N
        Xi = data(:,:,i);
        % mo1.V' * Xi * mo1.U => (q x r)*(r x d)*(d x q) => (q x q)
        % 取绝对值求 L1 范数
        outerObjective = outerObjective + sum(sum(abs(mo1.V' * Xi * mo1.U)));
    end
    eNow = outerObjective;   % 本次外层目标值
    if iterCount == 1
        eOld = eNow;
        continue;   % 直接开始下一轮
    end
    % 收敛判定
    if abs(eNow - eOld)/abs(eNow) < tol
        break;  % 满足收敛阈值，退出外层 while
    end
    if iterCount >= maxIter
        % warning('外层：迭代达到最大次数，退出。');
        break;
    end
    % 记录本轮外层目标值
    eOld = eNow;
end  % while (交替更新)
end  % while 交替更新U、Vend
