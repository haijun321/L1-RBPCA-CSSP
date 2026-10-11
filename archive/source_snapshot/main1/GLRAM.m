function [bp, options] = GLRAM(mo,data, M, Q, options)
% 目标维度行列方向上目标维度不同
% data:    3D数组，维度为 (dr x dc x N)，包含 N 个大小为 (dr x dc) 的矩阵
% M:       行方向的目标降维维度
% Q:       列方向的目标降维维度
% options: 控制迭代、显示等的参数 (与原代码类似)

tic;  % 计时开始

% 1. 从 options 中获取迭代次数 niters
if (options(14))
    niters = options(14);
else
    niters = 100;
end

display = options(1);
store   = 0;       % 是否记录误差
if (nargout > 1)
    store = 1;     % 若需要输出 bp 结构，存储迭代误差
    bp.errlog = zeros(1, niters);
end

test = 0;          % 用于判断是否提前停止
if options(3) > 0.0
    test = 1;      % 当误差变化小于某阈值时提前停止
end

% 2. 解析 data 的维度 (dr x dc x N)
[dr, dc, N] = size(data);

% 3. 随机初始化投影矩阵
%    V: (dr x M)   —— 行方向降维
%    U: (dc x Q)   —— 列方向降维
V = mo.V;
U = mo.U;

% 4. 进入迭代
for n = 1:niters
    
    % ---- (a) 固定 V，更新 U ----
    %
    % 思想：对每个样本矩阵 X_i (dr x dc)，先左乘 V' (M x dr) 得到 (M x dc)，
    % 再根据这些结果，求能最好地右乘得到 (M x Q) 的 U (dc x Q)。
    
    % 先把 data reshape 成适合与 V' 相乘的形状:
    % data(:,:) 在 MATLAB 中会把 (dr, dc, N) 展平为 (dr, dc*N)
    yx = data(:,:);       % 形状: (dr) x (dc*N)
    
    % V' * yx => (M x dr) * (dr x dc*N) => (M x dc*N)
    tmp1 = V' * yx;       
    
    % 重塑回三维 (M, dc, N)
    y1 = reshape(tmp1, [M, dc, N]);
    
    % permute 到 (dc, M, N)，便于后续处理
    y2 = permute(y1, [2, 1, 3]);
    
    % 再把 (dc, M, N) 打平到 (dc, M*N)
    yy = y2(:,:);
    
    % 构造一个 (dc x dc) 矩阵用于特征分解
    YU = yy * yy';    % (dc x dc)
    
    % 取前 Q 个特征向量，得到新的 U: (dc x Q)
    [Utemp, ~] = eigs(YU, Q);
    U = Utemp;
    
    % ---- (b) 固定 U，更新 V ----
    %
    % 思想：对每个样本矩阵 X_i (dr x dc)，先右乘 U (dc x Q) 得到 (dr x Q)，
    % 再根据这些结果，找出能最好地把 dr 降到 M 的 V (dr x M)。
    
    % 把 data 从 (dr, dc, N) permute 到 (dc, dr, N)，然后再打平
    yz  = permute(data, [2, 1, 3]);   % (dc, dr, N)
    yzz = yz(:,:);                    % (dc, dr*N)
    
    % U' (Q x dc) 与 yzz (dc x (dr*N)) => (Q x dr*N)
    tmp2 = U' * yzz;
    
    % reshape 到 (Q, dr, N)
    y3 = reshape(tmp2, [Q, dr, N]);
    
    % permute 到 (dr, Q, N)
    y4 = permute(y3, [2, 1, 3]);
    
    % 打平成 (dr, Q*N)
    yy2 = y4(:,:);
    
    % 构造 (dr x dr) 矩阵，用于更新 V
    YV = yy2 * yy2';  % (dr x dr)
    
    % 取前 M 个特征向量，得到新的 V: (dr x M)
    [Vtemp, ~] = eigs(YV, M);
    V = Vtemp;
    
    % 把当前 U、V 记录在 bp 里
    bp.U = U;
    bp.V = V;
    
    % ---- (c) 计算误差 / 检查收敛 ----
    if (display || store || test)
        % 假设有一个 Objective 函数来计算在 (U,V) 下的重构误差
        e = Objective(data, bp);
        
        if store
            bp.errlog(n) = e;
        end
        
        if (display > 0)
            if n == 1
                fprintf('Iteration %4d,  Error = %.6f\n', n, e);
            else
                fprintf('Iteration %4d,  Error = %.6f,  RelChange = %.6e\n', ...
                        n, e, (e - eold)/max(abs(eold),1e-12));
            end
        end
        
        % 若设置了 test (options(3) > 0), 检查相对误差变化以提前停止
        if test
            if (n > 1) && (abs(e - eold)/max(abs(e),1e-12) < options(3))
                options(8) = e;
                bp.errlog = bp.errlog(1:n);
                bp.time = toc;
                return;
            else
                eold = e;
            end
        else
            eold = e;
        end
    end
end

% 若迭代完还没退出，则记录最终结果
bp.errlog = bp.errlog(1:niters);
bp.time   = toc;

end
