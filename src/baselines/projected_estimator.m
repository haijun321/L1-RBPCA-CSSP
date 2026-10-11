function [R_tilde, C_tilde] = projected_estimator(mo,X, k1, k2)
    % 输入参数：
    % X        : 输入数据矩阵，大小为 T x p1 x p2（时间序列矩阵）
    % k1, k2   : 因子数，分别用于行和列方向的降维
    %
    % 输出参数：
    % R_tilde  : 行方向的载荷矩阵，大小为 p1 x k1
    % C_tilde  : 列方向的载荷矩阵，大小为 p2 x k2
    
    [ p1, p2,T] = size(X); % 获取输入矩阵维度
    
    %% Step 1: 初始估计器的获取 (R_hat 和 C_hat)
    % 使用随机正交矩阵作为初始估计器
    R_hat = mo.V; % 行方向初始估计
    C_hat = mo.U; % 列方向初始估计

    %% Step 2: 数据投影到低维空间
    % 定义 Y_t 和 Z_t
    Y_t = zeros( p1, k2,T); % T 个投影到列方向的矩阵
    Z_t = zeros( p2, k1,T); % T 个投影到行方向的矩阵
    for t = 1:T
        Xt = reshape(X(:, :, t), [p1, p2]); % 当前时刻的矩阵
        Y_t(:, :, t) = Xt * C_hat; % 列方向投影
        Z_t(:, :, t) = Xt' * R_hat; % 行方向投影
    end

    %% Step 3: 计算协方差矩阵 M1 和 M2
    % 初始化
    M1 = zeros(p1, p1); % 行方向协方差矩阵
    M2 = zeros(p2, p2); % 列方向协方差矩阵
    for t = 1:T
        Yt = reshape(Y_t( :, :,t), [p1, k2]);
        Zt = reshape(Z_t( :, :,t), [p2, k1]);
        M1 = M1 + Yt * Yt'; % 行方向协方差累积
        M2 = M2 + Zt * Zt'; % 列方向协方差累积
    end
    M1 = M1 / T; % 归一化
    M2 = M2 / T; % 归一化

    %% Step 4: 求解主方向 Q1 和 Q2
    % 使用 eigs 计算前 k1 和 k2 个特征向量
    [Q1, ~] = eigs(M1, k1); % 取 M1 的前 k1 个特征向量
    [Q2, ~] = eigs(M2, k2); % 取 M2 的前 k2 个特征向量

    %% Step 5: 计算最终的 R_tilde 和 C_tilde
    % 修改输出，使其维度为 p1 x k1 和 p2 x k2
    R_tilde = Q1; % 行方向的载荷矩阵，大小为 p1 x k1
    C_tilde = Q2; % 列方向的载荷矩阵，大小为 p2 x k2
end
