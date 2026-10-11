function reconstructed = reconstruct(X, V, U)
    % 重构函数
    % 输入:
    %   X - 原始数据矩阵 (dr x dc x N)
    %   V - 列空间投影矩阵 (dr x k)
    %   U - 行空间投影矩阵 (dc x k)
    % 输出:
    %   reconstructed - 重构后的数据矩阵 (dr x dc x N)
    
    % 数据维度
    [dr, dc, N] = size(X);
    
    % 初始化重构矩阵
    reconstructed = zeros(dr, dc, N);
    
    % 对每个样本进行重构
    for i = 1:N
        X_i = X(:, :, i);          % 提取单张图像
        % 投影到低维空间并重构
        X_low = V' * X_i * U;      % 低维表示
        reconstructed(:, :, i) = V * X_low * U'; % 重构
    end
end
