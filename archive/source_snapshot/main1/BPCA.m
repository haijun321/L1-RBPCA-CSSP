function [R, C] = BPCA(data, k1, k2)
% BPCA - 双边主成分分析 (Bi-Directional PCA)
% 输入:
%   data: 输入数据矩阵集合 (3D数组)，大小为 [m, n, N]，每个样本是 m×n 矩阵
%   k1: 行方向降维目标维度
%   k2: 列方向降维目标维度
% 输出:
%   R: 行方向投影矩阵 (m × k1)
%   C: 列方向投影矩阵 (n × k2)

[m, n, N] = size(data); % 获取数据维度

% 1. 数据中心化
% mean_data = mean(data, 3); % 计算所有样本的均值
% for i = 1:N
%     data(:, :, i) = data(:, :, i) - mean_data; % 每个样本减去均值
% end

% 2. 计算协方差矩阵
% 行方向协方差矩阵
C_r = reshape(data, m, n * N) * reshape(data, m, n * N)' / N;

% 列方向协方差矩阵
C_c = reshape(permute(data, [2, 1, 3]), n, m * N) * reshape(permute(data, [2, 1, 3]), n, m * N)' / N;

% 3. 特征分解或 SVD
% 行方向 SVD
[U_r, ~, ~] = svd(C_r, 'econ');
R = U_r(:, 1:k1); % 取前 k1 个奇异向量

% 列方向 SVD
[U_c, ~, ~] = svd(C_c, 'econ');
C = U_c(:, 1:k2); % 取前 k2 个奇异向量

end
