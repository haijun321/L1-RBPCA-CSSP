function NX_train_outlier = add_outliers(NX_train, outlier_indices, anomaly_type, outlier_size, anomaly_length)
% 添加异常值函数
% 
% INPUTS:
%   NX_train        - 原始三维数据，大小 dr×dc×N
%   outlier_indices - 需要加异常的样本索引向量
%   anomaly_type    - 异常类型，支持 'salt_and_pepper'、'impulse'、'uniform'
%   outlier_size    - 异常块的边长（pixels 或 time steps）
%   anomaly_length  - （仅对 'uniform' 有效）均匀分布异常块的数值范围上界
%
% OUTPUT:
%   NX_train_outlier - 添加异常后的数据

[dr, dc, N] = size(NX_train);
NX_train_outlier = NX_train;

for idx = outlier_indices'
    img = NX_train(:, :, idx);
    % 随机选取块的起始位置
    row0 = randi(dr);
    col0 = randi(dc);

    % 确保不越界
    h = min(outlier_size, dr - row0 + 1);
    w = min(outlier_size, dc - col0 + 1);

    original_patch = img(row0:row0 + h - 1, col0:col0 + w - 1);

    switch anomaly_type
        case 'salt_and_pepper'
            density = 0.8;
            anomaly_vals = original_patch;
            mask = rand(h, w) < density;
            anomaly_vals(mask) = randi([0, 1], sum(mask(:)), 1) * 500;% 原来是100，PEMS数据集做修改

        case 'impulse'
            density = 0.8;
            mask = rand(h, w) < density;
            salt = rand(h, w) < 0.5;
            anomaly_vals = original_patch;
            anomaly_vals(mask & salt)   = 100;
            anomaly_vals(mask & ~salt) = 0;

        case 'uniform'
            % 在 [0, anomaly_length] 上做均匀采样
            anomaly_vals = rand(h, w) * anomaly_length;

        otherwise
            error('未知异常类型：%s', anomaly_type);
    end

    % 把生成的异常块贴回去
    img(row0:row0 + h - 1, col0:col0 + w - 1) = anomaly_vals;
    NX_train_outlier(:, :, idx) = img;
end
end
