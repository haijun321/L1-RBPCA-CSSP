clear all; close all; clc;
%此版本实验结果显示几种异常值类型下，不同异常比例下，重构误差随目标维度变化的结果
% 初始化变量
addpath('C:\Users\蒋海俊\Desktop\DC\DOC\BRPCA代码\data\MTS\auslan');
projectRoot = 'C:\Users\蒋海俊\Desktop\DC\DOC\20250401';
p1 = fullfile(projectRoot, 'main','PK');
p3 = fullfile(projectRoot, 'main');
addpath(p1, p3);


dataname = {'auslan','trx','wafer','YALE', 'ORL','PIE', 'AR', 'xm2vts'}; % 数据集
outlier_ratios = [0,0.2, 0.4, 0.6, 0.8, 1]; % 异常值比例
outlier_size = [3, 50]; % 噪声矩形大小范围
algorithms = {'NewGreedy', 'BPCA', 'GLRAM', 'PE'};
options(1) = 0; options(3) = 1e-8; options(14) = 4; options(15) = 0;
anomaly_types = {'uniform', 'gaussian', 'impulse', 'sinusoidal', 'constant', 'salt_and_pepper', 'poisson', 'speckle', 'log_normal', 'exponential'}; % 所有异常类型

% 存储重构误差
ALL_RECONSTRUCTION_ERRORS_test = cell(length(dataname), length(outlier_ratios), length(anomaly_types));
ALL_RECONSTRUCTION_ERRORS_train = cell(length(dataname), length(outlier_ratios), length(anomaly_types));

% 数据集加载和预处理
for dataset_idx = 1:1%length(dataname)
    selected_dataset = dataname{dataset_idx}
    switch selected_dataset
        case 'YALE'
            load('YALE64.mat'); load('YALElabel.mat'); labels = label; q = 60;
        case 'ORL'
            load('ORL.mat'); load('ORLlabel.mat'); labels = label; q = 60;
        case 'PIE'
            load('data_norm.mat'); load('label.mat'); labels = label; q = 60;
        case 'AR'
            load('AR.mat'); load('ARlabel.mat'); labels = label; q = 60;
        case 'xm2vts'
            load('xm2vts.mat'); load('xm2vtslabel.mat'); labels = label; q = 50;
        case 'auslan'
            load('auslan.mat'); labels = label_org; q = 22;
        case 'trx'
            load('auslan.mat'); labels = label_org; q = 10;
        case 'ecg'
            load('ecg.mat'); labels = label_org; q = 2;
        case 'wafer'
            load('wafer.mat'); labels = label_org; q = 10;
    end
    NX_org = X_org;
    [dr, dc, N] = size(NX_org); % 数据维度和样本数
    unique_labels = unique(labels); % 唯一标签数
    num_classes = length(unique_labels); % 类别数

    % 划分训练集和测试集（每个类别取0.85对应样本作训练）
    num_train = floor(0.85 * N / num_classes); % 每类训练样本数量
    num_test = N / num_classes - num_train; % 每类测试样本数量

    train_indices = []; test_indices = [];
    for class_idx = 1:num_classes
        class_samples = find(labels == unique_labels(class_idx));
        shuffled_indices = class_samples(randperm(length(class_samples)));
        train_indices = [train_indices; shuffled_indices(1:num_train)];
        test_indices = [test_indices; shuffled_indices(num_train + 1:end)];
    end

    % 训练集和测试集划分
    NX_train = NX_org(:, :, train_indices);
    NX_test = NX_org(:, :, test_indices);
    train_labels = labels(train_indices);
    test_labels = labels(test_indices);

    %% 添加异常值并进行实验
    for ratio_idx = 1:length(outlier_ratios)
        outlier_ratio = outlier_ratios(ratio_idx)
        num_outliers = round(outlier_ratio * num_train * num_classes);
        outlier_indices = randsample(num_train * num_classes, num_outliers);

        NX_train_outlier = NX_train;
        %  添加异常值
        % 循环所有异常类型
        for anomaly_type_idx = 1:length(anomaly_types)
            anomaly_type = anomaly_types{anomaly_type_idx}    % 获取当前的异常类型
            
            % 循环每个outlier_indices样本
            for idx = outlier_indices'
                img = NX_train(:, :, idx);

                % 为图像添加异常值：生成不同类型的异常
                variable_idx = randi(dr);
                start_time = randi([1, dc - 20 + 1]);
                anomaly_height = randi(outlier_size);  % 随机生成异常块的高度
                anomaly_width = randi(outlier_size);   % 随机生成异常块的宽度

                % 确保异常块不超出图像边界
                if variable_idx + anomaly_height - 1 > dr
                    anomaly_height = dr - variable_idx + 1;
                end
                if start_time + anomaly_width - 1 > dc
                    anomaly_width = dc - start_time + 1;
                end

                original_patch = img(variable_idx:variable_idx + anomaly_height - 1, start_time:start_time + anomaly_width - 1);
               %% 人脸数据加入异常的有用形式类似于下述
             
                % 根据异常类型添加不同的异常
                switch anomaly_type
                    case 'uniform'  %产生的异常没有任何模式和结构，通常看起来像一个“块”，其值在一个范围内均匀分布。
                        anomaly_values = 100 * rand(anomaly_height, anomaly_width);
                    case 'gaussian' %异常的像素值在正常情况下应该呈现一个连续的变化模式，较小的值偏离中心的概率较低。与均匀分布相比，高斯噪声的异常值更集中且呈现出一定的结构。
                        mu = 50; sigma = 15;
                        anomaly_values = mu + sigma * randn(anomaly_height, anomaly_width);
                        anomaly_values = max(0, min(100, anomaly_values));
                    case 'impulse'   %这种异常非常突兀，通常是由电气问题、信号干扰等原因引起的。图像上通常会看到黑白相间的“点”。
                        density = 0.1;
                        noise_mask = rand(anomaly_height, anomaly_width) < density;
                        salt = rand(anomaly_height, anomaly_width) < 0.5;
                        anomaly_values = original_patch;
                        anomaly_values(noise_mask & salt) = 100;
                        anomaly_values(noise_mask & ~salt) = 0;
                    case 'sinusoidal'  %生成的异常值呈现周期性的波动，这种异常类似于周期性的干扰或扰动，常见于通信信号中的周期性噪声。
                        [X, ~] = meshgrid(1:anomaly_width, 1:anomaly_height);
                        frequency = 0.1;
                        amplitude = 40;
                        anomaly_values = 50 + amplitude * sin(2*pi*frequency*X);
                        anomaly_values = max(0, min(100, anomaly_values));
                    case 'constant'   %异常区域内所有像素值相同，通常看起来像一个“方块”或者“区域”，这类异常没有任何变化或随机性。
                        constant_value = 100;
                        anomaly_values = constant_value * ones(anomaly_height, anomaly_width);
                    case 'salt_and_pepper'   %它在图像中表现为非常随机的黑白点，给图像添加杂乱无章的干扰。
                        density = 0.1;
                        anomaly_values = original_patch;
                        anomaly_values(rand(anomaly_height, anomaly_width) < density) = randi([0, 1]) * 100;
                    case 'poisson'     %生成的异常值会根据泊松分布产生，通常是随机的，但在特定场景下可能会集中或呈现出某些统计特性
                        anomaly_values = poissrnd(10, anomaly_height, anomaly_width);
                    case 'speckle'     %产生的异常通常表现为局部区域的细小颗粒或斑点，通常是由传感器噪声引起的。
                        noise = randn(anomaly_height, anomaly_width);
                        anomaly_values = original_patch .* (1 + 0.1 * noise);
                    case 'log_normal'   %这种噪声通常会导致较小的异常值集中在低范围，而较大的异常值则更为罕见。它通常在大部分正常值较小的情况下对异常值产生较大影响。
                        mu = 0; sigma = 0.5;
                        anomaly_values = exp(mu + sigma * randn(anomaly_height, anomaly_width));
                        anomaly_values = max(0, min(100, anomaly_values));
                    case 'exponential'   %指数分布具有右偏特性，产生较小的异常值较为常见，较大的异常值则较为罕见。
                        anomaly_values = exprnd(10, anomaly_height, anomaly_width);
                        anomaly_values = max(0, min(100, anomaly_values));
                    case 'missing'  % 数据缺失，模拟传感器数据丢失或遮挡
                        anomaly_values = NaN(anomaly_height, anomaly_width);
                    otherwise
                        error('未知异常类型');
                end

                % 将生成的异常块加入图像的指定位置
                img(variable_idx:variable_idx + anomaly_height - 1, start_time:start_time + anomaly_width - 1) = anomaly_values;

                % 更新训练集
                NX_train_outlier(:, :, idx) = img;
            end

            %% 计算不同算法和不同维度的重构误差
            for k = 1:q
                for alg_idx = 1:length(algorithms)
                    algorithm = algorithms{alg_idx};
                      mo1.U = orth(rand(dc, k));
                      mo1.V = orth(rand(dr, k));
                    switch algorithm
                        case 'NewGreedy'
                            dim=[k,k];
                            result = TPCA_RBPCA(mo1, NX_train_outlier, dim,1e-6);
                        case 'BPCA'
                            [R, C] = BPCA(NX_train_outlier, k, k);
                            result.U = C; result.V = R;
                        case 'GLRAM'
                            [bp, ~] = GLRAM(mo1,NX_train_outlier,k,k, options);
                            result.U = bp.U; result.V = bp.V;
                        case 'PE'
                            [R, C] = projected_estimator(mo1,NX_train_outlier, k, k);
                            result.U = C; result.V = R;
                        otherwise
                            error('未知算法');
                    end
                    %  重构训练集并计算误差
                    reconstructed_train = reconstruct(NX_train_outlier, result.V, result.U);
                    reconstruction_errors_train(alg_idx, k) = mean(arrayfun(@(i) norm(reconstructed_train(:, :, i) - NX_train_outlier(:, :, i), 2), 1:num_train));

                    %  重构测试集并计算误差
                    reconstructed_test = reconstruct(NX_test, result.V, result.U);
                    reconstruction_errors_test(alg_idx, k) = mean(arrayfun(@(i) norm(reconstructed_test(:, :, i) - NX_test(:, :, i), 2), 1:num_test));
                end
            end

            % 为不同异常类型存储重构误差
            ALL_RECONSTRUCTION_ERRORS_test{dataset_idx, ratio_idx, anomaly_type_idx} = reconstruction_errors_test;
            ALL_RECONSTRUCTION_ERRORS_train{dataset_idx, ratio_idx, anomaly_type_idx} = reconstruction_errors_train;
        end
    end
end

%% 可视化并保存结果
%% 可视化并保存结果
save_root = fullfile('C:\Users\蒋海俊\Desktop\DC\DOC\20250401\Reconstruction\result\difoutliers_case');  % 统一保存路径
if ~exist(save_root, 'dir')
    mkdir(save_root);
end

for dataset_idx = 1:1 %length(dataname)
    selected_dataset = dataname{dataset_idx};
    dataset_folder = fullfile(save_root, selected_dataset);
    if ~exist(dataset_folder, 'dir')
        mkdir(dataset_folder);
    end

    for ratio_idx = 1:length(outlier_ratios)
        ratio_str = ['OutlierRatio_', num2str(outlier_ratios(ratio_idx))];
        ratio_folder = fullfile(dataset_folder, ratio_str);
        if ~exist(ratio_folder, 'dir')
            mkdir(ratio_folder);
        end

        fig = figure('Units','normalized','OuterPosition',[0 0 1 1]);  % 全屏显示
        for anomaly_type_idx = 1:length(anomaly_types)
            subplot(2, 5, anomaly_type_idx);
            hold on;
            errors_for_ratio = ALL_RECONSTRUCTION_ERRORS_test{dataset_idx, ratio_idx, anomaly_type_idx};

            for alg_idx = 1:length(algorithms)
                errors_for_algorithm = errors_for_ratio(alg_idx, :);
                plot(1:q, errors_for_algorithm, '-o', 'LineWidth', 1.5, 'DisplayName', algorithms{alg_idx});
            end

            xlabel('Feature Dimension (q)');
            ylabel('Reconstruction Error (Frobenius Norm)');
            title(['Anomaly: ', anomaly_types{anomaly_type_idx}]);
            legend('Location', 'Best');
            grid on;
            hold off;
        end

        % 保存图像为 PNG 和 FIG
        save_name_base = fullfile(ratio_folder, ['ReconstructionError_', ratio_str]);
        saveas(fig, [save_name_base, '.png']);
        savefig(fig, [save_name_base, '.fig']);
        close(fig);  % 关闭图形以节省内存
    end
end
