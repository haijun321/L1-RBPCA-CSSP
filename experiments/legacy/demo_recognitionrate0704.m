clear all; close all; clc;
projectRoot = 'H:\Students\Jiang_haijun\202506';
p1 = fullfile(projectRoot, 'main', 'PK');
p3 = fullfile(projectRoot, 'data');
p6 = fullfile(projectRoot, 'data','PEMS');
% p1 = fullfile(projectRoot, 'Realdata', 'main');
% p2 = fullfile(projectRoot, 'Realdata', 'data');
p3 = fullfile(projectRoot, 'main','PK');
% p6 = fullfile(projectRoot, 'Realdata', 'data','KickvsPunch');
% p7 = fullfile(projectRoot, 'Realdata', 'data','walkVsRun');
addpath('H:\Students\Jiang_haijun\RB2DLDA(11.29)\egc');
addpath('H:\Students\Jiang_haijun\RB2DLDA(11.29)\wafer');
addpath('H:\Students\Jiang_haijun\B2DLDA1111\bci')
p4 = fullfile(projectRoot, 'main');
p5 = fullfile(projectRoot, 'Reconstruction');
addpath('H:\Students\Jiang_haijun\202506\data\PEMS');
addpath(p1,  p3, p4, p5);  % 设置输出目录 & 日志文件

% 数据集名称
dataname = {'bci','walkvsRun','PEMS','wafer','ecg','japan','auslan','AR','XM2VTS', 'ORL','YALE','trx', 'PIE', 'xm2vts'};
% outlier_ratios = [0.8];  % 异常值比例

outlier_ratios = [0,0.2,0.4,0.6,0.8];  % 异常值比例
% outlier_size = [55];  % 噪声矩形大小
outlier_size = [5,25];  % 噪声矩形大小

algorithms = {'NewGreedy', 'TPCA-L1-Greedy','BPCA', 'GLRAM', 'PE','NonGreedy'};
options(1) = 0; options(3) = 1e-8; options(14) = 4; options(15) = 0;
% anomaly_types = {'salt_and_pepper', 'impulse'};  % 只保留这两种异常类型
anomaly_types = {'impulse'};  % 只保留这两种异常类型

% anomaly_types = {'uniform', 'impulse'};  % 只保留这两种异常类型
% 存储识别率
ALL_RECOGNITION_ACCURACIES_test = cell(length(dataname), length(outlier_ratios), length(anomaly_types), length(outlier_size));
ALL_RECOGNITION_ACCURACIES_train = cell(length(dataname), length(outlier_ratios), length(anomaly_types), length(outlier_size));

% 数据集加载和预处理
for dataset_idx =3:3%length(dataname)
    selected_dataset = dataname{dataset_idx}
    switch selected_dataset
        case 'bci'
        load('bci.mat'); labels=label; q=28;X_org = datatruncated;
        case 'japan'
            load('japan.mat'); labels = label_org; q = 7;
        case 'YALE'
            load('yaledata_norm_64x64.mat'); load('yalelabel.mat'); labels = label; q = 60;
        case 'ORL'
            load('data_norm.mat'); load('label.mat'); labels = label; q = 70;
        case 'PIE'
            load('data_norm.mat'); load('label.mat'); labels = label; q = 60;
        case 'AR'
            load('AR.mat'); load('ARlabel.mat'); labels = label; q = 60;
        case 'XM2VTS'
            load('xm2data_norm.mat'); load('xm2label.mat'); labels = label; q = 50;
        case 'auslan'
            load('auslan.mat'); labels = label_org; q = 22;
        case 'trx'
            load('trx.mat'); labels = label_org; q = 10;
        case 'ecg'
            load('ecg.mat'); labels = label_org; q = 2;
        case 'wafer'
            load('wafer.mat'); labels = label_org; q = 6;
        case 'walkvsRun'
            load('WalkvsRunall.mat'); X_org = datatruncated; labels = label; q = 60;
        case 'PEMS'
            load('PEMSsorttrans.mat'); X_org = datatruncated; labels = label; q = 66;
    end

    NX_org = X_org;
    [dr, dc, N] = size(NX_org);  % 数据维度和样本数
    unique_labels = unique(labels); % 唯一标签数
    num_classes = length(unique_labels); % 类别数

    % 划分训练集和测试集（每个类别取0.85对应样本作训练）
    train_indices_cell = cell(num_classes, 1); % 存储每个类别的训练集索引
    test_indices_cell = cell(num_classes, 1); % 存储每个类别的测试集索引

    for class_idx = 1:num_classes
        class_samples = find(labels == unique_labels(class_idx));
        num_class_samples = length(class_samples);
        num_train = floor(0.85 * num_class_samples);  % 85%作为训练集
        num_test = num_class_samples - num_train;    % 剩余作为测试集
        shuffled_indices = class_samples(randperm(num_class_samples));  % 随机打乱样本
        train_indices_cell{class_idx} = shuffled_indices(1:num_train);  % 存储每个类的训练集索引
        test_indices_cell{class_idx} = shuffled_indices(num_train + 1:end);  % 存储每个类的测试集索引
    end

    % 合并训练集和测试集的索引
    test_indices = [];
    train_indices = [];
    for class_idx = 1:num_classes
        train_indices = [train_indices; train_indices_cell{class_idx}(:)];  % 合并训练集
    end
    for class_idx = 1:num_classes
        test_indices = [test_indices; test_indices_cell{class_idx}(:)];  % 合并测试集
    end

    % 划分训练集和测试集数据
    NX_train = NX_org(:, :, train_indices);
    NX_test = NX_org(:, :, test_indices);
    train_labels = labels(train_indices);
    test_labels = labels(test_indices);

    %% 添加异常值并进行实验
      %% 添加异常值并进行实验（按 ratio × size × type 循环）
    total_train = size(NX_train, 3);
    for ratio_idx = 1:length(outlier_ratios)
        r = outlier_ratios(ratio_idx)
        num_outliers = round(r * total_train);
        outlier_indices = randsample(total_train, num_outliers);
        
        for size_idx = 1:length(outlier_size)
            s = outlier_size(size_idx)
            
            for type_idx = 1:length(anomaly_types)
                t = anomaly_types{type_idx}
                
                % 从干净训练集开始
                NX_train_outlier = add_outliers(NX_train, outlier_indices, t, s);
                
                % 执行降维+分类
                accuracy_results = dimensionality_reduction_and_classification( ...
                    NX_train_outlier, NX_test, train_labels, test_labels, ...
                    algorithms, q, options );
                
                % 存储到对应的 cell 位置
                ALL_RECOGNITION_ACCURACIES_test{ ...
                  dataset_idx, ratio_idx, type_idx, size_idx } = accuracy_results;
            end
        end
    end
end
% 输出 ALL_RECOGNITION_ACCURACIES_test 的尺寸
disp(size(ALL_RECOGNITION_ACCURACIES_test));

save_folder = 'H:\Students\Jiang_haijun\202506\recognitionrate等1个文件\recognitionrate\0923\PEMS';  % 指定保存文件的文件夹路径
% 可视化并保存结果
for dataset_idx = 3:3 % length(dataname)
    selected_dataset = dataname{dataset_idx};
    for ratio_idx = 1:length(outlier_ratios)
        for outlier_size_idx = 1:length(outlier_size)
            figure;
            % 绘制图表并保存
            outlier_block_size = outlier_size(outlier_size_idx);

            subplot(1, 2, 1);  % 左侧子图，显示 salt_and_pepper
            hold on;

            accuracy_for_ratio_sap = ALL_RECOGNITION_ACCURACIES_test{dataset_idx, ratio_idx, 1, outlier_size_idx};

            % 检查 accuracy_for_ratio_sap 的大小，确保它有足够的列数
            disp('accuracy_for_ratio_sap size:');
            disp(size(accuracy_for_ratio_sap));  % 打印数组尺寸，检查是否符合预期

            % 在绘图之前，检查 `accuracy_for_ratio_sap` 的维度是否适合使用 `q` 索引
            % 如果 accuracy_for_ratio_sap 的列数小于 q，动态调整 q
            [num_algorithms, num_features] = size(accuracy_for_ratio_sap);
            if num_features < q
                % 如果列数不足，调整 q 为实际的列数
                q = num_features;
                warning('accuracy_for_ratio_sap 列数不足，q 已调整为实际列数：%d', q);
            end

            for alg_idx = 1:num_algorithms
                plot(1:q, accuracy_for_ratio_sap(alg_idx, 1:q), '-o', 'LineWidth', 1.5, 'DisplayName', ...
                    [algorithms{alg_idx}, ' | Block Size: ', num2str(outlier_block_size)]);
            end
            xlabel('Feature Dimension (q)');
            ylabel('Recognition Accuracy (%)');
            title(['Salt and Pepper Noise']);
            legend('Location', 'Best');
            grid on;
            hold off;

            subplot(1, 2, 2);  % 右侧子图，显示 impulse
            hold on;

            accuracy_for_ratio_impulse = ALL_RECOGNITION_ACCURACIES_test{dataset_idx, ratio_idx, 2, outlier_size_idx};

            % 输出 accuracy_for_ratio_impulse 的尺寸，调试用
            disp('accuracy_for_ratio_impulse size:');
            disp(size(accuracy_for_ratio_impulse));  % 打印数组尺寸，检查是否符合预期

            % 检查 accuracy_for_ratio_impulse 数组的维度是否正确
            [num_algorithms_impulse, num_features_impulse] = size(accuracy_for_ratio_impulse);
            if num_algorithms_impulse < length(algorithms) || num_features_impulse < q
                error('accuracy_for_ratio_impulse 的尺寸不匹配，请检查数据存储过程');
            end

            for alg_idx = 1:num_algorithms_impulse
                plot(1:q, accuracy_for_ratio_impulse(alg_idx, 1:q), '-o', 'LineWidth', 1.5, 'DisplayName', ...
                    [algorithms{alg_idx}, ' | Block Size: ', num2str(outlier_block_size)]);
            end
            xlabel('Feature Dimension (q)');
            ylabel('Recognition Accuracy (%)');
            title(['Impulse Noise']);
            legend('Location', 'Best');
            grid on;
            hold off;

            sgtitle(['Dataset: ', selected_dataset, ' | Outlier Ratio: ', num2str(outlier_ratios(ratio_idx)), ' | Block Size: ', num2str(outlier_block_size)]);

            % 保存图像
            file_name = sprintf('%s_outratio_%.2f_blocksize_%d.png', selected_dataset, outlier_ratios(ratio_idx), outlier_block_size);
            save_path = fullfile(save_folder, selected_dataset, file_name);
            if ~exist(fullfile(save_folder, selected_dataset), 'dir')
                mkdir(fullfile(save_folder, selected_dataset));
            end
            print(gcf, save_path, '-dpng', '-r300');
        end
    end
end

% —— 提取并保存“最优错误率”及其标准差表格 —— 

% 假设以下变量已在工作区中：
%   dataname, outlier_ratios, outlier_size, anomaly_types,
%   algorithms, q, ALL_RECOGNITION_ACCURACIES_test, save_folder

% 1) 预估总行数（每组只保留一次）
numDatasets = numel(dataname);
numRatios   = numel(outlier_ratios);
numSizes    = numel(outlier_size);
numTypes    = numel(anomaly_types);
numAlgs     = numel(algorithms);
N = numDatasets * numRatios * numSizes * numTypes * numAlgs;
results_cell = cell(N, 8);  % 增加一列用于存储标准差

% 2) 展开所有组合，找出最优错误率及对应维度
row = 1;
for di = 3:3%numDatasets
    ds = dataname{di};
    for ri = 1:numRatios
        r = outlier_ratios(ri);
        for si = 1:numSizes
            s = outlier_size(si);
            for ti = 1:numTypes
                t = anomaly_types{ti};
                mat = ALL_RECOGNITION_ACCURACIES_test{di,ri,ti,si};  % [numAlgs × q]
                if isempty(mat) || any(size(mat) < [numAlgs, q])
                    warning('跳过无效数据: %s %.2f %d %s', ds, r, s, t);
                    continue;
                end
                for ai = 1:numAlgs
                    alg = algorithms{ai};
                    % 计算错误率矩阵
                    err = 1 - mat(ai,:);  % 错误率 = 1 - 精度
                    % 找最小错误率及对应维度
                    [bestErr, bestIdx] = min(err);
                    % 计算错误率的标准差
                    stdErr = std(err);  % 计算该算法的错误率标准差

                    results_cell(row,:) = { ...
                        ds, ...              % Dataset
                        r, ...               % OutlierRatio
                        s, ...               % BlockSize
                        t, ...               % AnomalyType
                        alg, ...             % Algorithm
                        bestIdx, ...         % BestDimension
                        bestErr, ...         % BestErrorRate
                        stdErr ...           % Standard Deviation of Error
                    };
                    row = row + 1;
                end
            end
        end
    end
end

% 裁剪空行
results_cell = results_cell(1:row-1, :);

% 3) 转 table 并写 Excel
T = cell2table(results_cell, ...
  'VariableNames', { ...
    'Dataset', ...
    'OutlierRatio', ...
    'BlockSize', ...
    'AnomalyType', ...
    'Algorithm', ...
    'BestDimension', ...
    'BestErrorRate', ...
    'ErrorStdDev' ...   % 新增标准差列
  });

T.Dataset     = categorical(T.Dataset);
T.AnomalyType = categorical(T.AnomalyType);


% save_folder = 'H:\Students\Jiang_haijun\202506\recognitionrate等1个文件\recognitionrate\result3\pens';  % 指定保存文件夹
% 
% % 假设 q 在之前已经定义为最大的特征维度
% % q = 50;  % (示例)
% 
% for dataset_idx = 3:3 % length(dataname)
%     selected_dataset = dataname{dataset_idx};
%     for ratio_idx = 1:length(outlier_ratios)
%         for outlier_size_idx = 1:length(outlier_size)
%             outlier_block_size = outlier_size(outlier_size_idx);
% 
%             %% —— 图 1：Salt and Pepper Noise —— %%
%             figure;
%             hold on;
%             % 读取并检查数据
%             accuracy_for_ratio_sap = ALL_RECOGNITION_ACCURACIES_test{dataset_idx, ratio_idx, 1, outlier_size_idx};
%             fprintf('Dataset %s, ratio %.2f, block %d: size of SAP = [%d %d]\n', ...
%                     selected_dataset, outlier_ratios(ratio_idx), outlier_block_size, size(accuracy_for_ratio_sap));
%             [num_algorithms, num_features] = size(accuracy_for_ratio_sap);
%             if num_features < q
%                 q = num_features;
%                 warning('SAP 列数不足，q 已调整为 %d', q);
%             end
%             % 绘制曲线
%             for alg_idx = 1:num_algorithms
%                 plot(1:q, accuracy_for_ratio_sap(alg_idx, 1:q), '-o', ...
%                      'LineWidth', 1.5, ...
%                      'DisplayName', sprintf('%s | Block %d', algorithms{alg_idx}, outlier_block_size));
%             end
%             xlabel('Feature Dimension (q)');
%             ylabel('Recognition Accuracy (%)');
%             title(sprintf('Dataset: %s   —   Salt and Pepper Noise', selected_dataset));
            % legend('Location', 'Best');
            % grid on;
            % hold off;
            % % 保存
            % sap_name = sprintf('%s_outratio_%.2f_blocksize_%d_sap.png', ...
            %                    selected_dataset, outlier_ratios(ratio_idx), outlier_block_size);
            % sap_folder = fullfile(save_folder, selected_dataset);
            % if ~exist(sap_folder, 'dir')
            %     mkdir(sap_folder);
            % end
            % print(gcf, fullfile(sap_folder, sap_name), '-dpng', '-r300');
            % 
            % 
            % %% —— 图 2：Impulse Noise —— %%
            % figure;
            % hold on;
            % % 读取并检查数据
            % accuracy_for_ratio_impulse = ALL_RECOGNITION_ACCURACIES_test{dataset_idx, ratio_idx, 2, outlier_size_idx};
            % fprintf('Dataset %s, ratio %.2f, block %d: size of Impulse = [%d %d]\n', ...
            %         selected_dataset, outlier_ratios(ratio_idx), outlier_block_size, size(accuracy_for_ratio_impulse));
            % [num_algorithms_imp, num_features_imp] = size(accuracy_for_ratio_impulse);
            % if num_features_imp < q
            %     q = num_features_imp;
            %     warning('Impulse 列数不足，q 已调整为 %d', q);
            % end
            % if num_algorithms_imp < length(algorithms)
            %     error('Impulse 噪声数据行数不匹配算法数量，请检查数据存储');
            % end
            % % 绘制曲线
            % for alg_idx = 1:num_algorithms_imp
                % plot(1:q, accuracy_for_ratio_impulse(alg_idx, 1:q), '-o', ...
%                      'LineWidth', 1.5, ...
%                      'DisplayName', sprintf('%s | Block %d', algorithms{alg_idx}, outlier_block_size));
%             end
%             xlabel('Feature Dimension (q)');
%             ylabel('Recognition Accuracy (%)');
%             title(sprintf('Dataset: %s   —   Impulse Noise', selected_dataset));
%             legend('Location', 'Best');
%             grid on;
%             hold off;
%             % 保存
%             imp_name = sprintf('%s_outratio_%.2f_blocksize_%d_impulse.png', ...
%                                selected_dataset, outlier_ratios(ratio_idx), outlier_block_size);
%             print(gcf, fullfile(sap_folder, imp_name), '-dpng', '-r300');
% 
%         end
%     end
% end






if ~exist(save_folder,'dir')
  mkdir(save_folder);
end
outfile = fullfile(save_folder, 'recognition_best_errorrate_with_std.xlsx');
writetable(T, outfile);
fprintf('已生成最优错误率和标准差表格：%s\n', outfile);



for dataset_idx = 3:3
    selected_dataset = dataname{dataset_idx};
    for ratio_idx = 1:length(outlier_ratios)
        for outlier_size_idx = 1:length(outlier_size)
            figure;

            % 只有 1 个异常类型（impulse），直接取 type_idx = 1
            type_idx = 1;
            outlier_block_size = outlier_size(outlier_size_idx);

            % 读取数据
            acc_mat = ALL_RECOGNITION_ACCURACIES_test{dataset_idx, ratio_idx, type_idx, outlier_size_idx};

            % 容错：空就跳过
            if isempty(acc_mat)
                warning('空结果：%s ratio=%.2f block=%d', selected_dataset, outlier_ratios(ratio_idx), outlier_block_size);
                close(gcf);  % 不保存空图
                continue;
            end

            [num_algorithms, num_features] = size(acc_mat);
            q_plot = min(q, num_features);   % 不要改全局 q

            hold on;
            for alg_idx = 1:num_algorithms
                plot(1:q_plot, acc_mat(alg_idx, 1:q_plot), '-o', 'LineWidth', 1.5, ...
                    'DisplayName', sprintf('%s | Block %d', algorithms{alg_idx}, outlier_block_size));
            end
            xlabel('Feature Dimension (q)');
            ylabel('Recognition Accuracy'); % 如果 acc 是 0~1 小数；若是百分比就改成 "(%)"
            title('Impulse Noise');
            legend('Location', 'Best');
            grid on; hold off;

            sgtitle(sprintf('Dataset: %s | Outlier Ratio: %.2f | Block Size: %d', ...
                selected_dataset, outlier_ratios(ratio_idx), outlier_block_size));

            % 保存
            file_name = sprintf('%s_outratio_%.2f_blocksize_%d.png', ...
                selected_dataset, outlier_ratios(ratio_idx), outlier_block_size);
            save_path = fullfile(save_folder, selected_dataset, file_name);
            if ~exist(fullfile(save_folder, selected_dataset), 'dir')
                mkdir(fullfile(save_folder, selected_dataset));
            end
            print(gcf, save_path, '-dpng', '-r300');
        end
    end
end


num_repeats = 10;                           % 想重复的次数
seeds = 1000 + (1:num_repeats);            % 固定种子，保证可复现

% 结果容器：每个组合、每个算法，存 num_repeats 次的"本次最佳准确率"
bestAcc_runs = cell(length(dataname), length(outlier_ratios), length(anomaly_types), length(outlier_size));
bestDim_runs = cell(size(bestAcc_runs));    % 同时记录"本次最佳维度"，可做众数/分布分析

for dataset_idx = 3:3
  selected_dataset = dataname{dataset_idx};
  % —— 加载数据（与你现在一致）——

  for ri = 1:length(outlier_ratios)
    r = outlier_ratios(ri);
    for si = 1:length(outlier_size)
      s = outlier_size(si);
      for ti = 1:length(anomaly_types)
        t = anomaly_types{ti};

        % 预分配：numAlgs × num_repeats
        numAlgs = numel(algorithms);
        accMat_runs = nan(numAlgs, num_repeats);
        dimMat_runs = nan(numAlgs, num_repeats);

        for rep = 1:num_repeats
          rng(seeds(rep), 'twister');      % 每次重现

          % 1) 分层划分（按你现在的 85%/15% 逻辑）
          % 2) 计算 total_train，依据 r 随机挑 outlier_indices
          % 3) NX_train_outlier = add_outliers(...)
          % 4) accuracy_results = dimensionality_reduction_and_classification(...)

          % 取"本次"的最佳准确率（注意：若 accuracy_results 是 0~1，此处无需/不应再 /100）
          [numAlgs_chk, numFeat] = size(accuracy_results);
          assert(numAlgs_chk == numAlgs, '算法数不匹配');
          for ai = 1:numAlgs
            [bestAcc_thisRun, bestDim_thisRun] = max(accuracy_results(ai, :));
            accMat_runs(ai, rep) = bestAcc_thisRun;
            dimMat_runs(ai, rep) = bestDim_thisRun;
          end
        end

        % 存起来，供后续统计与导出
        bestAcc_runs{dataset_idx, ri, ti, si} = accMat_runs;
        bestDim_runs{dataset_idx, ri, ti, si} = dimMat_runs;
      end
    end
  end
end

% —— 汇总成表：对 accMat_runs 做均值&标准差 —— 
rows = {};
for dataset_idx = 3:3
  ds = dataname{dataset_idx};
  for ri = 1:length(outlier_ratios)
    r = outlier_ratios(ri);
    for si = 1:length(outlier_size)
      s = outlier_size(si);
      for ti = 1:length(anomaly_types)
        t = anomaly_types{ti};
        accRuns = bestAcc_runs{dataset_idx, ri, ti, si};   % [numAlgs × num_repeats]
        dimRuns = bestDim_runs{dataset_idx, ri, ti, si};

        if isempty(accRuns), continue; end
        for ai = 1:numAlgs
          muAcc  = mean(accRuns(ai, :), 'omitnan');        % 平均"最佳准确率"
          sdAcc  = std(accRuns(ai, :),  'omitnan');        % 跨重复的标准差
          % 也可以给出"最佳维度"的众数（最常出现的最佳维度）
          bestDim_mode = mode(dimRuns(ai, :));
          rows(end+1, :) = {ds, r, s, t, algorithms{ai}, muAcc, sdAcc, bestDim_mode};
        end
      end
    end
  end
end

T_repeat = cell2table(rows, 'VariableNames', ...
  {'Dataset','OutlierRatio','BlockSize','AnomalyType','Algorithm', ...
   'BestAccuracyMean','BestAccuracyStd','BestDimensionMode'});

writetable(T_repeat, fullfile(save_folder, 'best_accuracy_mean_std_across_repeats.xlsx'));
