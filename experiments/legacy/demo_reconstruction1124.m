%% 全量实验脚本（修正版：修复 r=0、默认注入版本、输出路径、尺寸检查、绘图美化 等）
clear; close all; clc;

%% ---------------- 修复/改进: 可复现随机种子 ----------------
rng(20240913, 'twister');   % 固定随机种子，保证可复现

%% 0) 切换开关：是否使用第一版的 add_outliers（默认 false 保持"安全实现"口径）
%% 若你想对比第一版，将此设为 true 并在日志/输出中注明
USE_V1_OUTLIER = true;  % <<-- 修复：统一默认行为为 safe（更稳健/可比）
addpath('H:\Students\Jiang_haijun\202506\data\PEMS');
addpath('H:\Students\Jiang_haijun\202506\data\auslan');
addpath('H:\Students\Jiang_haijun\202506\data\wafer');
%% 1) 路径设置（合并、去重、按需开启）
projectRoot = 'H:\Students\Jiang_haijun\202506';
p_add = {
    fullfile(projectRoot, 'main', 'PK'), ...
    fullfile(projectRoot, 'main'), ...
    fullfile(projectRoot, 'Reconstruction'), ...
    % 'H:\Students\Jiang_haijun\B2DLDA1111\orl', ...
    % 'H:\Students\Jiang_haijun\BPCAL1\data\auslan', ...
    % 'H:\Students\Jiang_haijun\202506\data\walkVsRun'
};
for i=1:numel(p_add), if exist(p_add{i},'dir'), addpath(p_add{i}); end, end

%% 2. 参数定义（仅 outlier_size 改回第一版那组）
% dataname        = {'auslan','YALE','wafer','japan','PEMS','bci','CMU','ORL','PIE','AR','xm2vts','ecg'};
dataname={'PEMS'};
dataset_indices = 1;   % 只跑 auslan（保持第二版默认行为）
outlier_ratios  = [0.2,0.4,0.6,0.8];
outlier_size    = [129];

% outlier_ratios  = [0,0.2,0.4,0.6,0.8];
% outlier_size    = [5,25,35];
anomaly_types   = {'impulse'};
% anomaly_types   = {'impulse','salt_and_pepper'};

algorithms      = {'L1RBPCA','TPCA-L1-Greedy','TPCA-L1 Non-greedy','GLRAM','PE','BPCA'};
options(1)=0; options(3)=1e-8; options(14)=4; options(15)=0;
tol = 1e-6;

% 结果存储容器
ALL_RECON_ERRORS_test  = cell(numel(dataname), numel(outlier_ratios), numel(anomaly_types), numel(outlier_size));
ALL_RECON_ERRORS_train = cell(size(ALL_RECON_ERRORS_test));

%% 3. 主循环：加载 -> 划分 -> 注入噪声 -> 降维重构 -> 误差计算
for dataset_idx = dataset_indices
    ds = dataname{dataset_idx};
    fprintf('Dataset = %s\n', ds);

    % —— 加载数据：统一接口 —— %
    [X_org, labels, q_default] = load_dataset_unified(ds);
    NX_org = X_org;
    [dr, dc, ~] = size(NX_org);

    % —— 划分（分层 85/15） —— %
    unique_labels = unique(labels(:));
    train_idx = []; test_idx = [];
    for c = 1:numel(unique_labels)
        idxs = find(labels==unique_labels(c)); % 列向量
        n = numel(idxs); nt = max(1, floor(0.85*n));   % <<-- 修复：确保每类至少 1 个训练样本
        sh = idxs(randperm(n)).';              % 行向量
        train_idx = [train_idx; sh(1:nt).'];
        test_idx  = [test_idx;  sh(nt+1:end).'];
    end

    NX_train    = NX_org(:,:,train_idx);
    NX_test     = NX_org(:,:,test_idx);
    NX_test_out = NX_test;   % 测试集默认不加噪（与第二版一致）

    % —— 防止 q 超维 —— %
    q_max = min([q_default, dr, dc]);

    total_train = size(NX_train,3);
    numAlgs = numel(algorithms);

    for rI = 1:numel(outlier_ratios)
        r = outlier_ratios(rI)
        % ====== 修复：当 r=0 时，不注入任何异常 ======
        num_out = round(r * total_train);
        if num_out > 0
            outlier_indices = randperm(total_train, num_out);
        else
            outlier_indices = [];  % 表示无异常
        end

        for sI = 1:numel(outlier_size)
            block_sz = outlier_size(sI);

            for tI = 1:numel(anomaly_types)
                anomaly_type = anomaly_types{tI};

                % —— 注入噪声：默认用第二版的 safe 实现（保持原有结果口径）—— %
                if ~USE_V1_OUTLIER
                    % 第二版口径（与之前保持一致）：density = 0.6（显式传入）
                    NX_train_out = add_outliers_safe(NX_train, outlier_indices, anomaly_type, block_sz, 0.6);
                else
                    % 第一版实现（原样保留供比较）
                    NX_train_out = add_outliers(NX_train, outlier_indices, anomaly_type, block_sz, 0);
                end

                % 扁平任务表：(alg, k)
                recon_err_tr = nan(numAlgs, q_max);
                recon_err_te = nan(numAlgs, q_max);

                % —— 串行执行（保持与第二版一致）—— %
                for aI = 1:numAlgs
                    alg = algorithms{aI};
                    for k = 1:q_max
                        % 随机初始化（与第二版一致）
                        mo1.U = orth(rand(dc, k));  % 行空间 (dc x k)
                        mo1.V = orth(rand(dr, k));  % 列空间 (dr x k)

                        % —— 统一入口：根据算法拟合 U/V —— %
                        [Uhat, Vhat] = fit_subspaces_by_alg_checked(alg, mo1, NX_train_out, k, options, tol, dr, dc);

                        if isempty(Uhat) || isempty(Vhat)
                            % 该算法未产出有效解或尺寸不符，跳过
                            continue;
                        end

                        % —— 向量化重构 & 误差（与第二版一致）—— %
                        rec_tr = reconstruct_pages_safe(NX_train_out, Vhat, Uhat);
                        err_tr = meanFrobPerPage(rec_tr - NX_train_out);

                        rec_te = reconstruct_pages_safe(NX_test_out, Vhat, Uhat);
                        err_te = meanFrobPerPage(rec_te - NX_test_out);

                        recon_err_tr(aI, k) = err_tr;
                        recon_err_te(aI, k) = err_te;
                    end
                end

                ALL_RECON_ERRORS_train{dataset_idx, rI, tI, sI} = recon_err_tr;
                ALL_RECON_ERRORS_test{ dataset_idx, rI, tI, sI} = recon_err_te;
            end
        end
    end
end

%% 4. 可视化 & 保存（输出目录动态化，图形美化）
%% ====== 输出根目录（带日期标签） ======
dateTag    = datestr(now,'yyyymmdd');
baseFolder = fullfile('H:\Students\Jiang_haijun\202506', ...
                      'Reconstruction', 'Reconstruction_PEMS_129', ...
                      dateTag);
if ~exist(baseFolder,'dir')
    mkdir(baseFolder);
end

%% ====== 统一配色与线型配置 ======
baseColors = lines(numel(algorithms));   % 经典颜色表
algStyles  = struct;
for aI = 1:numel(algorithms)
    alg   = algorithms{aI};
    color = baseColors(aI,:);
    switch alg
        case 'TPCA-L1-Greedy'
            % 稍微变浅一点，区分开
            color = 0.6*color + 0.4*[1 1 1];
            algStyles.(fixname(alg)) = struct( ...
                'LineStyle','--', 'LineWidth',2.2, ...
                'Marker','o',    'Color',color);
        otherwise
            algStyles.(fixname(alg)) = struct( ...
                'LineStyle','-',  'LineWidth',2.2, ...
                'Marker','o',     'Color',color);
    end
end

%% ====== 主循环：数据集 / 异常比例 / 异常块大小 / 异常类型 ======
for dataset_idx = dataset_indices
    ds = dataname{dataset_idx};

    % 当前数据集的输出文件夹
    output_folder = fullfile(baseFolder, ds);
    if ~exist(output_folder,'dir')
        mkdir(output_folder);
    end

    for rI = 1:numel(outlier_ratios)
        for sI = 1:numel(outlier_size)
            for tI = 1:numel(anomaly_types)

                anomaly_type = anomaly_types{tI};
                err = ALL_RECON_ERRORS_test{dataset_idx, rI, tI, sI};
                if isempty(err)
                    continue;
                end

                q_plot = size(err, 2);   % 可视化的维度个数

                %% ---- 画图 ----
                hfig = figure('Color','w', ...
                              'Position',[100 100 900 600]); 
                hold on;

                for aI = 1:numel(algorithms)
                    alg   = algorithms{aI};
                    style = algStyles.(fixname(alg));

                    if aI <= size(err,1) && ~all(isnan(err(aI,:)))
                        plot(1:q_plot, err(aI,:), ...
                            'LineStyle',  style.LineStyle, ...
                            'LineWidth',  style.LineWidth, ...
                            'Marker',     style.Marker, ...
                            'MarkerSize', 6, ...
                            'Color',      style.Color, ...
                            'DisplayName', alg);
                    end
                end

                xlabel('Feature Dimension (q)', ...
                       'FontSize',12);
                ylabel('Reconstruction Error (Frobenius)', ...
                       'FontSize',12);

                title(sprintf('%s | Ratio=%.2f | Block=%d | %s', ...
                    ds, outlier_ratios(rI), outlier_size(sI), ...
                    strrep(anomaly_type,'_','\_')), ...
                    'FontSize',13);

                legend('Location','best','FontSize',10);
                grid on; box on;
                set(gca,'FontSize',11,'LineWidth',1);

                % 统一图像尺寸
                set(hfig,'Position',[100 100 900 600]);

                %% ---- 统一基础文件名（不带后缀）----
                fn_base = sprintf('%s_ratio_%.2f_block_%d_type_%s', ...
                    ds, outlier_ratios(rI), outlier_size(sI), anomaly_type);

                %% 1) 保存 PNG 高清图
                png_file = fullfile(output_folder, [fn_base '.png']);
                exportgraphics(hfig, png_file, 'Resolution', 300);

                %% 2) 保存 MAT 文件（关键数据）
                x_q = 1:q_plot;  % 横坐标
                mat_file = fullfile(output_folder, [fn_base '.mat']);
                save(mat_file, 'x_q', 'err', 'algorithms', ...
                               'ds', 'outlier_ratios', 'outlier_size', ...
                               'anomaly_type', 'dataset_idx', ...
                               'rI', 'sI', 'tI');

                %% 3) 保存 FIG 文件（MATLAB 原生 figure）
                fig_file = fullfile(output_folder, [fn_base '.fig']);
                savefig(hfig, fig_file);

                % 如果不想开太多窗口，可以解除下面的注释：
                % close(hfig);

            end
        end
    end
end

%% 5. 提取并保存"最优重构误差"表格（与第二版一致）
numDatasets = numel(dataname); numRatios = numel(outlier_ratios);
numBlocks   = numel(outlier_size); numTypes  = numel(anomaly_types);
numAlgs     = numel(algorithms);

results_cell = cell(numDatasets * numRatios * numBlocks * numTypes * numAlgs, 7);
row = 1;

for di = dataset_indices
    ds = dataname{di};
    for ri = 1:numRatios
        r = outlier_ratios(ri);
        for bi = 1:numBlocks
            b = outlier_size(bi);
            for ti = 1:numTypes
                t = anomaly_types{ti};
                mat = ALL_RECON_ERRORS_test{di, ri, ti, bi};  % [numAlgs × q]
                if isempty(mat) || size(mat,1) < numAlgs || size(mat,2) < 1
                    warning('跳过无效数据: %s ratio=%.2f block=%d type=%s', ds, r, b, t);
                    continue;
                end
                for ai = 1:numAlgs
                    alg = algorithms{ai};
                    errs = mat(ai, :);
                    if all(isnan(errs)), continue; end
                    [bestErr, bestDim] = min(errs);
                    results_cell(row, :) = { ...
                        ds, r, b, t, alg, bestDim, bestErr ...
                    };
                    row = row + 1;
                end
            end
        end
    end
end

% 裁剪空行
results_cell = results_cell(1:row-1, :);

% 转 table 并写 Excel
T = cell2table(results_cell, ...
  'VariableNames', {'Dataset','OutlierRatio','BlockSize','AnomalyType','Algorithm','BestDimension','BestReconstructionError'});

T.Dataset     = categorical(T.Dataset);
T.AnomalyType = categorical(T.AnomalyType);

if ~exist(output_folder, 'dir')
  mkdir(output_folder);
end
outfile = fullfile(output_folder, 'reconstruction_best_error.xlsx');
writetable(T, outfile);
fprintf('已生成最优重构误差表格：%s\n', outfile);

%% ================= 工具函数区（含尺寸检查版 fit） =================

function [Uhat, Vhat] = fit_subspaces_by_alg_checked(alg, mo1, data3d, k, options, tol, dr, dc)
% 统一入口（加强版）：调用算法并做输出尺寸检查
    Uhat = []; Vhat = [];
    try
        switch alg
            case 'L1RBPCA'
                result = TPCA_RBPCA(mo1, data3d, [k,k], tol);
                if isfield(result,'U') && isfield(result,'V')
                    Uhat = result.U; Vhat = result.V;
                end
            case 'TPCA-L1-Greedy'
                mo_greedy = TPCA_Greed_test(mo1, data3d,[k,k], tol);
                if isfield(mo_greedy,'U') && isfield(mo_greedy,'V')
                    Uhat = mo_greedy.U; Vhat = mo_greedy.V;
                end
            case 'TPCA-L1 Non-greedy'
                mo_ng = Non_Greedy(mo1, data3d, k, tol);
                if isfield(mo_ng,'U') && isfield(mo_ng,'V')
                    Uhat = mo_ng.U; Vhat = mo_ng.V;
                end
            case 'GLRAM'
                [bp,~] = GLRAM(mo1, data3d, k, k, options);
                if isfield(bp,'U') && isfield(bp,'V')
                    Uhat = bp.U; Vhat = bp.V;
                end
            case 'PE'
                [R,C] = projected_estimator(mo1, data3d, k, k);
                Uhat = C; Vhat = R;
            case 'BPCA'
                [R,C] = BPCA(data3d, k, k);
                Uhat = C; Vhat = R;
            otherwise
                warning('未知算法：%s（跳过）', alg);
        end
    catch ME
        warning('算法 %s 在 k=%d 出错：%s', alg, k, ME.message);
        Uhat = []; Vhat = [];
        return;
    end

    % 尺寸检查（必须为 U: [dc x k], V: [dr x k]）
    if ~isempty(Uhat) && ~isempty(Vhat)
        if ~isequal(size(Uhat), [dc, k]) || ~isequal(size(Vhat), [dr, k])
            warning('算法 %s 返回的 U/V 尺寸不符 (U: %s, V: %s) for k=%d. 跳过。', ...
                alg, mat2str(size(Uhat)), mat2str(size(Vhat)), k);
            Uhat = []; Vhat = [];
        end
    end
end

function [X_org, labels, q] = load_dataset_unified(ds)
    switch ds
        case 'japan'
            load('japan.mat');       labels=label_org; q=7;     %#ok<LOAD>
            if ~exist('X_org','var'), error('japan.mat 中未找到 X_org'); end
        case 'CMU'
            load('CMUsubject16all.mat'); labels=label; q=60; X_org=datatruncated;
        case 'YALE'
            load('YALE64.mat'); load('YALElabel.mat'); labels=label; q=60;
            if ~exist('X_org','var'), error('YALE64.mat 中未找到 X_org'); end
        case 'ORL'
            load('ORL.mat'); load('ORLlabel.mat'); labels=label; q=60;
        case 'PIE'
            load('data_norm.mat'); load('label.mat'); labels=label; q=60;
            if ~exist('X_org','var'), error('data_norm.mat 中未找到 X_org'); end
        case 'AR'
            load('AR.mat'); load('ARlabel.mat'); labels=label; q=60;
        case 'xm2vts'
            load('xm2vts.mat'); load('xm2vtslabel.mat'); labels=label; q=50;
        case 'auslan'
            load('auslan.mat'); labels=label_org; q=22;
            if ~exist('X_org','var'), error('auslan.mat 中未找到 X_org'); end
        case 'trx'
            load('trx.mat'); labels=label_org; q=10;
            if ~exist('X_org','var'), error('trx.mat 中未找到 X_org'); end
        case 'ecg'
            load('ecg.mat'); labels=label_org; q=2;
            if ~exist('X_org','var'), error('ecg.mat 中未找到 X_org'); end
        case 'wafer'
            load('wafer.mat'); labels=label_org; q=6;
            if ~exist('X_org','var'), error('wafer.mat 中未找到 X_org'); end
        case 'bci'
            load('bci500.mat'); labels=X_label_org; q=28;
            if ~exist('X_org','var'), error('bci500.mat 中未找到 X_org'); end
        case 'PEMS'
            load('PEMSsorttrans.mat'); X_org=datatruncated; labels=label; q=60;
        otherwise
            error('未知数据集：%s', ds);
    end

    if ~isa(X_org,'double'), X_org = double(X_org); end
    labels = labels(:);
end

function NX_train_outlier = add_outliers_safe(NX_train, outlier_indices, anomaly_type, outlier_size, density)
    % 若没有要注入的样本，直接返回原数据
    if isempty(outlier_indices)
        NX_train_outlier = NX_train;
        return;
    end
    if nargin < 5, density = 0.1; end
    [dr, dc, N] = size(NX_train);
    NX_train_outlier = NX_train;

    for idx = outlier_indices(:).'
        assert(idx >= 1 && idx <= N, 'outlier_indices 越界');
        img  = NX_train(:, :, idx);
        work = double(img);

        h = min(outlier_size, dr);  w = min(outlier_size, dc);
        r1 = randi([1, dr - h + 1]); c1 = randi([1, dc - w + 1]);

        block = work(r1:r1+h-1, c1:c1+w-1);

        lo = min(work(:));
        hi = max(work(:));
        s  = std(work(:));
        if hi == lo
            hi = lo + 1;
            if s == 0, s = 1; end
        end

        switch lower(anomaly_type)
            case 'salt_and_pepper'
                mask = rand(h, w) < density;
                salt = rand(h, w) < 0.5;
                block(mask & salt)  = hi;
                block(mask & ~salt) = lo;

            case 'impulse'
                k = 3; wspan = 2; mu = mean(work(:));
                up_lo = mu + k*s;           up_hi = mu + (k+wspan)*s;
                dn_lo = mu - (k+wspan)*s;   dn_hi = mu - k*s;
                mask = rand(h, w) < density; side = rand(h, w) < 0.5;
                up_rand = up_lo + (up_hi - up_lo) .* rand(h, w);
                dn_rand = dn_lo + (dn_hi - dn_lo) .* rand(h, w);
                block(mask & side)  = up_rand(mask & side);
                block(mask & ~side) = dn_rand(mask & ~side);

            otherwise
                error('未知异常类型：%s', anomaly_type);
        end

        work(r1:r1+h-1, c1:c1+w-1) = block;
        NX_train_outlier(:, :, idx) = work;   % 保持 double
    end
end

function NX_train_outlier = add_outliers(NX_train, outlier_indices, anomaly_type, outlier_size, anomaly_length)
    % 第一版实现（保留供对比），并处理空索引
    if isempty(outlier_indices)
        NX_train_outlier = NX_train; return;
    end
    [dr, dc, N] = size(NX_train);
    NX_train_outlier = NX_train;

    for idx = outlier_indices'
        img = NX_train(:, :, idx);
        row0 = randi(dr);
        col0 = randi(dc);
        h = min(outlier_size, dr - row0 + 1);
        w = min(outlier_size, dc - col0 + 1);
        original_patch = img(row0:row0 + h - 1, col0:col0 + w - 1);

        switch anomaly_type
            case 'salt_and_pepper'
                density = 0.8;
                anomaly_vals = original_patch;
                mask = rand(h, w) < density;
                anomaly_vals(mask) = randi([0, 1], sum(mask(:)), 1) * 500;
            case 'impulse'
                density = 0.8;
                mask = rand(h, w) < density;
                salt = rand(h, w) < 0.5;
                anomaly_vals = original_patch;
                anomaly_vals(mask & salt)   = 100;
                anomaly_vals(mask & ~salt)  = 0;
            case 'uniform'
                anomaly_vals = rand(h, w) * anomaly_length;
            otherwise
                error('未知异常类型：%s', anomaly_type);
        end

        img(row0:row0 + h - 1, col0:col0 + w - 1) = anomaly_vals;
        NX_train_outlier(:, :, idx) = img;
    end
end

function reconstructed = reconstruct_pages_safe(X, V, U)
    if isempty(U) || isempty(V)
        reconstructed = zeros(size(X),'like',X);
        return;
    end
    if exist('pagemtimes','file') == 2
        X_low = pagemtimes(pagemtimes(permute(V,[2 1]), X), U);     % V' * X * U
        reconstructed = pagemtimes(pagemtimes(V, X_low), permute(U,[2 1])); % V * X_low * U'
    else
        [dr, dc, N] = size(X);
        reconstructed = zeros(dr, dc, N, 'like', X);
        Ut = U'; Vt = V';
        for i = 1:N
            Xi = X(:,:,i);
            Xlow = Vt * Xi * U;             % [k x k]
            reconstructed(:,:,i) = V * Xlow * Ut;
        end
    end
end

function m = meanFrobPerPage(AminusB)
    D2 = sum((AminusB).^2, [1 2]);  % [1 x 1 x N]
    m  = mean(sqrt(D2), 3);
end

function s = fixname(alg)
    s = matlab.lang.makeValidName(alg);
end
