% function demo_select_dim_robust_outliers_noPCA_kneeAvg
% ============================================================
% Robust bilinear DR with local outliers (NO PCA baseline)
% - L1-RBPCA vs NonGreedy
% - Fig1..5(原有) + Fig6(平均重构@k*：FVU/NRMSE) + Fig7(平均识别率) + Fig8(识别流程时间)
% - 总表1: summary_all_at_knee.csv (原有：L1@k* & NG@k* 的 FVU/NRMSE)
% - 总表2: summary_all_nrmse_ngbest.csv (NG/L1 的 NRMSE + best-k 信息)
% - 总表3: summary_all_acc_time.csv (新增：识别率 + 时间对比)
%   summary_all_nrmse 列为：
%   Dataset, TrainRatio, OutlierRate, KneeDim_L1, CAP_L1_at_knee, RepeatsUsed,
%   NRMSE_L1_knee_mean,  NRMSE_L1_knee_std,
%   NRMSE_NG_atL1k_mean, NRMSE_NG_atL1k_std,
%   [MOD] k_L1_valBest_mean, k_L1_valBest_std,
%   [MOD] NRMSE_L1_valBest_mean, NRMSE_L1_valBest_std,
%   [MOD] k_L1_testBest_mean, k_L1_testBest_std,
%   [MOD] NRMSE_L1_testBest_mean, NRMSE_L1_testBest_std,
%   k_NG_valBest_mean,   k_NG_valBest_std,
%   NRMSE_NG_valBest_mean,  NRMSE_NG_valBest_std,
%   k_NG_testBest_mean,  k_NG_testBest_std,
%   NRMSE_NG_testBest_mean, NRMSE_NG_testBest_std
% ============================================================

clc; close all; clearvars;

%% ---------- Paths ----------
projectRoot = 'H:\Students\Jiang_haijun\202506';
dataFolders = {
    fullfile(projectRoot, 'data','auslan')
    fullfile(projectRoot, 'data','bci')
    fullfile(projectRoot, 'main','PK')
    fullfile(projectRoot, 'main')
    fullfile(projectRoot, 'data','japan')
    fullfile(projectRoot, 'data','yale')
    fullfile(projectRoot, 'data','PEMS')
};
addpath('H:\Students\Jiang_haijun\202506\main');

%% ---------- Global switches ----------
OUTDIR         = fullfile(projectRoot,'result','robust_outliers_noPCA_kneeAvg1');
if ~exist(OUTDIR,'dir'), mkdir(OUTDIR); end
MAKE_APPENDIX  = false;  %#ok<NASGU>
USE_SVM        = false;  %#ok<NASGU>

%% ---------- Dataset & protocol ----------
% dataname          = {'auslan','bci','japan','pems'};
dataname          = {'auslan','bci','japan'};
% dataname          = {'pems'};

train_ratio_list  = [0.70];
% outlier_rate_list = [0];
outlier_rate_list = [0,0.20, 0.4, 0.6];

% [MOD] 保留原有结构，不改其他设置；真正的异常块大小改为按数据集自动指定
size_list         = [13];
repeat            = 3;
val_ratio         = 0.10;
tolerance         = 1e-8;

% 曲线/knee参数
SMOOTH_W        = 3;
DERIV_THR_RATIO = 0.5;

% CAP/FVU 对应：FVU = 1 - CAP
CAP_TARGET      = 0.80;

rng(20250809,'twister');

%% ---------- Outlier injection (仅训练核心集) ----------
OUTLIER.ENABLE          = true;
OUTLIER.CORRUPT_CORE    = true;     % 只污染训练核心集
OUTLIER.CORRUPT_VALTEST = false;    % 验证/测试默认不污染
OUTLIER.TYPE            = 'impulse';% 'salt_and_pepper'|'impulse'|'uniform'
OUTLIER.BLOCK_RATIO_R   = 0.50;
OUTLIER.BLOCK_RATIO_C   = 0.50;
OUTLIER.DENSITY_INBLOCK = 0.50;
OUTLIER.BLOCK_SIZE      = [];       % [MOD] 新增：优先使用固定块边长；为空时才退回比例

%% 用于汇总所有数据集/参数配置的总表
summary_all       = table();  % 原有：含 FVU + NRMSE + CAP 等
summary_all_nrmse = table();  % 新增：NRMSE + NG/L1 best-k
summary_all_acc_time = table(); % 新增：识别率 + 时间对比

%% ================= Grid over settings =================
for ds_i = 1:numel(dataname)
for trr  = 1:numel(train_ratio_list)
for oor  = 1:numel(outlier_rate_list)
for bsz  = 1:numel(size_list)

    ds            = dataname{ds_i};
    train_ratio   = train_ratio_list(trr);
    outlier_rate  = outlier_rate_list(oor);

    % [MOD] 按数据集自动指定固定异常块大小
    blocksize          = get_dataset_blocksize(ds);
    OUTLIER.BLOCK_SIZE = blocksize;

    fprintf('[DATA] %s | train=%.2f | outlier=%.2f | size=%d | OUT:%s/%s\n', ...
        ds, train_ratio, outlier_rate, blocksize, ...
        yn(OUTLIER.CORRUPT_CORE), yn(OUTLIER.CORRUPT_VALTEST));

    %% ---------- Load & split ----------
    [data, labels, qmax] = load_dataset(ds);
    labels = labels(:);
    [dr, dc, ~] = size(data); %#ok<ASGLU>

    [data_train, labels_train, data_test_clean, labels_test] = stratified_split(data, labels, train_ratio);
    [data_tr_core_clean, labels_tr_core, data_val_clean, labels_val] = split_val(data_train, labels_train, val_ratio);

    % 能量（用于 FVU/CAP）
    norm2_val  = arrayfun(@(i) norm(data_val_clean(:,:,i),'fro')^2,  1:size(data_val_clean,3))';
    norm2_test = arrayfun(@(i) norm(data_test_clean(:,:,i),'fro')^2, 1:size(data_test_clean,3))';

    %% ---------- Containers ----------
    ACC_VAL.L1.knn  = zeros(repeat, qmax); ACC_VAL.NG.knn  = zeros(repeat, qmax);
    ACC_TEST.L1.knn = zeros(repeat, qmax); ACC_TEST.NG.knn = zeros(repeat, qmax);

    FVU_TEST.L1     = zeros(repeat,qmax);  FVU_TEST.NG     = zeros(repeat,qmax);
    CAP_TEST.L1     = zeros(repeat,qmax);  CAP_TEST.NG     = zeros(repeat,qmax);
    OBJ_L1          = zeros(repeat,qmax);  OBJ_NG          = zeros(repeat,qmax);

    TRUST.L1        = zeros(repeat,qmax);  TRUST.NG        = zeros(repeat,qmax);
    KNN_HIT.L1      = zeros(repeat,qmax);  KNN_HIT.NG      = zeros(repeat,qmax);

    % 【新增】验证集上的 FVU（给 NG/L1 自己选 best-k 用）
    FVU_VAL.L1      = zeros(repeat,qmax);  FVU_VAL.NG      = zeros(repeat,qmax);

    % 【新增】识别率流程时间：训练 + 特征提取 + 1-NN结果获取
    TIME_ACC_L1     = zeros(repeat,qmax);
    TIME_ACC_NG     = zeros(repeat,qmax);
    TIME_KNEE_L1    = zeros(repeat,1);
    TIME_PIPE_L1    = zeros(repeat,1);
    TIME_PIPE_NG_QMAX = zeros(repeat,1);
    TIME_PIPE_NG_FULL = zeros(repeat,1);

    KNEE_L1 = NaN(1,repeat);

    %% ===================== Repeats =====================
    for rep = 1:repeat
        rng(1000+rep,'twister');

        % ---- 污染控制 ----
        data_tr_core = data_tr_core_clean;
        data_val     = data_val_clean;
        data_test    = data_test_clean;
        if OUTLIER.ENABLE
            if OUTLIER.CORRUPT_CORE
                data_tr_core = inject_outliers_block(data_tr_core, outlier_rate, OUTLIER);  % [MOD] outlier=0 不污染
            end
            if OUTLIER.CORRUPT_VALTEST
                data_val  = inject_outliers_block(data_val,  outlier_rate, OUTLIER);
                data_test = inject_outliers_block(data_test, outlier_rate, OUTLIER);
            end
        end
        [Ntr_core, ~, Nte] = deal(size(data_tr_core,3), size(data_val,3), size(data_test,3)); %#ok<NASGU>

        % ---- 原空间 1-NN baseline（横线参考） ----
        rawAcc = raw_knn_baseline(data_tr_core, labels_tr_core, data_test, labels_test);
        K = 10;
        Xhi = reshape(permute(data_test, [3,1,2]), Nte, dr*dc);

        % ---- 扫描所有 k ----
        for k = 1:qmax
            mo.U = orth(rand(dc,k)); mo.V = orth(rand(dr,k));

            % ===== L1-RBPCA =====
            tAccL1 = tic;
            M_L1 = TPCA_RBPCA(mo, data_tr_core, [k,k], tolerance);
            OBJ_L1(rep,k)       = mean_obj_abs(data_tr_core, M_L1);

            % 提取验证/测试重构误差
            [ftr_tr_L1, ftr_va_L1, rec_va_L1]   = extract_feats_and_recon(M_L1, data_tr_core, data_val);
            [~,        ftr_te_L1, rec_te_L1]    = extract_feats_and_recon(M_L1, data_tr_core, data_test);

            ACC_VAL.L1.knn(rep,k)  = knn_acc(ftr_tr_L1, labels_tr_core, ftr_va_L1, labels_val);
            ACC_TEST.L1.knn(rep,k) = knn_acc(ftr_tr_L1, labels_tr_core, ftr_te_L1, labels_test);
            TIME_ACC_L1(rep,k)     = toc(tAccL1);

            FVU_VAL.L1(rep,k)  = mean((rec_va_L1.^2) ./ (norm2_val  + eps));
            FVU_TEST.L1(rep,k) = mean((rec_te_L1.^2) ./ (norm2_test + eps));
            CAP_TEST.L1(rep,k) = 1 - FVU_TEST.L1(rep,k);

            TRUST.L1(rep,k)   = trustworthiness_score(Xhi, ftr_te_L1, K);
            KNN_HIT.L1(rep,k) = knn_hit_rate(Xhi, ftr_te_L1, K);

            % ===== NonGreedy-L1 =====
            tAccNG = tic;
            M_NG = Non_Greedy(mo, data_tr_core, k, tolerance);
            OBJ_NG(rep,k)       = mean_obj_abs(data_tr_core, M_NG);

            [ftr_tr_NG, ftr_va_NG, rec_va_NG]   = extract_feats_and_recon(M_NG, data_tr_core, data_val);
            [~,        ftr_te_NG, rec_te_NG]    = extract_feats_and_recon(M_NG, data_tr_core, data_test);

            ACC_VAL.NG.knn(rep,k)  = knn_acc(ftr_tr_NG, labels_tr_core, ftr_va_NG, labels_val);
            ACC_TEST.NG.knn(rep,k) = knn_acc(ftr_tr_NG, labels_tr_core, ftr_te_NG, labels_test);
            TIME_ACC_NG(rep,k)     = toc(tAccNG);

            FVU_VAL.NG(rep,k)  = mean((rec_va_NG.^2) ./ (norm2_val  + eps));
            FVU_TEST.NG(rep,k) = mean((rec_te_NG.^2) ./ (norm2_test + eps));
            CAP_TEST.NG(rep,k) = 1 - FVU_TEST.NG(rep,k);

            TRUST.NG(rep,k)   = trustworthiness_score(Xhi, ftr_te_NG, K);
            KNN_HIT.NG(rep,k) = knn_hit_rate(Xhi, ftr_te_NG, K);
        end

        % ---- Knee: 由 L1 的训练目标曲线给出 ----
        tKnee = tic;
        [KNEE_L1(rep), ~] = knee_voted_with_strength(OBJ_L1(rep,:), SMOOTH_W, DERIV_THR_RATIO);
        TIME_KNEE_L1(rep) = toc(tKnee);

        TIME_PIPE_L1(rep)     = sum(TIME_ACC_L1(rep,:)) + TIME_KNEE_L1(rep);
        TIME_PIPE_NG_QMAX(rep)= TIME_ACC_NG(rep,qmax);
        TIME_PIPE_NG_FULL(rep)= sum(TIME_ACC_NG(rep,:));

        accL1_rep     = pick_at_k(ACC_TEST.L1.knn(rep,:), KNEE_L1(rep));
        accNGsame_rep = pick_at_k(ACC_TEST.NG.knn(rep,:), KNEE_L1(rep));
        accNGqmax_rep = ACC_TEST.NG.knn(rep,qmax);

        fprintf(['rep=%02d | L1-knee=%s | ACC[L1@k*]=%.4f | ACC[NG@k*]=%.4f | ' ...
                 'ACC[NG@qmax]=%.4f | Time[L1-pipe]=%.3fs | Time[NG@qmax]=%.3fs | raw 1-NN=%.4f\n'], ...
                 rep, num2str_or_nan(KNEE_L1(rep)), accL1_rep, accNGsame_rep, accNGqmax_rep, ...
                 TIME_PIPE_L1(rep), TIME_PIPE_NG_QMAX(rep), rawAcc);
    end % repeat

    %% ---------- NG: 选自己的 best-k（验证集 + 测试 oracle） ----------
    K_BEST_NG_val  = NaN(repeat,1);
    K_BEST_NG_test = NaN(repeat,1);
    for r = 1:repeat
        [~, K_BEST_NG_val(r)]  = min(FVU_VAL.NG(r, :));   % val-based
        [~, K_BEST_NG_test(r)] = min(FVU_TEST.NG(r, :));  % test oracle
    end

    % 对应在测试集上的 NRMSE
    NRMSE_NG_valBest  = zeros(repeat,1);
    NRMSE_NG_testBest = zeros(repeat,1);
    for r = 1:repeat
        NRMSE_NG_valBest(r)  = sqrt(FVU_TEST.NG(r, K_BEST_NG_val(r)));
        NRMSE_NG_testBest(r) = sqrt(FVU_TEST.NG(r, K_BEST_NG_test(r)));
    end

    % NG best-k 的统计量（选维稳定性 & 性能）
    k_ng_val_mean  = mean(K_BEST_NG_val);
    k_ng_val_std   = std(K_BEST_NG_val);
    k_ng_test_mean = mean(K_BEST_NG_test);
    k_ng_test_std  = std(K_BEST_NG_test);

    nrm_ng_valBest_mean  = mean(NRMSE_NG_valBest);
    nrm_ng_valBest_std   = std(NRMSE_NG_valBest);
    nrm_ng_testBest_mean = mean(NRMSE_NG_testBest);
    nrm_ng_testBest_std  = std(NRMSE_NG_testBest);

    %% ---------- [MOD] L1: 选自己的 best-k（验证集 + 测试 oracle） ----------
    K_BEST_L1_val  = NaN(repeat,1);
    K_BEST_L1_test = NaN(repeat,1);
    for r = 1:repeat
        [~, K_BEST_L1_val(r)]  = min(FVU_VAL.L1(r, :));   % val-based
        [~, K_BEST_L1_test(r)] = min(FVU_TEST.L1(r, :));  % test oracle
    end

    NRMSE_L1_valBest  = zeros(repeat,1);
    NRMSE_L1_testBest = zeros(repeat,1);
    for r = 1:repeat
        NRMSE_L1_valBest(r)  = sqrt(FVU_TEST.L1(r, K_BEST_L1_val(r)));
        NRMSE_L1_testBest(r) = sqrt(FVU_TEST.L1(r, K_BEST_L1_test(r)));
    end

    k_l1_val_mean  = mean(K_BEST_L1_val);
    k_l1_val_std   = std(K_BEST_L1_val);
    k_l1_test_mean = mean(K_BEST_L1_test);
    k_l1_test_std  = std(K_BEST_L1_test);

    nrm_l1_valBest_mean  = mean(NRMSE_L1_valBest);
    nrm_l1_valBest_std   = std(NRMSE_L1_valBest);
    nrm_l1_testBest_mean = mean(NRMSE_L1_testBest);
    nrm_l1_testBest_std  = std(NRMSE_L1_testBest);

    %% ---------- [MOD] rowN 固定列名（避免 table(...) + VariableNames 写法炸掉） ----------
    VarNamesN = { ...
        'Dataset','TrainRatio','OutlierRate', ...
        'KneeDim_L1','CAP_L1_at_knee','RepeatsUsed', ...
        'NRMSE_L1_knee_mean','NRMSE_L1_knee_std', ...
        'NRMSE_NG_atL1k_mean','NRMSE_NG_atL1k_std', ...
        'k_L1_valBest_mean','k_L1_valBest_std', ...
        'NRMSE_L1_valBest_mean','NRMSE_L1_valBest_std', ...
        'k_L1_testBest_mean','k_L1_testBest_std', ...
        'NRMSE_L1_testBest_mean','NRMSE_L1_testBest_std', ...
        'k_NG_valBest_mean','k_NG_valBest_std', ...
        'NRMSE_NG_valBest_mean','NRMSE_NG_valBest_std', ...
        'k_NG_testBest_mean','k_NG_testBest_std', ...
        'NRMSE_NG_testBest_mean','NRMSE_NG_testBest_std' ...
    };

    VarNamesA = { ...
        'Dataset','TrainRatio','OutlierRate','KneeDim_L1','RepeatsUsed', ...
        'ACC_L1_knee_mean','ACC_L1_knee_std', ...
        'ACC_NG_atL1k_mean','ACC_NG_atL1k_std', ...
        'ACC_NG_qmax_mean','ACC_NG_qmax_std', ...
        'Time_L1_pipeline_mean','Time_L1_pipeline_std', ...
        'Time_NG_qmax_mean','Time_NG_qmax_std', ...
        'Time_NG_fullscan_mean','Time_NG_fullscan_std' ...
    };

    %% ---------- 原有 Fig1..5（不变） ----------
    tag   = sprintf('%s_tr%.2f_out%.2f_s%d', ds, train_ratio, outlier_rate, blocksize);
    outDir= fullfile(OUTDIR, tag); if ~exist(outDir,'dir'), mkdir(outDir); end
    kx    = 1:qmax;  kL1 = round(nanmean(KNEE_L1));

    % Fig1: FVU@test vs k
    figure('Color','w'); hold on; box on;
    plot_with_band(kx, mean(FVU_TEST.L1,1), std(FVU_TEST.L1,[],1), [0.80 0.86 1.00], 'b-', 2);
    plot_with_band(kx, mean(FVU_TEST.NG,1), std(FVU_TEST.NG,[],1), [1.00 0.86 0.80], 'r--',2);
    if ~isnan(kL1), xline(kL1,':b','L1-knee'); end
    yline(1-CAP_TARGET,'k:','FVU=0.2 (CAP=0.8)');
    xlabel('q'); ylabel('FVU (TEST) = 1 - CAP'); title(sprintf('%s | FVU@test', ds));
    legend({'L1 band','L1 mean','NG band','NG mean'},'Location','NorthEast');
    saveas(gcf, fullfile(outDir,'fig1_fvu_vs_k.png'));

    % Fig2: Trust & kNN-hit
    figure('Color','w','Position',[100 100 1100 420]); tiledlayout(1,2,'Padding','compact','TileSpacing','compact');
    nexttile; hold on; box on;
    plot_with_band(kx, mean(TRUST.L1,1), std(TRUST.L1,[],1), [0.80 0.86 1.00], 'b-', 2);
    plot_with_band(kx, mean(TRUST.NG,1), std(TRUST.NG,[],1), [1.00 0.86 0.80], 'r--',2);
    if ~isnan(kL1), xline(kL1,':b'); end
    ylabel('Trustworthiness (K=10)'); xlabel('q'); title('Trustworthiness vs k');
    legend({'L1 band','L1 mean','NG band','NG mean'},'Location','SouthEast');

    nexttile; hold on; box on;
    plot_with_band(kx, mean(KNN_HIT.L1,1), std(KNN_HIT.L1,[],1), [0.80 0.86 1.00], 'b-', 2);
    plot_with_band(kx, mean(KNN_HIT.NG,1), std(KNN_HIT.NG,[],1), [1.00 0.86 0.80], 'r--',2);
    if ~isnan(kL1), xline(kL1,':b'); end
    ylabel('kNN hit-rate (K=10)'); xlabel('q'); title('kNN-hit vs k');
    saveas(gcf, fullfile(outDir,'fig2_trust_knnhit_vs_k.png'));

    % Fig3: 1-NN Acc@test vs k（含原空间）
    figure('Color','w'); hold on; grid on;
    plot(kx, mean(ACC_TEST.L1.knn,1),'b-','LineWidth',2);
    plot(kx, mean(ACC_TEST.NG.knn,1),'r--','LineWidth',2);
    if ~isnan(kL1), xline(kL1,':b','L1-knee'); end
    rawAcc_draw = raw_knn_baseline(data_tr_core_clean, labels_tr_core, data_test_clean, labels_test);
    yline(rawAcc_draw,'k:','orig 1-NN');
    xlabel('q'); ylabel('Acc@test (1-NN)'); ylim([0,1]); xlim([1 qmax]);
    title(sprintf('%s | 1-NN accuracy vs k (outlier=%.0f%%, %s)', ds, 100*outlier_rate, OUTLIER.TYPE));
    legend({'L1','NonGreedy','raw 1-NN'},'Location','SouthEast');
    saveas(gcf, fullfile(outDir,'fig3_acc_vs_k.png'));

    % Fig4: Objective vs k（带 knee）
    figure('Color','w'); hold on; box on;
    plot_with_band(kx, mean(OBJ_L1,1), std(OBJ_L1,[],1), [0.80 0.86 1.00], 'b-', 2);
    plot_with_band(kx, mean(OBJ_NG,1), std(OBJ_NG,[],1), [1.00 0.86 0.80], 'r--',2);
    if ~isnan(kL1), xline(kL1,':b','L1-knee'); end
    xlabel('q'); ylabel('Objective (train)'); title(sprintf('%s | Objective vs k', ds));
    legend({'L1 band','L1 mean','NG band','NG mean'},'Location','NorthWest');
    saveas(gcf, fullfile(outDir,'fig4_objective_vs_k.png'));

    % Fig5: CAP@test vs k
    figure('Color','w'); hold on; box on;
    plot_with_band(kx, mean(CAP_TEST.L1,1), std(CAP_TEST.L1,[],1), [0.80 0.86 1.00], 'b-', 2);
    plot_with_band(kx, mean(CAP_TEST.NG,1), std(CAP_TEST.NG,[],1), [1.00 0.86 0.80], 'r--',2);
    if ~isnan(kL1), xline(kL1,':b','L1-knee'); end
    yline(CAP_TARGET,'k--','80%');
    xlabel('q'); ylabel('CAP (TEST)'); ylim([0.6,1]); xlim([1 qmax]);
    title(sprintf('%s | CAP@test vs k', ds));
    legend({'L1 band','L1 mean','NG band','NG mean'},'Location','SouthEast');
    saveas(gcf, fullfile(outDir,'fig5_cap_vs_k.png'));

    %% ---------- Fig6 + 总表：测试集平均重构 @ L1 的 k* ----------
    idxRep = find(~isnan(KNEE_L1));
    if ~isempty(idxRep)
        % ---- 有有效膝点的情况 ----
        fvu_L1_atk   = zeros(numel(idxRep),1);
        fvu_NG_atk   = zeros(numel(idxRep),1);
        cap_L1_atk   = zeros(numel(idxRep),1);
        acc_L1_atk   = zeros(numel(idxRep),1);
        acc_NG_atk   = zeros(numel(idxRep),1);
        acc_NG_qmax  = zeros(numel(idxRep),1);
        for t=1:numel(idxRep)
            r  = idxRep(t);
            qk = max(1, min(qmax, round(KNEE_L1(r))));
            fvu_L1_atk(t)  = FVU_TEST.L1(r, qk);
            fvu_NG_atk(t)  = FVU_TEST.NG(r, qk);
            cap_L1_atk(t)  = 1 - FVU_TEST.L1(r, qk);
            acc_L1_atk(t)  = ACC_TEST.L1.knn(r, qk);
            acc_NG_atk(t)  = ACC_TEST.NG.knn(r, qk);
            acc_NG_qmax(t) = ACC_TEST.NG.knn(r, qmax);
        end
        nrm_L1_atk = sqrt(fvu_L1_atk);
        nrm_NG_atk = sqrt(fvu_NG_atk);

        time_L1_pipe  = TIME_PIPE_L1(idxRep);
        time_NG_qmax  = TIME_PIPE_NG_QMAX(idxRep);
        time_NG_full  = TIME_PIPE_NG_FULL(idxRep);

        % 画图（左：FVU；右：NRMSE）
        figure('Color','w','Position',[100 100 920 400]); tiledlayout(1,2,'Padding','compact','TileSpacing','compact');
        nexttile; hold on; box on;
        Y = [mean(fvu_L1_atk) mean(fvu_NG_atk)];
        E = [std(fvu_L1_atk)  std(fvu_NG_atk)];
        bar(1:2, Y); errorbar(1:2, Y, E,'k.','LineWidth',1);
        set(gca,'XTick',1:2,'XTickLabel',{'L1@k*','NG@k* (same k)'}); ylabel('FVU@test (lower is better)');
        title(sprintf('%s | Avg FVU at L1-k*  (repeats=%d)', ds, numel(idxRep)));

        nexttile; hold on; box on;
        Y2 = [mean(nrm_L1_atk) mean(nrm_NG_atk)];
        E2 = [std(nrm_L1_atk)  std(nrm_NG_atk)];
        bar(1:2, Y2); errorbar(1:2, Y2, E2,'k.','LineWidth',1);
        set(gca,'XTick',1:2,'XTickLabel',{'L1@k*','NG@k* (same k)'}); ylabel('NRMSE@test (= sqrt(FVU))');
        title(sprintf('%s | Avg NRMSE at L1-k*  (repeats=%d)', ds, numel(idxRep)));
        saveas(gcf, fullfile(outDir,'fig6_avg_recon_at_knee.png'));

        % ---- 新增 Fig7：测试集平均识别率 ----
        figure('Color','w','Position',[120 120 860 420]); hold on; box on;
        Yacc = [mean(acc_L1_atk), mean(acc_NG_atk), mean(acc_NG_qmax)];
        Eacc = [std(acc_L1_atk),  std(acc_NG_atk),  std(acc_NG_qmax)];
        bar(1:3, Yacc); errorbar(1:3, Yacc, Eacc,'k.','LineWidth',1);
        set(gca,'XTick',1:3,'XTickLabel',{'L1@k*','NG@k*','NG@qmax'});
        ylabel('Acc@test (1-NN)'); ylim([0,1]);
        title(sprintf('%s | Avg test accuracy at key dimensions', ds));
        saveas(gcf, fullfile(outDir,'fig7_avg_acc_keydims.png'));

        % ---- 新增 Fig8：识别流程时间 ----
        figure('Color','w','Position',[140 140 860 420]); hold on; box on;
        Ytime = [mean(time_L1_pipe), mean(time_NG_full), mean(time_NG_qmax)];
        Etime = [std(time_L1_pipe),  std(time_NG_full),  std(time_NG_qmax)];
        bar(1:3, Ytime); errorbar(1:3, Ytime, Etime,'k.','LineWidth',1);
        set(gca,'XTick',1:3,'XTickLabel',{'L1 scan+knee','NG full-scan','NG@qmax'});
        ylabel('Time (seconds)');
        title(sprintf('%s | Accuracy-pipeline time comparison', ds));
        saveas(gcf, fullfile(outDir,'fig8_time_compare_acc_pipeline.png'));

        % ---- 原 summary_all 表的一行（含 FVU+NRMSE） ----
        knee_valid   = KNEE_L1(idxRep);
        knee_dim_L1  = round(mean(knee_valid));   % L1 拐点维度（平均后取整）
        CAP_L1_knee  = mean(cap_L1_atk);          % L1 在拐点维度处的 CAP@test 平均值
        RepeatsUsed  = numel(idxRep);

        rowT = table( ...
            string(ds), ...
            train_ratio, ...
            knee_dim_L1, ...
            CAP_L1_knee, ...
            RepeatsUsed, ...
            mean(fvu_L1_atk), std(fvu_L1_atk), ...
            mean(fvu_NG_atk), std(fvu_NG_atk), ...
            mean(nrm_L1_atk), std(nrm_L1_atk), ...
            mean(nrm_NG_atk), std(nrm_NG_atk), ...
            'VariableNames', {'Dataset','TrainRatio','KneeDim_L1','CAP_L1_at_knee','RepeatsUsed', ...
                              'FVU_L1_mean','FVU_L1_std','FVU_NG_mean','FVU_NG_std', ...
                              'NRMSE_L1_mean','NRMSE_L1_std','NRMSE_NG_mean','NRMSE_NG_std'} );

        writetable(rowT, fullfile(outDir,'summary_at_knee.csv'));
        fprintf('[CSV] %s\n', fullfile(outDir,'summary_at_knee.csv'));
        summary_all = [summary_all; rowT];

        % ---- 表2（NRMSE + best-k 信息）----
        NRMSE_L1_knee_mean   = mean(nrm_L1_atk);
        NRMSE_L1_knee_std    = std(nrm_L1_atk);
        NRMSE_NG_atL1k_mean  = mean(nrm_NG_atk);
        NRMSE_NG_atL1k_std   = std(nrm_NG_atk);

        ValsN = { ...
            string(ds), train_ratio, outlier_rate, ...
            knee_dim_L1, CAP_L1_knee, RepeatsUsed, ...
            NRMSE_L1_knee_mean,  NRMSE_L1_knee_std, ...
            NRMSE_NG_atL1k_mean, NRMSE_NG_atL1k_std, ...
            k_l1_val_mean,  k_l1_val_std, ...
            nrm_l1_valBest_mean,  nrm_l1_valBest_std, ...
            k_l1_test_mean, k_l1_test_std, ...
            nrm_l1_testBest_mean, nrm_l1_testBest_std, ...
            k_ng_val_mean,  k_ng_val_std, ...
            nrm_ng_valBest_mean,  nrm_ng_valBest_std, ...
            k_ng_test_mean, k_ng_test_std, ...
            nrm_ng_testBest_mean, nrm_ng_testBest_std ...
        };
        rowN = cell2table(ValsN, 'VariableNames', VarNamesN);
        summary_all_nrmse = [summary_all_nrmse; rowN];

        % ---- 表3（识别率 + 时间）----
        ACC_L1_knee_mean  = mean(acc_L1_atk);
        ACC_L1_knee_std   = std(acc_L1_atk);
        ACC_NG_atL1k_mean = mean(acc_NG_atk);
        ACC_NG_atL1k_std  = std(acc_NG_atk);
        ACC_NG_qmax_mean  = mean(acc_NG_qmax);
        ACC_NG_qmax_std   = std(acc_NG_qmax);

        Time_L1_pipeline_mean = mean(time_L1_pipe);
        Time_L1_pipeline_std  = std(time_L1_pipe);
        Time_NG_qmax_mean     = mean(time_NG_qmax);
        Time_NG_qmax_std      = std(time_NG_qmax);
        Time_NG_fullscan_mean = mean(time_NG_full);
        Time_NG_fullscan_std  = std(time_NG_full);

        ValsA = { ...
            string(ds), train_ratio, outlier_rate, knee_dim_L1, RepeatsUsed, ...
            ACC_L1_knee_mean,  ACC_L1_knee_std, ...
            ACC_NG_atL1k_mean, ACC_NG_atL1k_std, ...
            ACC_NG_qmax_mean,  ACC_NG_qmax_std, ...
            Time_L1_pipeline_mean, Time_L1_pipeline_std, ...
            Time_NG_qmax_mean, Time_NG_qmax_std, ...
            Time_NG_fullscan_mean, Time_NG_fullscan_std ...
        };
        rowA = cell2table(ValsA, 'VariableNames', VarNamesA);
        summary_all_acc_time = [summary_all_acc_time; rowA];
        writetable(rowA, fullfile(outDir,'summary_acc_time.csv'));
        fprintf('[CSV] %s\n', fullfile(outDir,'summary_acc_time.csv'));

        % ---- 命令窗口：打印识别率与时间总结 ----
        fprintf('\n[ACC/TIME SUMMARY] %s | train=%.2f | outlier=%.2f | repeats=%d\n', ...
            ds, train_ratio, outlier_rate, RepeatsUsed);
        fprintf('  L1  @k*    : Acc = %.4f ± %.4f\n', ACC_L1_knee_mean,  ACC_L1_knee_std);
        fprintf('  NG  @k*    : Acc = %.4f ± %.4f\n', ACC_NG_atL1k_mean, ACC_NG_atL1k_std);
        fprintf('  NG  @qmax  : Acc = %.4f ± %.4f\n', ACC_NG_qmax_mean,  ACC_NG_qmax_std);
        fprintf('  Time[L1 scan+knee] = %.3f ± %.3f s\n', Time_L1_pipeline_mean, Time_L1_pipeline_std);
        fprintf('  Time[NG @qmax]     = %.3f ± %.3f s\n', Time_NG_qmax_mean,    Time_NG_qmax_std);
        fprintf('  Time[NG full-scan] = %.3f ± %.3f s\n\n', Time_NG_fullscan_mean, Time_NG_fullscan_std);

    else
        % ---- 若所有 repeat 都没有检测到 knee，则退而选 q^ = argmin_k mean(FVU_TEST.L1) ----
        [~, qhat] = min(mean(FVU_TEST.L1,1));
        fprintf('WARN: no L1-knee detected across repeats. Use q^=%d (min mean FVU) as fallback.\n', qhat);

        fvu_L1_atk  = FVU_TEST.L1(:, qhat);
        fvu_NG_atk  = FVU_TEST.NG(:, qhat);
        nrm_L1_atk  = sqrt(fvu_L1_atk);
        nrm_NG_atk  = sqrt(fvu_NG_atk);
        cap_L1_atk  = 1 - fvu_L1_atk;
        acc_L1_atk  = ACC_TEST.L1.knn(:, qhat);
        acc_NG_atk  = ACC_TEST.NG.knn(:, qhat);
        acc_NG_qmax = ACC_TEST.NG.knn(:, qmax);
        time_L1_pipe = TIME_PIPE_L1;
        time_NG_qmax = TIME_PIPE_NG_QMAX;
        time_NG_full = TIME_PIPE_NG_FULL;

        figure('Color','w','Position',[100 100 920 400]); tiledlayout(1,2,'Padding','compact','TileSpacing','compact');
        nexttile; hold on; box on;
        Y = [mean(fvu_L1_atk) mean(fvu_NG_atk)];
        E = [std(fvu_L1_atk)  std(fvu_NG_atk)];
        bar(1:2, Y); errorbar(1:2, Y, E,'k.','LineWidth',1);
        set(gca,'XTick',1:2,'XTickLabel',{'L1@q^','NG@q^'}); ylabel('FVU@test');
        title(sprintf('%s | Avg FVU at q^=%d', ds, qhat));

        nexttile; hold on; box on;
        Y2 = [mean(nrm_L1_atk) mean(nrm_NG_atk)];
        E2 = [std(nrm_L1_atk)  std(nrm_NG_atk)];
        bar(1:2, Y2); errorbar(1:2, Y2, E2,'k.','LineWidth',1);
        set(gca,'XTick',1:2,'XTickLabel',{'L1@q^','NG@q^'}); ylabel('NRMSE@test');
        title(sprintf('%s | Avg NRMSE at q^=%d', ds, qhat));
        saveas(gcf, fullfile(outDir,'fig6_avg_recon_at_qhat.png'));

        figure('Color','w','Position',[120 120 860 420]); hold on; box on;
        Yacc = [mean(acc_L1_atk), mean(acc_NG_atk), mean(acc_NG_qmax)];
        Eacc = [std(acc_L1_atk),  std(acc_NG_atk),  std(acc_NG_qmax)];
        bar(1:3, Yacc); errorbar(1:3, Yacc, Eacc,'k.','LineWidth',1);
        set(gca,'XTick',1:3,'XTickLabel',{'L1@q^','NG@q^','NG@qmax'});
        ylabel('Acc@test (1-NN)'); ylim([0,1]);
        title(sprintf('%s | Avg test accuracy at fallback q^', ds));
        saveas(gcf, fullfile(outDir,'fig7_avg_acc_keydims_qhat.png'));

        figure('Color','w','Position',[140 140 860 420]); hold on; box on;
        Ytime = [mean(time_L1_pipe), mean(time_NG_full), mean(time_NG_qmax)];
        Etime = [std(time_L1_pipe),  std(time_NG_full),  std(time_NG_qmax)];
        bar(1:3, Ytime); errorbar(1:3, Ytime, Etime,'k.','LineWidth',1);
        set(gca,'XTick',1:3,'XTickLabel',{'L1 scan+knee','NG full-scan','NG@qmax'});
        ylabel('Time (seconds)');
        title(sprintf('%s | Accuracy-pipeline time comparison', ds));
        saveas(gcf, fullfile(outDir,'fig8_time_compare_acc_pipeline_qhat.png'));

        rowT = table( ...
            string(ds), ...
            train_ratio, ...
            qhat, ...
            mean(1 - fvu_L1_atk), ...
            repeat, ...
            mean(fvu_L1_atk), std(fvu_L1_atk), ...
            mean(fvu_NG_atk), std(fvu_NG_atk), ...
            mean(nrm_L1_atk), std(nrm_L1_atk), ...
            mean(nrm_NG_atk), std(nrm_NG_atk), ...
            'VariableNames', {'Dataset','TrainRatio','KneeDim_L1','CAP_L1_at_knee','RepeatsUsed', ...
                              'FVU_L1_mean','FVU_L1_std','FVU_NG_mean','FVU_NG_std', ...
                              'NRMSE_L1_mean','NRMSE_L1_std','NRMSE_NG_mean','NRMSE_NG_std'} );

        writetable(rowT, fullfile(outDir,'summary_at_qhat.csv'));
        fprintf('[CSV] %s\n', fullfile(outDir,'summary_at_qhat.csv'));
        summary_all = [summary_all; rowT];

        knee_dim_L1 = qhat;
        CAP_L1_knee = mean(1 - fvu_L1_atk);
        RepeatsUsed = repeat;

        NRMSE_L1_knee_mean   = mean(nrm_L1_atk);
        NRMSE_L1_knee_std    = std(nrm_L1_atk);
        NRMSE_NG_atL1k_mean  = mean(nrm_NG_atk);
        NRMSE_NG_atL1k_std   = std(nrm_NG_atk);

        ValsN = { ...
            string(ds), train_ratio, outlier_rate, ...
            knee_dim_L1, CAP_L1_knee, RepeatsUsed, ...
            NRMSE_L1_knee_mean,  NRMSE_L1_knee_std, ...
            NRMSE_NG_atL1k_mean, NRMSE_NG_atL1k_std, ...
            k_l1_val_mean,  k_l1_val_std, ...
            nrm_l1_valBest_mean,  nrm_l1_valBest_std, ...
            k_l1_test_mean, k_l1_test_std, ...
            nrm_l1_testBest_mean, nrm_l1_testBest_std, ...
            k_ng_val_mean,  k_ng_val_std, ...
            nrm_ng_valBest_mean,  nrm_ng_valBest_std, ...
            k_ng_test_mean, k_ng_test_std, ...
            nrm_ng_testBest_mean, nrm_ng_testBest_std ...
        };
        rowN = cell2table(ValsN, 'VariableNames', VarNamesN);
        summary_all_nrmse = [summary_all_nrmse; rowN];

        ACC_L1_knee_mean  = mean(acc_L1_atk);
        ACC_L1_knee_std   = std(acc_L1_atk);
        ACC_NG_atL1k_mean = mean(acc_NG_atk);
        ACC_NG_atL1k_std  = std(acc_NG_atk);
        ACC_NG_qmax_mean  = mean(acc_NG_qmax);
        ACC_NG_qmax_std   = std(acc_NG_qmax);

        Time_L1_pipeline_mean = mean(time_L1_pipe);
        Time_L1_pipeline_std  = std(time_L1_pipe);
        Time_NG_qmax_mean     = mean(time_NG_qmax);
        Time_NG_qmax_std      = std(time_NG_qmax);
        Time_NG_fullscan_mean = mean(time_NG_full);
        Time_NG_fullscan_std  = std(time_NG_full);

        ValsA = { ...
            string(ds), train_ratio, outlier_rate, knee_dim_L1, RepeatsUsed, ...
            ACC_L1_knee_mean,  ACC_L1_knee_std, ...
            ACC_NG_atL1k_mean, ACC_NG_atL1k_std, ...
            ACC_NG_qmax_mean,  ACC_NG_qmax_std, ...
            Time_L1_pipeline_mean, Time_L1_pipeline_std, ...
            Time_NG_qmax_mean, Time_NG_qmax_std, ...
            Time_NG_fullscan_mean, Time_NG_fullscan_std ...
        };
        rowA = cell2table(ValsA, 'VariableNames', VarNamesA);
        summary_all_acc_time = [summary_all_acc_time; rowA];
        writetable(rowA, fullfile(outDir,'summary_acc_time.csv'));
        fprintf('[CSV] %s\n', fullfile(outDir,'summary_acc_time.csv'));

        fprintf('\n[ACC/TIME SUMMARY] %s | train=%.2f | outlier=%.2f | repeats=%d\n', ...
            ds, train_ratio, outlier_rate, RepeatsUsed);
        fprintf('  L1  @q^    : Acc = %.4f ± %.4f\n', ACC_L1_knee_mean,  ACC_L1_knee_std);
        fprintf('  NG  @q^    : Acc = %.4f ± %.4f\n', ACC_NG_atL1k_mean, ACC_NG_atL1k_std);
        fprintf('  NG  @qmax  : Acc = %.4f ± %.4f\n', ACC_NG_qmax_mean,  ACC_NG_qmax_std);
        fprintf('  Time[L1 scan+knee] = %.3f ± %.3f s\n', Time_L1_pipeline_mean, Time_L1_pipeline_std);
        fprintf('  Time[NG @qmax]     = %.3f ± %.3f s\n', Time_NG_qmax_mean,    Time_NG_qmax_std);
        fprintf('  Time[NG full-scan] = %.3f ± %.3f s\n\n', Time_NG_fullscan_mean, Time_NG_fullscan_std);
    end
end
end
end
end

%% ---------- 写出所有数据集的总表 ----------
summary_path1 = fullfile(OUTDIR,'summary_all_at_knee.csv');
writetable(summary_all, summary_path1);
fprintf('===> DONE. Summary saved at: %s\n', summary_path1);

summary_path2 = fullfile(OUTDIR,'summary_all_nrmse_ngbest.csv');
writetable(summary_all_nrmse, summary_path2);
fprintf('===> DONE. NG/L1-best NRMSE summary saved at: %s\n', summary_path2);

summary_path3 = fullfile(OUTDIR,'summary_all_acc_time.csv');
writetable(summary_all_acc_time, summary_path3);
fprintf('===> DONE. ACC/TIME summary saved at: %s\n', summary_path3);

fprintf('===> Figures & per-setting summaries at: %s\n', OUTDIR);

%% ---------- 趋势图（保留你的结构 + [MOD] 同时画 L1 best） ----------
T = readtable(summary_path2);

datasets = {'auslan','bci','japan'};
outliers = [0 0.2 0.4 0.6]; %#ok<NASGU>

figure('Color','w','Position',[100 100 900 300]);
tiledlayout(1,3,'Padding','compact','TileSpacing','compact');

for di = 1:3
    ds = datasets{di};
    idx = strcmpi(T.Dataset, ds);
    Tds = T(idx,:);

    % 保证按 Outlier 排序
    [~, order] = sort(Tds.OutlierRate);
    Tds = Tds(order,:);

    o = Tds.OutlierRate;
    nL1  = Tds.NRMSE_L1_knee_mean;
    nNGk = Tds.NRMSE_NG_atL1k_mean;
    nNGb = Tds.NRMSE_NG_valBest_mean;  % 或 testBest

    % [MOD] L1 best-k 结果也画出来
    nL1_valBest  = Tds.NRMSE_L1_valBest_mean;
    nL1_testBest = Tds.NRMSE_L1_testBest_mean;

    nexttile; hold on; box on;
    plot(o, nL1,        '-o','LineWidth',1.8);
    plot(o, nNGk,       '--s','LineWidth',1.8);
    plot(o, nNGb,       '-.^','LineWidth',1.8);
    plot(o, nL1_valBest,':d','LineWidth',1.6);
    plot(o, nL1_testBest,'-.x','LineWidth',1.6);

    xlabel('Outlier rate');
    ylabel('NRMSE@test');
    title(upper(ds));
    if di==1
        legend({'L1@k*','NG@k*','NG@best-k(val)','L1@best-k(val)','L1@best-k(test)'},'Location','NorthWest');
    end
end

%% ======================== Helper functions ========================

function y = yn(b), y = "N"; if b, y="Y"; end, y=char(y); end

function blocksize = get_dataset_blocksize(ds)
% [MOD] 按数据集指定固定异常块边长
switch lower(ds)
    case 'auslan'
        blocksize = 10;
    case 'bci'
        blocksize = 13;
    case 'japan'
        blocksize = 5;
    otherwise
        blocksize = 13;   % 其他数据集默认仍使用 13
end
end

function [data, labels, qmax] = load_dataset(ds)
switch lower(ds)
    case 'bci'
        load('bci.mat');            data = X_org;         labels = X_label_org; qmax = 20;
    case 'auslan'
        load('auslan.mat');         data = X_org;         labels = label_org;   qmax = 22;
    case 'japan'
        load('japan.mat');          data = X_org;         labels = label_org;   qmax = 7;
    case 'yale'
        load('yaledata_norm_64x64.mat'); load('yalelabel.mat'); data = X_org; labels = label;   qmax = 62;
    case 'pems'
        load('PEMSsorttrans.mat');  data = datatruncated; labels = label;       qmax = 66;
    otherwise
        error('Unknown dataset: %s', ds);
end
end

function [data_train, labels_train, data_test, labels_test] = stratified_split(data, labels, train_ratio)
labels = labels(:); cls = unique(labels); idx_tr=[]; idx_te=[];
for ci=1:numel(cls)
    idx_c = find(labels==cls(ci)); n_c = numel(idx_c); n_tr = max(1, round(train_ratio*n_c));
    rp = randperm(n_c);
    idx_tr = [idx_tr; idx_c(rp(1:n_tr))]; %#ok<AGROW>
    idx_te = [idx_te; idx_c(rp(n_tr+1:end))];
end
idx_tr = sort(idx_tr); idx_te = sort(idx_te);
data_train = data(:,:,idx_tr); labels_train = labels(idx_tr);
data_test  = data(:,:,idx_te); labels_test  = labels(idx_te);
end

function [data_tr_core, labels_tr_core, data_val, labels_val] = split_val(data_train, labels_train, val_ratio)
Ntr = size(data_train,3); n_val = max(1, round(val_ratio*Ntr));
rp = randperm(Ntr); idx_val = rp(1:n_val); idx_core = rp(n_val+1:end);
data_tr_core = data_train(:,:,idx_core); labels_tr_core = labels_train(idx_core);
data_val     = data_train(:,:,idx_val);  labels_val     = labels_train(idx_val);
end

function data_out = inject_outliers_block(data_in, outlier_rate, O)
[dr, dc, N] = size(data_in);
data_out = data_in;

% ===================== [MOD] outlier_rate=0 时不污染 =====================
n_out = round(outlier_rate * N);
if n_out <= 0
    fprintf('  -> CORE corrupted: 0/%d samples (outlier_rate=0)\n', N);
    return;
end
% ========================================================================

idx = randperm(N, n_out);

% ===================== [MOD] 优先使用固定块大小 =====================
if isfield(O,'BLOCK_SIZE') && ~isempty(O.BLOCK_SIZE) && O.BLOCK_SIZE > 0
    h = min(dr, O.BLOCK_SIZE);
    w = min(dc, O.BLOCK_SIZE);
else
    h = max(1, floor(O.BLOCK_RATIO_R * dr));
    w = max(1, floor(O.BLOCK_RATIO_C * dc));
end
% ===================================================================

for ii = idx
    X = data_out(:,:,ii);
    r1 = randi([1, max(1, dr-h+1)]);
    c1 = randi([1, max(1, dc-w+1)]);
    blk = X(r1:r1+h-1, c1:c1+w-1);

    lo = double(min(X(:)));
    hi = double(max(X(:)));
    mu = mean(double(X(:)));
    s  = std(double(X(:)));

    if hi==lo, hi=lo+1; end
    if s==0, s=1; end

    density = O.DENSITY_INBLOCK;
    switch lower(O.TYPE)
        case 'salt_and_pepper'
            mask = rand(h,w) < density;
            salt = rand(h,w) < 0.5;
            blk(mask & salt)  = hi;
            blk(mask & ~salt) = lo;

        case 'impulse'
            k = 3; span = 2;
            up_lo = mu + k*s;            up_hi = mu + (k+span)*s;
            dn_hi = mu - k*s;            dn_lo = mu - (k+span)*s;
            mask = rand(h,w) < density;  side = rand(h,w) < 0.5;
            hi_mask = mask & side;       lo_mask = mask & ~side;
            up_rand = up_lo + (up_hi - up_lo).*rand(h,w);
            dn_rand = dn_lo + (dn_hi - dn_lo).*rand(h,w);
            blk(hi_mask) = up_rand(hi_mask);
            blk(lo_mask) = dn_rand(lo_mask);

        case 'uniform'
            gamma = 0.30;
            rngspan = (hi-lo);
            ext_hi = hi + gamma*rngspan;
            ext_lo = lo - gamma*rngspan;
            mask = rand(h,w) < density;
            side = rand(h,w) < 0.5;
            hi_mask = mask & side;
            lo_mask = mask & ~side;
            up_rand = hi + (ext_hi - hi).*rand(h,w);
            dn_rand = ext_lo + (lo - ext_lo).*rand(h,w);
            blk(hi_mask) = up_rand(hi_mask);
            blk(lo_mask) = dn_rand(lo_mask);

        otherwise
            error('Unknown outlier type %s', O.TYPE);
    end

    X(r1:r1+h-1, c1:c1+w-1) = blk;
    data_out(:,:,ii) = X;
end

fprintf('  -> CORE corrupted: %d/%d samples, type=%s, block=%dx%d, pix-density=%.2f\n', ...
    numel(idx), N, O.TYPE, h, w, O.DENSITY_INBLOCK);
end

function acc = knn_acc(Ftr, ytr, Fte, yte)
mdl = fitcknn(Ftr, ytr, 'NumNeighbors', 1);
acc = mean(predict(mdl, Fte) == yte);
end

function acc = raw_knn_baseline(data_tr_core, ytr, data_te, yte)
[dr,dc,Ntr] = size(data_tr_core); [~,~,Nte] = size(data_te);
Xtr = reshape(permute(data_tr_core,[3,1,2]), Ntr, dr*dc);
Xte = reshape(permute(data_te,     [3,1,2]), Nte, dr*dc);
mdl = fitcknn(Xtr, ytr, 'NumNeighbors', 1);
acc = mean(predict(mdl, Xte)==yte);
end

function [feat_tr, feat_te, rec_te] = extract_feats_and_recon(model, data_tr, data_te)
[~,~,Ntr] = size(data_tr); [~,~,Nte] = size(data_te);
k = size(model.U,2);
feat_tr = zeros(Ntr, k*k);
for n=1:Ntr
    Y = model.V' * data_tr(:,:,n) * model.U;
    feat_tr(n,:) = Y(:).';
end
feat_te = zeros(Nte, k*k); rec_te = zeros(Nte,1);
for n=1:Nte
    Y = model.V' * data_te(:,:,n) * model.U;
    feat_te(n,:) = Y(:).';
    Xh = model.V * Y * model.U';
    rec_te(n) = norm(data_te(:,:,n) - Xh, 'fro');
end
end

function Jmean = mean_obj_abs(data3d, model)
[~,~,N] = size(data3d); s=0;
for n=1:N
    Y = model.V' * data3d(:,:,n) * model.U;
    s = s + sum(abs(Y(:)));
end
Jmean = s / N;
end

function [knee, info] = knee_voted_with_strength(J, smooth_w, deriv_thr_ratio)
J = J(:)'; x = 1:numel(J);
Jn = (J - min(J)) / (max(J)-min(J) + eps);
Jn = moving_smooth(Jn, smooth_w);
yline = linspace(Jn(1), Jn(end), numel(Jn));
[~, k1] = max(Jn - yline);

sse2 = inf(1,numel(Jn));
for t=2:numel(Jn)-1
    p1 = polyfit(x(1:t), Jn(1:t), 1);
    p2 = polyfit(x(t:end), Jn(t:end), 1);
    y1 = polyval(p1,x(1:t));   e1 = y1 - Jn(1:t);
    y2 = polyval(p2,x(t:end)); e2 = y2 - Jn(t:end);
    sse2(t) = sum(e1.^2) + sum(e2.^2) + 1e-3*t;
end
[bestSSE2, k2] = min(sse2); %#ok<NASGU>

p_all = polyfit(x, Jn, 1);
yfit_all = polyval(p_all,x);
sse1 = sum((yfit_all-Jn).^2);
deltaSSE = max(0,(sse1-bestSSE2)/(sse1+eps));

dy = diff(Jn);
dy = moving_smooth(dy, smooth_w);
thr = deriv_thr_ratio*mean(dy);
k3 = find(dy<thr,1,'first');
if isempty(k3), k3=numel(Jn); end

knee = round(median([k1,k2,k3]));

maxDist        = max(Jn - yline);
info.score     = 0.5*deltaSSE + 0.5*maxDist;
info.deltaSSE  = deltaSSE;
info.maxDist   = maxDist;
info.drop_ratio= (max(dy)-min(dy)) / (abs(max(dy))+eps);
ssres = sum((Jn - yfit_all).^2);
sstot = sum((Jn-mean(Jn)).^2) + eps;
info.R2_global = 1 - ssres/sstot;
d2 = diff(Jn,2);
info.curv_msd = mean(d2.^2);
end

function y = moving_smooth(x, w)
if nargin<2, w=3; end
k=floor(w/2); y=x;
for i=1:numel(x)
    lo=max(1,i-k);
    hi=min(numel(x),i+k);
    y(i)=mean(x(lo:hi));
end
end

function t = trustworthiness_score(X_high, X_low, K)
Dhi = squareform(pdist(X_high));
Dlo = squareform(pdist(X_low));
[~, rHi] = sort(Dhi, 2, 'ascend');
[~, rLo] = sort(Dlo, 2, 'ascend');
N = size(X_high,1);
Tsum = 0;
for i=1:N
    Nh = rHi(i, 2:K+1);
    Nl = rLo(i, 2:K+1);
    U  = setdiff(Nl, Nh);
    if ~isempty(U)
        ranks = arrayfun(@(u) find(rHi(i,:)==u,1,'first')-1, U);
        Tsum  = Tsum + sum(ranks - K);
    end
end
C = 2 / (N*K*(2*N - 3*K - 1));
t = 1 - C*Tsum;
t = max(0,min(1,t));
end

function h = knn_hit_rate(X_high, X_low, K)
Dhi = squareform(pdist(X_high));
Dlo = squareform(pdist(X_low));
[~, rHi] = sort(Dhi, 2, 'ascend');
[~, rLo] = sort(Dlo,  2, 'ascend');
N = size(X_high,1);
hits = 0;
for i=1:N
    Nh = rHi(i, 2:K+1);
    Nl = rLo(i, 2:K+1);
    hits = hits + numel(intersect(Nh, Nl));
end
h = hits / (N*K);
end

function s = num2str_or_nan(v)
if isnan(v), s='NaN'; else, s=num2str(v); end
end

function v = pick_at_k(vec, k)
if isnan(k)
    v = NaN;
    return;
end
kk = max(1, min(numel(vec), round(k)));
v = vec(kk);
end

function plot_with_band(x, mu, sd, faceRGB, lineSpec, lw)
upper = mu + sd;
lower = mu - sd;
fill([x, fliplr(x)], [upper, fliplr(lower)], faceRGB, ...
    'EdgeColor','none','FaceAlpha',0.25); hold on;
plot(x, mu, lineSpec, 'LineWidth', lw);
end