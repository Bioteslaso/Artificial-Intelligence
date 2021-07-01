%% three similar -yet independent- pieces fo code
%% empty
clear all, close all, clc
%%
% load:
dataset = load('fisheriris.mat');
features = dataset.meas;

features = features(:, [2 3 4]);
cla
plot3(features(:, 1), features(:, 2), features(:, 3), 'k.', 'markersize',10)
xlabel('Feature 2')
ylabel('Feature 3')
zlabel('Feature 4')
axis equal

% #93
hold on
obs_idx = 93;
plot3(features(obs_idx, 1), features(obs_idx, 2), features(obs_idx, 3), 'rd', 'markersize',10, 'linewidth',2)

% distance 
Nobs = size(features, 1);
dist_93_all = nan(Nobs, 1);
for i = 1:Nobs
    dist_93_all(i) = sum((features(obs_idx, :) - features(i, :)).^2);   % sqrt() non necessaria perché la misura è relativa
end

% order
[~, dist_93_all_srt_idx] = sort(dist_93_all);
dist_93_all_srt_idx = dist_93_all_srt_idx(2:end);   % elimino il primo (0) perché è la distanza con se stesso

% KNN
plot3(features(dist_93_all_srt_idx(1), 1), features(dist_93_all_srt_idx(1), 2), features(dist_93_all_srt_idx(1), 3), 'go', 'markersize',10, 'linewidth',2)
plot3(features(dist_93_all_srt_idx(2), 1), features(dist_93_all_srt_idx(2), 2), features(dist_93_all_srt_idx(2), 3), 'bo', 'markersize',10, 'linewidth',2)
plot3(features(dist_93_all_srt_idx(3), 1), features(dist_93_all_srt_idx(3), 2), features(dist_93_all_srt_idx(3), 3), 'o', 'markersize',10, 'linewidth',2, 'color',[1 .5 0])

%%

% load
dataset = load('fisheriris.mat');
features = dataset.meas;
labels = dataset.species;

% features 1, 4 --> number:
features = features(:, [1 4]);

labels = categorical(labels);
labels = double(labels);

% binary:
labels(labels == 2) = 1;
labels(labels == 3) = 2;
classes = unique(labels);

% validation set
validation_set_idx = [51 53 106 108 118 119 123 126 130 131 132 136];
features_validation_set = features(validation_set_idx, :);
labels_validation_set = labels(validation_set_idx);

% non-validation set:
features = features(setdiff(1:length(labels), validation_set_idx), :);
labels = labels(setdiff(1:length(labels), validation_set_idx));

% plot:
cla
hold on
plot(features(labels == 1, 1), features(labels == 1, 2), 'r.', 'markersize',10)
plot(features(labels == 2, 1), features(labels == 2, 2), 'g.', 'markersize',10)
xlabel('Feature 1')
ylabel('Feature 4')
axis equal

% % new samples:
p1 = [6.5 2.4];
p2 = [6.5 1.4];
p3 = [4.8 0.6];

% distance:
Nobs = size(features, 1);
dist_p1_all = nan(Nobs, 1);
dist_p2_all = nan(Nobs, 1);
dist_p3_all = nan(Nobs, 1);
for i = 1:Nobs
    dist_p1_all(i, :) = sum((p1 - features(i, :)).^2);
    dist_p2_all(i, :) = sum((p2 - features(i, :)).^2);
    dist_p3_all(i, :) = sum((p3 - features(i, :)).^2);
end

% order: 
[dist_p1_all_srt, dist_p1_all_srt_idx] = sort(dist_p1_all);
[dist_p2_all_srt, dist_p2_all_srt_idx] = sort(dist_p2_all);
[dist_p3_all_srt, dist_p3_all_srt_idx] = sort(dist_p3_all);

% plot new samples and class:
class_colors = {'r', 'g'};

p1_label = classes(labels(dist_p1_all_srt_idx(1)));
plot(p1(1), p1(2), 'k.', 'markersize',10)
plot(p1(1), p1(2), [class_colors{p1_label} 'o'], 'markersize',10, 'linewidth',2)  

p2_label = classes(labels(dist_p2_all_srt_idx(1)));
plot(p2(1), p2(2), 'k.', 'markersize',10)
plot(p2(1), p2(2), [class_colors{p2_label} 'o'], 'markersize',10, 'linewidth',2)

p3_label = classes(labels(dist_p3_all_srt_idx(1)));
plot(p3(1), p3(2), 'k.', 'markersize',10)
plot(p3(1), p3(2), [class_colors{p3_label} 'o'], 'markersize',10, 'linewidth',2)

% % Plot validation set:
for i = 1:size(features_validation_set, 1)
    plot(features_validation_set(i, 1), features_validation_set(i, 2), [class_colors{labels_validation_set(i)} '*'], 'markersize',10, 'linewidth',2)
end

% distance:
dist_p1_all_ref = inf(size(features_validation_set, 1), 1);
dist_p2_all_ref = inf(size(features_validation_set, 1), 1);
dist_p3_all_ref = inf(size(features_validation_set, 1), 1);
for i = 1:size(features_validation_set, 1)
    if p1_label == labels_validation_set(i)
        dist_p1_all_ref(i, :) = sum((p1 - features_validation_set(i, :)).^2);   % sqrt() non necessaria perché la misura è relativa
    end
    if p2_label == labels_validation_set(i)
        dist_p2_all_ref(i, :) = sum((p2 - features_validation_set(i, :)).^2);
    end
    if p3_label == labels_validation_set(i)
        dist_p3_all_ref(i, :) = sum((p3 - features_validation_set(i, :)).^2);
    end
end

% % distances: "win" e "max":

% p1
labels_dist_p1_sorted = labels(dist_p1_all_srt_idx);

Owin_p1 = dist_p1_all_srt(1);
O2win_p1 = dist_p1_all_srt(find(labels_dist_p1_sorted ~= labels_dist_p1_sorted(1), 1));

dist_p1_all_ref = sort(dist_p1_all_ref);
Omax_p1 = dist_p1_all_ref(1);

% p2
labels_dist_p2_sorted = labels(dist_p2_all_srt_idx);

Owin_p2 = dist_p2_all_srt(1);
O2win_p2 = dist_p2_all_srt(find(labels_dist_p2_sorted ~= labels_dist_p2_sorted(1), 1));

dist_p2_all_ref = sort(dist_p2_all_ref);
Omax_p2 = dist_p2_all_ref(1);

% p3
labels_dist_p3_sorted = labels(dist_p3_all_srt_idx);

Owin_p3 = dist_p3_all_srt(1);
O2win_p3 = dist_p3_all_srt(find(labels_dist_p3_sorted ~= labels_dist_p3_sorted(1), 1));

dist_p3_all_ref = sort(dist_p3_all_ref);
Omax_p3 = dist_p3_all_ref(1);

% reliability:

%p1
psi_a_p1 = max(Owin_p1/Omax_p1, 0)
psi_b_p1 = 1 - Owin_p1/O2win_p1
psi_p1 = min(psi_a_p1, psi_b_p1)

%p2
psi_a_p2 = max(Owin_p2/Omax_p2, 0)
psi_b_p2 = 1 - Owin_p2/O2win_p2
psi_p2 = min(psi_a_p2, psi_b_p2)

%p3
psi_a_p3 = max(Owin_p3/Omax_p3, 0)
psi_b_p3 = 1 - Owin_p3/O2win_p3
psi_p3 = min(psi_a_p3, psi_b_p3)

%%

%load:
dataset = load('fisheriris.mat');

features = dataset.meas(:, [3 4]);
labels = double(categorical(dataset.species));
classes = unique(labels);

% % outer 10-fold Cross-Validation and inner 3-fold Cross-Validation:
Nfold = 10;
NSfold = 3;
to_test_k = 1:2:5;

% outer:
best_k_f10 = nan(Nfold, 1);
kf10_idx = cvpartition(length(labels), 'kfold', Nfold);
for i = 1:Nfold
        
    % outer training:
    features_tr = features(kf10_idx.training(i), :);
    labels_tr = labels(kf10_idx.training(i), :);
    
    % % inner training:
    acc_f3 = nan(NSfold, 1);
    skf3_of_kf10_i_idx = cvpartition(labels_tr, 'kfold', NSfold); 
    for j = 1:NSfold
        
        % sets:
        features_s_tr = features_tr(skf3_of_kf10_i_idx.training(j), :);
        labels_s_tr = labels_tr(skf3_of_kf10_i_idx.training(j), :);

        features_s_te = features_tr(skf3_of_kf10_i_idx.test(j), :);
        labels_s_te = labels_tr(skf3_of_kf10_i_idx.test(j), :);
        
        % classify: 
        kNN_addestrato = fitcknn(features_s_tr, labels_s_tr, 'NumNeighbors',to_test_k(j)); 
        predictions_s_te = predict(kNN_addestrato, features_s_te);

        % CM: 
        CM_j = confusionmat(labels_s_te, predictions_s_te, 'order',classes)';
        acc_f3(j) = trace(CM_j)/sum(CM_j(:));
    end
    
    % select best k:
    [~, best_acc_f3_idx] = max(acc_f3);
    best_k_f10(i) = to_test_k(best_acc_f3_idx);
end

% histogram:
best_k_hist = zeros(1, length(to_test_k));
for i = 1:length(best_k_f10)
    k_in_fold_val = best_k_f10(i);
    best_k_hist(to_test_k == k_in_fold_val) = best_k_hist(to_test_k == k_in_fold_val) + 1;
end
bar(best_k_hist)
set(gca, 'xticklabel',to_test_k, 'fontsize',18)
