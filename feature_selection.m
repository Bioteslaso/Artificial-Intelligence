%% empty
clear all,close all, clc
%% generate gaussian-distributed dataset
F_c1 = mvnrnd([4 3], [.4 1.2], 300);    
F_c2 = mvnrnd([2 5], [1.5 .5], 200);  

cla    
hold on
plot(F_c1(:, 1), F_c1(:, 2), 'r.', 'markersize',10)
plot(F_c2(:, 1), F_c2(:, 2), 'b.', 'markersize',10)
axis equal

xlabel('feature 1')    
ylabel('feature 2')
title('scatter plot: 2 l.i. variables')

%% steps

% add thrid feature:
F_c1 = [F_c1 1.5*F_c1(:, 1) + 0.8*F_c1(:, 2)];
F_c2 = [F_c2 1.5*F_c2(:, 1) + 0.8*F_c2(:, 2)];

figure    
hold on   
plot3(F_c1(:, 1), F_c1(:, 2), F_c1(:, 3), 'r.', 'markersize',10) 
plot3(F_c2(:, 1), F_c2(:, 2), F_c2(:, 3), 'b.', 'markersize',10) 
axis equal    
view(3)     

xlabel('feature 1')  
ylabel('feature 2')    
zlabel('feature 3')
title('3plot (thrid, l.d. variable)')

% create dataset (add labels):
F = [F_c1;F_c2];
L = [ones([300 1]);2*ones([200 1])]; 
data = [F L];

% corrupt feature 2:
data = data(randperm(size(data, 1)), :);
Nnan = round(0.2*size(data, 1)); % 20% smaples as invalid 
% random vector--> range:[1-size(data)] integers | size: Nnan
nan_idx = randi([1 size(data, 1)], [Nnan 1]);   
nan_idx = unique(nan_idx, 'stable'); % erase duplicated indexes
data(nan_idx, 2) = nan; % introduce Nans

% scale-corrupting:
scale = 5; 
data(:, 1) = scale*data(:, 1);

% plot
figure    
hold on    
c1_map = (data(:, end) == 1); 
c2_map = (data(:, end) == 2);
plot3(data(c1_map, 1), data(c1_map, 2), data(c1_map, 3), 'r.', 'markersize',10)     % plotto i campioni di classe 1 selezionandoli dal dataset con la relativa mappa
plot3(data(c2_map, 1), data(c2_map, 2), data(c2_map, 3), 'b.', 'markersize',10)     % plotto allo stesso modo i campioni di classe 2
axis equal 
view(3)

xlabel('feature 1') 
ylabel('feature 2') 
zlabel('feature 3')  
title('feature 1 scaled')

%% cleaning dataset

% -- substitute NaNs by the mean (class-depend.):
F = data(:, 1:end - 1); 
L = data(:, end);
C = unique(data(:, end));

F_nan_idx = find(any(isnan(F), 1)); % find nan-containing columns

for i = 1:length(F_nan_idx)
    for j = 1:length(C) % for each CLASS:
        C_j_map = (L == C(j)); % map for current-for-loop class
        nan_map = isnan(F(:, F_nan_idx(i))); % search for nans in colum reported      
        C_j_nan_map = and(nan_map, C_j_map); % only nans in current-for-loop class     
        C_j_not_nan_map = and(~nan_map, C_j_map); % ~nans and current-for-loop class

        mean_ij = mean(F(C_j_not_nan_map, F_nan_idx(i))); % mean from non-nans

        F(C_j_nan_map, F_nan_idx(i)) = mean_ij; % apply on nans in F
    end
end

% -- substitute Nans by KNN values (instance-depend.):
F = data(:, 1:end - 1);
L = data(:, end); 

obs_nan_map = any(isnan(F), 2); % nans map
    % features/labels | nans/non-nans :
F_nan = F(obs_nan_map, :); 
F_not_nan = F(~obs_nan_map, :);
L_nan = L(obs_nan_map); 
L_not_nan = L(~obs_nan_map);

for i = 1:size(F_nan, 1) % for all nan features (~90 samples)
    feat_nan_map = isnan(F_nan(i, :)); %nan column
%non-nan list for the class of the current-for-loop sample's LABEL:
    F_not_nan_C = F_not_nan(L_not_nan == L_nan(i), :); 
%distance between all non-nan smaples (in nan column) and current-for-loop nan sample:
    nan_not_nan_C_dist = dist(F_not_nan_C(:, ~feat_nan_map), F_nan(i, ~feat_nan_map)');
    [~, sort_pos] = sort(nan_not_nan_C_dist); % sort and get idxs
%substitute each nan by mean (computed from the first 5-NN)
    F_nan(i, feat_nan_map) = mean(F_not_nan_C(sort_pos(1:5), feat_nan_map), 1);
end

F(obs_nan_map, :)= F_nan; % replace in F
data(:, 1:end - 1)= F;


% -- normalize: 
F = data(:, 1:end - 1);   
F = (F - min(F, [], 1))./(max(F, [], 1) - min(F, [], 1)); 

% -- standardize:
F = data(:, 1:end - 1);
F = (F - mean(F, 1))./std(F, 1)
data(:, 1:end - 1) = F;  

%% Feature selection

% -- correlation:
F = data(:, 1:end - 1);
F_corr_abs = abs(corr(F));  % correlation matrix

figure 
heatmap(F_corr_abs) % show graphically 
title('Correlation Matrix')

F_corr_sum = sum(F_corr_abs, 1)/size(F, 2); % sum columns (symmetric) 
F = F(:, F_corr_sum <= .7); 

% -- pair-wise (Wilcoxon method):
F = data(:, 1:end - 1); 
L = data(:, end);  

W = nan(1, size(F, 2)); 
for i = 1:size(F, 2) 
    obs_C1_map = (L == 1); % c1 map (idxs)
    obs_C2_map = (L == 2);
    
    W(i) = ranksum(F(obs_C1_map, i), F(obs_C2_map, i));
end
F = F(:, W <= 0.05); 

% -- Relief:
F = data(:, 1:end - 1); 
L = data(:, end);

[W_rank, W] = relieff(F, L, 3);  

figure
bar(W(W_rank))
hold on 
th = 0.005;
plot([0 4], [th th], 'r', 'linewidth',3)
axis(axis + [0 0 0 .1*th]) 
title('Relieff method')
F = F(:, W >= th); 

% -- PCA
F = data(:, 1:end - 1);
[a_vet, ~, a_val] = pca(F, 'algorithm','eig', 'economy','off')

figure 
plot(0:length(a_val), [0;cumsum(a_val)/sum(a_val)], 'o-')
xlabel('#eigenvalue')
ylabel('eigen-value')
title('PCA')
F = F(:, a_vet(:, 1) >= 0.5);

F = data(:, 1:end - 1);
scores = F*a_vet; 

figure  
plot(scores(:, 1), scores(:, 2), '*m')  
title('PCA scores')


% -- Wrapper:
random_sort = randperm(size(data, 1)); % random idxs
F = data(random_sort, 1:end - 1); % random idxs to data
L = data(random_sort, end);
C = unique(L); % classes

tr_idx = 1:450; % training until #450
te_idx = tr_idx(end) + 1:length(L); % 450+[]
F_tr = F(tr_idx, :);  
F_te = F(te_idx, :);
L_tr = L(tr_idx);   
L_te = L(te_idx);  

acc_firsts_i = [];   
for i = 1:size(F, 2) 
    classificatore = fitcnb (F_tr(:, 1:i), L_tr); %Naive Bayes Classifier
    predictions = predict(classificatore, F_te(:, 1:i));  
    CM = confusionmat(L_te, predictions, 'order',C);    % Confusion Matrix 
    acc = trace(CM)/sum(CM(:)); 

    acc_firsts_i = cat(2, acc_firsts_i, acc);
end

figure   
plot(acc_firsts_i, '.-') 

acc_diff = diff(acc_firsts_i); % derivative of accuracy vector
acc_diff = [0 acc_diff]; 

hold on     
plot(acc_diff, 'r.-')  
legend('acc','acc_{deriv}')
title('accuracy')

F = F(:, 1:find(acc_diff < 0, 1) - 1); 

if isempty(F) 
    F = data(:, 1:end - 1);  
end

%%
disp('END') 