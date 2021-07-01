%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       Pablo Laso Mielgo        %
%       Ensemble Learning        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%% clear
clear all, close all, clc;
%% load data
load carsmall.mat 

Horsepower(isnan(Horsepower)) = mean(Horsepower(~isnan(Horsepower)));   % nans >> mean(non-nans)

features = [Acceleration Cylinders Displacement Horsepower Weight];  
label = double(categorical(Model_Year));   

classes = unique(label); 

figure   
colors = hsv(length(classes));  % colors for classes

scatter3(features(:, 1), features(:, 3), features(:, 4), 200, colors(label, :), '.')  % feature 1, 3 e 4 
xlabel('Acceleration', 'fontsize',18)  
ylabel('Displacement', 'fontsize',18)
zlabel('Horsepower', 'fontsize',18)
title('dataset')

%% Naive Bayes

Nfold = 3; 
cv5f_idx = cvpartition(label, 'kfold',Nfold); 

CM_NB = zeros(length(classes)); 
reliabilities_NB = nan(length(label), length(classes)); 
for i = 1:Nfold    % each FOLD
    features_tr = features(cv5f_idx.training(i), :);  
    label_tr = label(cv5f_idx.training(i));    
    features_te = features(cv5f_idx.test(i), :);   
    label_te = label(cv5f_idx.test(i));     
    
    [features_tr, mu, sigma] = zscore(features_tr);  
    features_te = (features_te - mu)./sigma; 
    
    NB = fitcnb(features_tr, label_tr);   
    
    [predictions, reliabilities_NB(cv5f_idx.test(i), :)] = predict(NB, features_te);  
    CM_NB = CM_NB + confusionmat(label_te, predictions, 'order',classes)';  
end
disp('--NB--')
CM_NB
acc = trace(CM_NB)/sum(CM_NB(:)) 

%% KNN
Nfold = 3;
cv5f_idx = cvpartition(label, 'kfold',Nfold);

CM_kNN = zeros(length(classes));
reliabilities_kNN = nan(length(label), length(classes));
for i = 1:Nfold
    features_tr = features(cv5f_idx.training(i), :);
    label_tr = label(cv5f_idx.training(i));
    features_te = features(cv5f_idx.test(i), :);
    label_te = label(cv5f_idx.test(i));
    
    [features_tr, mu, sigma] = zscore(features_tr);
    features_te = (features_te - mu)./sigma;
    
    kNN = fitcknn(features_tr, label_tr, 'numn',3); 
    [predictions, reliabilities_kNN(cv5f_idx.test(i), :)] = predict(kNN, features_te);
    
    CM_kNN = CM_kNN + confusionmat(label_te, predictions, 'order',classes)';
end
disp('--KNN--')
CM_kNN
acc = trace(CM_kNN)/sum(CM_kNN(:))

%% Random Forest 
% Classificazione singolo esperto (RF)
Nfold = 3;
cv5f_idx = cvpartition(label, 'kfold',Nfold);

CM_RF = zeros(length(classes));
reliabilities_RF = nan(length(label), length(classes));
for i = 1:Nfold
    features_tr = features(cv5f_idx.training(i), :);
    label_tr = label(cv5f_idx.training(i));
    features_te = features(cv5f_idx.test(i), :);
    label_te = label(cv5f_idx.test(i));
    
    [features_tr, mu, sigma] = zscore(features_tr);
    features_te = (features_te - mu)./sigma;
    
    RF = TreeBagger(100, features_tr, label_tr); %100 trees
    [predictions_str, reliabilities_RF(cv5f_idx.test(i), :)] = predict(RF, features_te);
    predictions = str2double(predictions_str);
    
    CM_RF = CM_RF + confusionmat(label_te, predictions, 'order',classes)';
end
disp('--RF--')
CM_RF
acc = trace(CM_RF)/sum(CM_RF(:))

%% Reliability
reliabilities_mex = cat(3, reliabilities_NB, reliabilities_kNN, reliabilities_RF);

reliabilities_mex_per_class = permute(reliabilities_mex, [1 3 2]);   

image(reliabilities_mex_per_class) 

set(gca, 'xtick',1:3, 'xticklabel',{'NB', 'kNN', 'RF'}, 'fontsize',18)  % mostro solo i valori 1, 2 e 3 sull'asse delle x sostituendo i numeri con il nome delle classi a dimensione 18
ylabel('Campioni', 'fontsize',18)   % scrivo il nome dell'asse delle y a dimensione 18
title('R: classe 1 - G: classe 2 - B: classe 3', 'fontsize',18)     % scrivo il titolo con l'associazione delle affidabilità delle classi ed i colori fondamentali a dimensione 18


