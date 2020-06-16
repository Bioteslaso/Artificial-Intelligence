%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       Pablo Laso Mielgo        %
%     Decomposition matrix       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%% clear
clear all, close all, clc;
%% load data
load fisheriris.mat    % dataset
features = meas(:, 3:4); 

label = double(categorical(species)); 

classes = unique(label); 

%% OVA decomposition (One-Against-All)
OVA_label = struct; 
for i = 1:length(classes)   % CLASS
    class_i_map = label == i;   % map for current-for-loop class
    
    label_Ci = label;  
    label_Ci(class_i_map) = 1; 
    label_Ci(~class_i_map) = 0;  % make naught all other non-current-for-loop classes >> 0

    OVA_label.(['C' num2str(i)]) = label_Ci; 
end

% PLOT 
figure 
gscatter(features(:, 1), features(:, 2), label)
xlabel('Larghezza petali', 'fontsize',18) 
ylabel('Lunghezza petali', 'fontsize',18) 
title('multi-class problem')

% OVA-decmposed dataset (each class):
colors = [0 0 0;1 0 0;0 1 0;0 0 1];  

figure   
subplot(3, 1, 1)  
hold on 

s1 = scatter(features(~~OVA_label.C1, 1), features(~~OVA_label.C1, 2), 300, colors(2, :), '.');
s0 = scatter(features(~OVA_label.C1, 1), features(~OVA_label.C1, 2), 300, colors(1, :), '.');   
legend([s1;s0], {'Classe 1', 'Classi 2 e 3'}, 'location','northwest', 'fontsize',16)    % scrivo la legenda relativa solo ai plot "s1" ed "s0" posizionandola in alto a
                                                                                        %   sinistra (nord-ovest) ed imposto la dimensione dei caratteri a 16
subplot(3, 1, 2)    % second plot
hold on
s1 = scatter(features(~~OVA_label.C2, 1), features(~~OVA_label.C2, 2), 300, colors(3, :), '.'); % ~~ = logical()
s0 = scatter(features(~OVA_label.C2, 1), features(~OVA_label.C2, 2), 300, colors(1, :), '.'); % ~ = NOT
legend([s1;s0], {'Classe 2', 'Classi 1 e 3'}, 'location','northwest', 'fontsize',16)
ylabel('Lunghezza petali', 'fontsize',18)   % aggiungo (solo a questo subplot) la label delle delle y (essendo la stessa per tutti i plot la scrivo solo al centrale)

subplot(3, 1, 3)    % mi focalizzo sul terzo plot della matrice di subplot 3x1 (l'ultimo)
hold on
s1 = scatter(features(~~OVA_label.C3, 1), features(~~OVA_label.C3, 2), 300, colors(4, :), '.');
s0 = scatter(features(~OVA_label.C3, 1), features(~OVA_label.C3, 2), 300, colors(1, :), '.');
legend([s1;s0], {'Classe 3', 'Classi 1 e 2'}, 'location','northwest', 'fontsize',16)
xlabel('Larghezza petali', 'fontsize',18)   % aggiungo (solo a questo subplot) la label delle delle x (essendo la stessa per tutti la scrivo solo a quello più in basso)
title('into binary problems with OVA')

% 10 fold cross-validation OVA classifiication
OVA_reliabilities = struct;
for i = 1:length(classes)  % for CLASS
    label_i = OVA_label.(['C' num2str(i)]);

    Nfold = 10; % 10-fold

    s10f_cv_idx = cvpartition(label_i, 'kfold',Nfold); % partition

    OVA_reliabilities.(['C' num2str(i)]) = nan(length(label_i), 1);  
    for j = 1:Nfold   
        feature_tr = features(s10f_cv_idx.training(j), :);  
        label_tr = label_i(s10f_cv_idx.training(j));    
        feature_te = features(s10f_cv_idx.test(j), :);
        label_te = label_i(s10f_cv_idx.test(j));  

        SVM = fitcsvm(feature_tr, label_tr, 'KernelFunction','rbf');  
        [~, reliabilities] = predict(SVM, feature_te);  
        
        OVA_reliabilities.(['C' num2str(i)])(s10f_cv_idx.test(j)) = reliabilities(:, 2);  
    end
end

% OVA predictions
all_reliabilities = nan(length(label), length(classes));
for i = 1:length(classes) 
    all_reliabilities(:, i) = OVA_reliabilities.(['C' num2str(i)]);     
end
[~, predictions] = max(all_reliabilities, [], 2);  % max! >> best prediction

% OVA performance:
CM = confusionmat(label, predictions, 'order',classes)'  
acc = trace(CM)/sum(CM(:)) 
precision = diag(CM)./sum(CM, 2) 
recall = diag(CM)'./sum(CM, 1) 
 
%% OVO decomposition (One-Versus-One)

classes_paris = nchoosek(classes, 2) % nchoosek(V,K) >> all combinations in vector V >> K-column matrix

OVO_features = struct;  
OVO_label = struct; 
for i = 1:size(classes_paris, 1)
    C1 = classes_paris(i, 1); 
    C2 = classes_paris(i, 2);  

    class_i0_map = label == C1;  
    class_i1_map = label == C2;  

    OVO_features.(['C' num2str(C1) 'C' num2str(C2)]) = features(class_i0_map | class_i1_map, :);  % logic OR >> class 0 OR class 1
    OVO_label.(['C' num2str(C1) 'C' num2str(C2)]) = label(class_i0_map | class_i1_map) - min([C1 C2]);

end

% Plot original dataset
figure
gscatter(features(:, 1), features(:, 2), label)
xlabel('Larghezza petali', 'fontsize',18)
ylabel('Lunghezza petali', 'fontsize',18)
title('multiclass')

% Plot OVO-decomposed dataset
colors = [1 0 0;0 1 0;0 0 1];   % costruisco come prima la matrice dei colori senza il nero perché l'etichetta "tutte le altre classi" in questo caso non c'è

figure
subplot(2, 2, 1)    % costruisco una matrice 2x2 di subplot (per combinare graficamente le coppie di classi OVO) e mi focalizzo sul primo plot (in alto a sinistra)
hold on
F01 = OVO_features.C1C2;    % copio (per comodità) in "F01" la sotto-matrice delle feature dell'i-esima iterazione
L01 = OVO_label.C1C2;     % copio in "L01" il sotto-vettore delle label
scatter(F01(~L01, 1), F01(~L01, 2), 300, colors(classes_paris(1, 1), :), '.');
scatter(F01(~~L01, 1), F01(~~L01, 2), 300, colors(classes_paris(1, 2), :), '.');
ylabel('C1', 'fontsize',18)     % scrivo la label delle y in (in questo plot) per avere "C1" in alto a sinistra della prima riga della matrice dei subplot

subplot(2, 2, 2)
hold on
F01 = OVO_features.C1C3;
L01 = OVO_label.C1C3;
scatter(F01(~L01, 1), F01(~L01, 2), 300, colors(classes_paris(2, 1), :), '.');
scatter(F01(~~L01, 1), F01(~~L01, 2), 300, colors(classes_paris(2, 2), :), '.');

subplot(2, 2, 3)
set(gca, 'color','none', 'xtick',[], 'ytick',[])
xlabel('C2', 'fontsize',18)     % scrivo la label delle x in (in questo plot) per avere "C2" in basso a sinistra della seconda riga della matrice dei subplot
ylabel('C2', 'fontsize',18)

subplot(2, 2, 4)
hold on
F01 = OVO_features.C2C3;
L01 = OVO_label.C2C3;
scatter(F01(~L01, 1), F01(~L01, 2), 300, colors(classes_paris(3, 1), :), '.');
scatter(F01(~~L01, 1), F01(~~L01, 2), 300, colors(classes_paris(3, 2), :), '.');
xlabel('C3', 'fontsize',18)
title('binary problems with OVO')

% SVM classification
% Classificazione OVO in 10 fold cross-validation
OVO_reliabilities = struct;     % inizializzo la struttura "OVO_reliabilities"
for i = 1:length(classes)   % per ogni classe
    iC0C1 = ['C' num2str(classes_paris(i, 1)) 'C' num2str(classes_paris(i, 2))]; % costruisco in "iC0C1" il nome del campo della coppia di classi dell'iterazione i-esima
    
    features_i01 = OVO_features.(iC0C1);    % copio (per comodità) in "features_i01" la sotto-matrice di feature dell'i-esima decomposizione OVO
    label_i01 = OVO_label.(iC0C1);      % copio in "label_i01" il sotto-vettore delle label
    
    Nfold = 10;
    s10f_cv_idx = cvpartition(label_i01, 'kfold',Nfold);
    OVO_reliabilities.(iC0C1) = nan(length(label_i01), 2);  % inizializzo con NaN la matrice di tutte le affidabilità in output dell'iesima decomposizione OVO
    for j = 1:Nfold    % per ogni fold
        feature_tr = features_i01(s10f_cv_idx.training(j), :);
        label_tr = label_i01(s10f_cv_idx.training(j));
        feature_te = features_i01(s10f_cv_idx.test(j), :);
        label_te = label_i01(s10f_cv_idx.test(j));
        
        SVM = fitcsvm(feature_tr, label_tr, 'KernelFunction','rbf');
        [~, reliabilities] = predict(SVM, feature_te);
        OVO_reliabilities.(iC0C1)(s10f_cv_idx.test(j), :) = reliabilities;  % scrivo nelle righe indicate dalla j-esima fold di test entrambe le clonne delle affidabilità
    end
end

% max reliabilities:
all_reliabilities = zeros(length(label), length(classes), size(classes_paris, 1));  % inzializzo a 0 una matrice 3D riassuntiva con tante righe quanti campioni, tante colonne
                                                                                    %   quante classi e tanti strati quante righe della matrice di coppie di decomposizione
                                                                                    %   OVO (quindi pari al numero di classificationi effettuate)

for i = 1:size(classes_paris, 1) 
    C1 = classes_paris(i, 1);
    C2 = classes_paris(i, 2);    

    class_i0_map = label == C1;
    class_i1_map = label == C2;
    
    reliabilities_nonneg = max(OVO_reliabilities.(['C' num2str(C1) 'C' num2str(C2)]), 0);   % visto che il classificatore (SVM) restituisce le affidabilità nelle due colonne
    all_reliabilities(class_i0_map | class_i1_map, classes_paris(i, :), i) = reliabilities_nonneg;  % scrivo i valori della matrice "reliabilities_nonneg" nella matrice
end

predictions_per_class = max(all_reliabilities, [], 3);
[~, predictions] = max(predictions_per_class, [], 2); 

%performance:
CM = confusionmat(label, predictions, 'order',classes)'
acc = trace(CM)/sum(CM(:))
precision = diag(CM)./sum(CM, 2)
recall = diag(CM)'./sum(CM, 1)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 