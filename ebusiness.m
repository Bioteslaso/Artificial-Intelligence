%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Pablo Laso Mielgo  %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% empty:
close all, clear all, clc
set(0,'DefaultFigureWindowStyle','docked')
%% load data:

data = readtable('Ecommerce_Customers.csv', 'readvariablenames',1); % first line as variables
data.Properties

%% Analyze (scatterhist(), pairplot()):
% data(1:5, :) 
data = data(:, 4:end); % take numerical data only

figure
scatterhist(data.Time_on_App, data.Time_on_Website)  % scatterplot at both sides
xlabel('Time_on_App', 'fontsize',18, 'interpreter','none')                                                    
ylabel('Time_on_Website', 'fontsize',18, 'interpreter','none')  
title('scatterhist') 

figure   
scatterhist(data.Time_on_App, data.Yearly_Amount_Spent)     
xlabel('Time_on_App', 'fontsize',18, 'interpreter','none')    
ylabel('Yearly_Amount_Spent', 'fontsize',18, 'interpreter','none')
title('scatterhist')

figure
data_names = data.Properties.VariableNames; 
label_names = num2cell(num2str(ones(size(data, 1), 1)));
pairplot(data.Variables, data_names, label_names)
title('pairplot')

%% Regression

% observations (randomize)
Nobs = size(data, 1); 
obs_idx = 1:Nobs; 
obs_idx = obs_idx(randperm(Nobs)); 

% features and labels
data_names = data.Properties.VariableNames; 
label_map = ismember(data_names, 'Yearly_Amount_Spent'); % choose True boolean 1 if...
vars = data.Variables; %take all variables
features = vars(obs_idx, ~label_map); %take all except label_map (Yearly_Amount_Spent)
label = vars(obs_idx, label_map);

% training and hold-out testing set
Nobs_te = round(0.4*Nobs);     
feature_tr = features(Nobs_te + 1:end, :);
label_tr = label(Nobs_te + 1:end); 
feature_te = features(1:Nobs_te, :);
label_te = label(1:Nobs_te);

% training
LRmodel = fitlm(feature_tr, label_tr); 
LRmodel_coeff = LRmodel.Coefficients; 
LRmodel_coeff(:, 1:2) 

% predcitions (label clienti)
predictions= predict(LRmodel, feature_te); 
error= (label_te - predictions); 
MAE= sum(abs(error))/length(error) %  (Mean Absolute Error)
MSE= sum((error).^2)/length(error) %  (Mean Squared Error)
RMSE= sqrt(MSE) % (Root Mean Squared Error)

% Plot error
figure  
histogram(error, 50) 
title('histogram')

%%
disp('END')