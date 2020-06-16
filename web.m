%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Pablo Laso Mielgo  %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% empty:
clear all, close all, clc
set(0,'DefaultFigureWindowStyle','docked')
%% load data:
data = readtable('Advertising.csv'); 
data.Properties
% data(1:5, :) 

%% Plot
figure 
histogram(data.Age) 
xlabel('age')

figure % gropued by the to-be-predicted column
scatterhist(data.Age, data.Area_Income, 'group',data.Clicked_on_Ad)
xlabel('Age', 'fontsize',18, 'interpreter','none')
ylabel('Area_Income', 'fontsize',18, 'interpreter','none') 

figure 
scatterhist(data.Daily_Internet_Usage, data.Daily_Time_Spent_on_Site, 'group',data.Clicked_on_Ad, 'kernel','on')
xlabel('Daily_Internet_Usage', 'fontsize',18, 'interpreter','none')      % scrivo il nome della variabile x, a dimensione 18 e senza interpretare il testo
ylabel('Daily_Time_Spent_on_Site', 'fontsize',18, 'interpreter','none')      % scrivo il nome della variabile y

%% Multinomial logistic regression values
% extract data:
data_names= data.Properties.VariableNames; % all features
data= data(:,ismember(data_names, {'Daily_Time_Spent_on_Site', 'Age', 'Area_Income', 'Daily_Internet_Usage', 'Male' 'Clicked_on_Ad'}));

Nobs= size(data,1);
obs_idx= 1:Nobs;
obs_idx= obs_idx(randperm(Nobs));

data_names = data.Properties.VariableNames; % only the features we want (6/10) 
label_map= ismember(data_names, 'Clicked_on_Ad');

vars=data.Variables; 
features= vars(obs_idx, ~label_map);
labels= vars(obs_idx, label_map);

Nobs_te= round(0.3*Nobs);
feature_tr= features(Nobs_te+1:end,:);
label_tr= labels(Nobs_te+1:end,:);
feature_te= features(1:Nobs_te,:);
label_te= labels(1:Nobs_te,:);

% training Logistic Classifier:
B= mnrfit(feature_tr,label_tr+1); % +1 to make labels different from 0.

% predict:
predictions_prob= mnrval(B, feature_te); 
[~, predictions] = max(predictions_prob, [], 2); % column 1 --> probability to class 1 // column 2 --> probability to class 2

% evaluate:
CM= confusionmat(label_te, predictions - 1, 'order',unique(label_te))' % -1 to correct previous: +1 to make labels different from 0.
acc= trace(CM)/sum(CM(:)) 


%%
disp('--')