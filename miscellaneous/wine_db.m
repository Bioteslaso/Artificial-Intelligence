%%clear
clear all, close all, clc
%% load
DB= load('wine_db.mat');
%% extract data
db= cell2mat(DB.dataset(2:end,:)); % take data values only
db= db(:,[1 6 end]); 

test_id= [3 16 24 57 59 83 105 147 160 166];
db_training= db(setdiff(1:size(db,1),test_id),:);
db_test= db(test_id,:);

features_tr= db_training(:,1:end-1);
label_tr= db_training(:,end);

features_te= db_test(:,1:end-1);
label_te= db_test(:,end);

classes=unique(db(:,end));
%% plot
gscatter(features_tr(:, 1), features_tr(:, 2), label_tr)
xlabel(DB.dataset(1, 1), 'fontsize',18)
ylabel(DB.dataset(1, 6), 'fontsize',18)
zlabel('Priors*Likelihood', 'fontsize',18)
title('training set (raw values)')

% normalize
features_tr_min= min(features_tr,[],1);
features_tr_max= max(features_tr,[],1);

features_tr= (features_tr-features_tr_min)./(features_tr_max-features_tr_min);

% plot normalized values:
figure
gscatter(features_tr(:, 1), features_tr(:, 2), label_tr)
hold on
% find gaussian:
BYS = fitcnb(features_tr, label_tr);
BYS_params=cell2mat(BYS.DistributionParameters); % column = class
BYS_means = BYS_params([1 3 5], :);
BYS_stds = BYS_params([2 4 6], :);

Fnval=100;
xy_lim = [min(features_tr(:, 1)) max(features_tr(:, 1)) min(features_tr(:, 2)) max(features_tr(:, 2))];
[F1, F2] = meshgrid(linspace(xy_lim(1), xy_lim(2), Fnval), linspace(xy_lim(3), xy_lim(4), Fnval));
colors = {'r', 'g', 'b'};

% Gaussian distribution:
for i = 1:length(classes) 
    prior_i = (sum(label_tr == i)/length(label_tr));    % a priori
    mvnpdf_class = prior_i*mvnpdf(cat(2, F1(:), F2(:)), BYS_means(i, :), BYS_stds(i, :));
    surf(F1, F2, .3 + reshape(mvnpdf_class, Fnval, Fnval), 'edgecolor',colors{i}, 'facecolor','none')
end
xlabel(DB.dataset(1, 1), 'fontsize',18)
ylabel(DB.dataset(1, 6), 'fontsize',18)
zlabel('Priors*Likelihood', 'fontsize',18)
title('Gaussian Distribution')
legend({'wine 1', 'wine 2', 'wine 3'})
view(3)
%%
disp('END')