%% clear
clear all, close all, clc
%% load and train
load dataset_sintetico.mat
X = data.features;
y = data.label;
c = unique(y);

% show classes:
figure
hold on
plot(X(y==c(1),1), X(y==c(1),2),'r.', 'markersize',20) % label c=1
plot(X(y==c(2),1), X(y==c(2),2),'b.', 'markersize',20)
xlabel('Feature 1','fontsize',18)
ylabel('Feature 2','fontsize',18)
title('dataset')

% train with SVM (no kernel specified):
SVM = fitcsvm(X, y);
sv = SVM.SupportVectors;
plot(sv(:, 1), sv(:, 2), 'ko', 'markersize',20)
title('dataset and SV (no kernel spec.ed)')
disp('too many SVs')

%% plot SVM (radial)

% show classes:
figure
hold on
plot(X(y==c(1),1), X(y==c(1),2),'r.', 'markersize',20) % plot samples for label c=1
plot(X(y==c(2),1), X(y==c(2),2),'b.', 'markersize',20)
xlabel('Feature 1','fontsize',18)
ylabel('Feature 2','fontsize',18)

% train SVM (rbf kernel! (ball))
SVM = fitcsvm(X, y, 'KernelFunction','rbf');
sv = SVM.SupportVectors;
plot(sv(:, 1), sv(:, 2), 'ko', 'markersize',20)
disp('lower # of SVs')

% meshgrid:
[X_f1,X_f2] = meshgrid(min(X(:,1)):.1:max(X(:,1)),min(X(:,2)):.1:max(X(:,2)));
grid= [X_f1(:),X_f2(:)];
grid_c= predict(SVM,grid); % predicted classes by SVM(kernel:rbf)

d1= scatter(grid(grid_c==1,1),grid(grid_c==1,2),'r.');
hold on
d2= scatter(grid(grid_c==2,1),grid(grid_c==2,2),'b.');

%%
disp('--')