%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       Pablo Laso Mielgo        %
%     Neuronal    Networks       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%% clear
% clear all, close all, clc;
nnet.guis.closeAllViews(), close all, clear all, clc
%% params
% w = [3 -1];
% b = -2;
% func = 'tansig'
% p = [2 3]; % input
% v = p*w' + b; % z
% y = tansig(v); % y € (-1,1)
% 
% % input value mesh
% span = -1:.01:2;
% [P1,P2] = meshgrid(span,span);
% pp = [P1(:) P2(:)]';
% % NEURON
% v = w*pp + repmat(b, [1, size(pp,2)]);
% y = tansig(v);
% % PLOT
% figure(1)
% m = mesh(P1,P2,reshape(y(1,:),length(span),length(span)));
% set(m,'facecolor',[1 0.2 .7],'linestyle','none');
% view(3)

%% Neruonal Network
% create NN:
net = network( ...
1, ... %numInputs, number of inputs
2, ... %numLayers, number of layers
[1; 0], ... %biasConnect, numLayers-by-1 Boolean vector
[1; 0], ... %inputConnect, numLayers-by-numInputs Boolean matrix
[0 0; 1 0], ... % layerConnect, numLayers-by-numLayers Boolean matrix
[0 1] ... % outputConnect, 1-by-numLayers Boolean vector
);
view(net) 
% number of hidden layer neurons
net.layers{1}.size = 4;
net.layers{1}.transferFcn = 'tansig'; % hidden layer transfer function
view(net);
% input vectors
p = rand(5,2); % just to make the following steps possible
target = [1 -1 ;-1 1];
net = configure(net, p, target);
view(net); 
% initial response without training
initial_output = net(p);
net.trainFcn = 'trainlm';
net.performFcn = 'mse';
net = train(net,p,target);
final_output = net(p);
%% Perceptron:
p = [0 0 1 1; 0 1 0 1];
t = [0 0 0 1];
plotpv(p,t);
hold on
net = perceptron;
net.layers{1}
net = train(net,p,t);
net.iw{1,1}
net.b{1}
plotpc(net.iw{1,1},net.b{1})
z = sim(net, p)
%% Perceptron to classify linearly-separable cluster:
N = 30; % number of samples of each class
offset = 6; % offset for second class
x = [randn(2,N) randn(2,N)+offset]; % inputs
y = [zeros(1,N) ones(1,N)]; %
figure(1)
plotpv(x,y); hold on
net = perceptron;
net = train(net,x,y);
view(net)
% decision boundary
figure,plotpc(net.IW{1},net.b{1});
t = randn(2,20) + offset/2;
t_output= net(t);
figure(1), plotpv(t,t_output)
plotpc(net.IW{1},net.b{1});
%% Perceptron to classify four linearly-separable cluster:
K = 40; % number of samples of each class
% define classes
q = .6; % offset of classes (quadrants)
A = [rand(1,K)-q; rand(1,K)+q];
B = [rand(1,K)+q; rand(1,K)+q];
C = [rand(1,K)+q; rand(1,K)-q];
D = [rand(1,K)-q; rand(1,K)-q];
% define output coding for classes
a = [0 1]';
b = [1 1]';
c = [1 0]';
d = [0 0]';
% Random order for inputs
P = [A B C D];
T = [repmat(a,1,length(A)) repmat(b,1,length(B)) ...
 repmat(c,1,length(C)) repmat(d,1,length(D)) ];
ordering = randperm(size(P,2));
P = P(:,ordering);
T = T(:,ordering);
net = perceptron;
net = train(net, P, T);
figure,plotpc(net.IW{1},net.b{1});
y = net([-0.3; 0]);
span = -1:.01:2;
[P1,P2] = meshgrid(span,span);
pp = [P1(:) P2(:)]';
y = net(pp);
figure(1)
ee = sum(y);
ee = ee.*2;
ee1 = find(y(1,:) == 0 & y(2,:) == 1);
ee2 = find(y(1,:) == 1 & y(2,:) == 0);
ee(:,ee1) = 2;
ee(:,ee2) = 3;
m = mesh(P1,P2,reshape(ee,length(span),length(span)));
view(2)
%% 10 fold cross validation - Perceptron to classify two linearly-separable cluster:
kfold = 10;
% number of samples of each class
N = 200;
% define classes
offset = 6; % offset for second class
x = [randn(2,N) randn(2,N)+offset]; % inputs
y = [zeros(1,N) ones(1,N)]; % outputs
% plot classes with text labels for classes
plotpv(x,y);
% Random order for inputs
ordering = randperm(size(x,2));
x = x(:,ordering);
y = y(:,ordering);
% divide into training and test set
Indices = crossvalind('Kfold', size(x,2), kfold);
% Learning and testing
confusion_matrix = zeros(2);
for i = 1:kfold
    P_test = x(:,Indices == i);
    P_training =x(:,Indices ~= i);
    T_test = y(:,Indices == i);
    T_training = y(:,Indices ~= i);
    %normalize data (mapminmax in [-1 1] or mapstd)
%     [P_training,PS(i).PS] = mapminmax(P_training,-1,1);
%     P_test = mapminmax('apply',P_test,PS(i).PS);

    [P_training,PS(i).PS] = mapstd(P_training,0,1);
    P_test = mapstd('apply',P_test,PS(i).PS);

    % train the network
    net(i).net = perceptron;
    net(i).net.divideParam.trainRatio = 1;
    net(i).net.divideParam.valRatio = 0;
    net(i).net.divideParam.testRatio = 0;
    net(i).net = train(net(i).net, P_training, T_training);

    output(i).y = net(i).net(P_test);

    %performance estimation
    CP(i).CP = classperf(T_test,output(i).y );
    confusion_matrix = confusion_matrix + CP(i).CP.DiagnosticTable;
end
average_accuracy = sum(diag(confusion_matrix))/sum(confusion_matrix(:));
