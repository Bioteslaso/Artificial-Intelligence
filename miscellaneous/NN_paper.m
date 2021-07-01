%% PDF: NN-MATLAB_examples // online: http://lab.fs.uni-lj.si/lasin/wp/IMIT_files/neural/NN-examples.pdf
%%
close all, clear all, clc, format compact
% Neuron weights
w = [4 -2]
% Neuron bias
b = -3
% Activation function
 func = 'tansig'
% func = 'purelin'
% func = 'hardlim'
% func = 'logsig'
%%
p = [2 3]

activation_potential = p*w'+b
neuron_output = feval(func, activation_potential)


[p1,p2] = meshgrid(-10:.25:10);
z = feval(func, [p1(:) p2(:)]*w'+b );
z = reshape(z,length(p1),length(p2));
plot3(p1,p2,z)
grid on
xlabel('Input 1')
ylabel('Input 2')
zlabel('Neuron output')
%%
close all, clear all, clc, format compact
inputs = [1:6]' % input vector (6-dimensional pattern)
outputs = [1 2]' % corresponding target output vector
%%
% create network
net = network( ...
1, ... % numInputs, number of inputs (vectors),
2, ... % numLayers, number of layers (hidden layers)
[1; 0], ... % b % biasConnect, numLayers-by-1 Boolean vector,
[1; 0], ... % input to w % inputConnect, numLayers-by-numInputs Boolean matrix,
[0 0; 1 0], ... % layerConnect, numLayers-by-numLayers Boolean matrix --> [layer connects first layer?; layer connects second layer; ...
[0 1] ... % outputConnect, 1-by-numLayers Boolean vector
);
% View network structure
view(net);
%% Define topology and transfer function
% number of hidden layer neurons
net.layers{1}.size = 5;
% hidden layer transfer function
net.layers{1}.transferFcn = 'logsig';
view(net);
%% Configure network
net = configure(net,inputs,outputs);
view(net);
%% Train net and calculate neuron output
% initial network response without training
initial_output = net(inputs)
% network training
net.trainFcn = 'trainlm';
net.performFcn = 'mse';
net = train(net,inputs,outputs);
% network response after training
final_output = net(inputs)
%%
disp('END')