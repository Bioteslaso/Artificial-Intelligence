%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Pablo Laso Mielgo  %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% empty:
clear all, close all, clc
set(0,'DefaultFigureWindowStyle','docked')
%% Create Simple Deep Learning Network for Classification

digitDatasetPath=fullfile(matlabroot,'toolbox','nnet',...
    'nndemos', 'nndatasets','DigitDataset');

digitData=imageDatastore(digitDatasetPath,'IncludeSubfolders',...
    true,'LabelSource','foldernames');

figure; % plot some pics
perm = randperm(10000,20);
for i = 1:20
    subplot(4,5,i);
    imshow(digitData.Files{perm(i)});
end
title('example pictures')

CountLabel = digitData.countEachLabel; % #pics per class

img= readimage(digitData,1);	

size(img)

trainingNumFiles = 750; % training >> 0.75
rng(1) % For reproducibility
[trainDigitData,testDigitData]=splitEachLabel(digitData,trainingNumFiles,'randomize');

% convolutional neural network architecture:
layers = [imageInputLayer([28 28 1]) % input image pixels
convolution2dLayer(5,20) % 5-sized kernel / #filters (#neurons connected to the same region of the output)
reluLayer % convolutional layer followed by non-linear activation function
maxPooling2dLayer(2,'Stride',2) % returns 2 max values / 2-sized strides
fullyConnectedLayer(10) % output neurons (ex: the 10 possible digits)
softmaxLayer % activation function for classification
classificationLayer()]; %   return probs by softmax
% training opts:
options = trainingOptions('sgdm','MaxEpochs',15, 'InitialLearnRate',0.0001);
% training:
convnet = trainNetwork(trainDigitData,layers,options);
% classify:
YTest = classify(convnet,testDigitData);
TTest = testDigitData.Labels;
accuracy = sum(YTest == TTest)/numel(TTest)

%% Transfer	Learning and Fine-Tuning of Convolutional Neural	Networks
load(fullfile(matlabroot,'examples','nnet','LettersClassificationNet.mat'))
net.Layers
% training/test datasets:
[trainDigitData,testDigitData] = splitEachLabel(digitData,0.5,'randomize');
% pics
numImages = numel(trainDigitData.Files);
idx = randperm(numImages,20);
for i = 1:20 subplot(4,5,i)
I = readimage(trainDigitData, idx(i));
imshow(I)
end
% transfer:
layersTransfer = net.Layers(1:end-3);
% load:
digitDatasetPath=fullfile(matlabroot,'toolbox','nnet','nndemos', 'nndatasets','DigitDataset');
digitData=imageDatastore(digitDatasetPath,'IncludeSubfolders',true,'LabelSource','foldernames');
% add fully-connected layer
numClasses = numel(categories(trainDigitData.Labels))
layers = [ layersTransfer
fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,...
    'BiasLearnRateFactor',20)
softmaxLayer
classificationLayer];
%opts:
optionsTransfer=trainingOptions('sgdm','MaxEpochs',5,'InitialLearnRate',0.0001);
% 
netTransfer = trainNetwork(trainDigitData,layers,optionsTransfer);
YPred =	classify(netTransfer,testDigitData);	
YTest =	testDigitData.Labels;	
accuracy	=	sum(YPred==YTest)/numel(YTest)
%pics:
idx = 501:500:5000;
figure
for i = 1:numel(idx)
subplot(3,3,i)
I = readimage(testDigitData, idx(i));
label = char(YTest(idx(i)));
imshow(I)
title(label)
end
% GPU:
deviceInfo = gpuDevice; % Check the GPU compute capability
computeCapability = str2double(deviceInfo.ComputeCapability);
assert(computeCapability >= 3.0, 'This example requires a GPU device with compute capability 3.0 or higher.')