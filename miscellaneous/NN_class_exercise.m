%% more examples on PC: C:\Program Files\MATLAB\R2019a\examples\nnet
%% load data
digitDatasetPath= fullfile(matlabroot,'toolbox','nnet','nndemos','nndatasets','DigitDataset');
digitData= imageDatastore(digitDatasetPath,'IncludeSubfolders',true,'LabelSource','foldernames');

figure;
perm= randperm(1000,20);
for i=1:20
    subplot(4,5,i);
    imshow(digitData.Files{perm(i)});
end

img=readimage(digitData,1);
size(img)

trainingNumFiles=750;
rng(1) %reproducitability in randomization
[trainDigitData,testDigitData]=splitEachLabel(digitData,trainingNumFiles,'randomize');

layers= [imageInputLayer([28 28 1])... % pixels for image: 1 because one color (rgb-->3)
    convolution2dLayer(5,20)... % 5x5 dimension for our mask/kernel // 20 filters --> feasure maps 
    reluLayer ... % relu
    maxPooling2dLayer(2,'Stride',2)... %mask 2x2 // stride: #columns (2)
    fullyConnectedLayer(10)... % 10 outputs for 10 classes
    softmaxLayer...
    classificationLayer()];

options= trainingOptions('sgdm','MaxEpochs',15,'InitialLearnRate',0.0001);

convnet= trainNetwork(trainDigitData,layers,options);

YTest= classify(convnet,testDigitData);
TTest= testDigitData.Labels;
accuracy= sum(YTest==TTest)/numel(TTest);
%%
disp('END')
%% results in cmd:
% |========================================================================================|
% |  Epoch  |  Iteration  |  Time Elapsed  |  Mini-batch  |  Mini-batch  |  Base Learning  |
% |         |             |   (hh:mm:ss)   |   Accuracy   |     Loss     |      Rate       |
% |========================================================================================|
% |       1 |           1 |       00:00:03 |        9.38% |      13.8418 |      1.0000e-04 |
% |       1 |          50 |       00:00:17 |       56.25% |       2.8707 |      1.0000e-04 |
% |       2 |         100 |       00:00:23 |       82.81% |       0.8151 |      1.0000e-04 |
% |       3 |         150 |       00:00:28 |       85.94% |       0.6059 |      1.0000e-04 |
% |       4 |         200 |       00:00:34 |       86.72% |       0.4217 |      1.0000e-04 |
% |       5 |         250 |       00:00:39 |       90.63% |       0.1973 |      1.0000e-04 |
% |       6 |         300 |       00:00:45 |       91.41% |       0.3867 |      1.0000e-04 |
% |       7 |         350 |       00:00:50 |       96.88% |       0.1088 |      1.0000e-04 |
% |       7 |         400 |       00:00:55 |       95.31% |       0.1552 |      1.0000e-04 |
% |       8 |         450 |       00:01:00 |       95.31% |       0.1358 |      1.0000e-04 |
% |       9 |         500 |       00:01:06 |       97.66% |       0.0943 |      1.0000e-04 |
% |      10 |         550 |       00:01:11 |       97.66% |       0.0858 |      1.0000e-04 |
% |      11 |         600 |       00:01:16 |       99.22% |       0.0412 |      1.0000e-04 |
% |      12 |         650 |       00:01:21 |       97.66% |       0.0610 |      1.0000e-04 |
% |      13 |         700 |       00:01:26 |       98.44% |       0.0510 |      1.0000e-04 |
% |      13 |         750 |       00:01:31 |      100.00% |       0.0423 |      1.0000e-04 |
% |      14 |         800 |       00:01:36 |       99.22% |       0.0187 |      1.0000e-04 |
% |      15 |         850 |       00:01:42 |      100.00% |       0.0171 |      1.0000e-04 |
% |      15 |         870 |       00:01:44 |      100.00% |       0.0228 |      1.0000e-04 |
% |========================================================================================|