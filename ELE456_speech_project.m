clear; clc; close all;

rng(1,'twister');

%% PATH
trainFolder = 'TrainData';
valFolder   = 'ValData';

%% LOAD DATA
imdsTrain = imageDatastore(trainFolder,...
    "IncludeSubfolders",true,...
    "LabelSource","foldernames");

imdsVal = imageDatastore(valFolder,...
    "IncludeSubfolders",true,...
    "LabelSource","foldernames");

disp(countEachLabel(imdsTrain))

classNames = categories(imdsTrain.Labels);
numClasses = numel(classNames);

inputSize = [98 50 1];

%% TASK 1 - BASELINE CNN

layers = [
    imageInputLayer(inputSize)

    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer([12 1])

    dropoutLayer(0.3)

    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer
];

options = trainingOptions('adam', ...
    'MaxEpochs',15, ...
    'MiniBatchSize',64, ...
    'ValidationData',imdsVal, ...
    'ValidationFrequency',30, ...
    'Plots','training-progress', ...
    'Verbose',true);

net = trainNetwork(imdsTrain, layers, options);

%% EVALUATION
preds = classify(net, imdsVal);
truth = imdsVal.Labels;

accuracy = mean(preds == truth)*100;
fprintf("Accuracy: %.2f%%\n", accuracy);

figure
confusionchart(truth, preds);
title("Baseline Confusion Matrix");
%% ========================================================
% TASK 2 - SIMPLE HYPERPARAMETER SEARCH
% 12 models total:
% base filters = 16 or 32
% filter size = 3 or 5
% num blocks   = 3, 4, 5
% Use 50% training data for speed-up
% =========================================================
disp('================ TASK 2: HYPERPARAMETER SEARCH =========');

% --------- create results folders ----------
resultsRoot = 'ELE456_results';
if ~exist(resultsRoot,'dir'); mkdir(resultsRoot); end

modelRoot = fullfile(resultsRoot,'saved_models_task2');
if ~exist(modelRoot,'dir'); mkdir(modelRoot); end

figRoot = fullfile(resultsRoot,'figures_task2');
if ~exist(figRoot,'dir'); mkdir(figRoot); end

% --------- use 50% training set for speed-up ----------
% --------- 手动随机选 50% 数据 ----------
numFiles = numel(imdsTrain.Files);
randIdx = randperm(numFiles);

numTrain50 = round(0.5 * numFiles);
selectedIdx = randIdx(1:numTrain50);

imdsTrain50 = subset(imdsTrain, selectedIdx);

disp('Task 2 training subset label counts:');
disp(countEachLabel(imdsTrain50));

disp('Task 2 training subset label counts:');
disp(countEachLabel(imdsTrain50));

% --------- hyperparameter candidates ----------
baseFilterList = [16 32];
filterSizeList = [3 5];
numBlockList   = [3 4 5];

% --------- search settings ----------
task2Epochs = 8;         % 先用 8，比 12/15 更稳更快
task2MiniBatch = 64;     % CPU 上更稳

results = struct([]);
rowCounter = 1;

for bf = baseFilterList
    for fs = filterSizeList
        for nb = numBlockList

            fprintf('\nTraining model %d/12: baseFilters=%d, filterSize=%d, numBlocks=%d\n', ...
                rowCounter, bf, fs, nb);

            % build network
           layers_task2 = buildCustomSpeechCNN(inputSize, numClasses, bf, fs, nb, 0.30);

            % training options
            options_task2 = trainingOptions('adam', ...
                'InitialLearnRate', 1e-3, ...
                'MaxEpochs', task2Epochs, ...
                'MiniBatchSize', task2MiniBatch, ...
                'Shuffle', 'every-epoch', ...
                'ValidationData', imdsVal, ...
                'ValidationFrequency', max(1,floor(numel(imdsTrain50.Files)/task2MiniBatch)), ...
                'Verbose', true, ...
                'Plots', 'none', ...
                'ExecutionEnvironment', 'auto', ...
                'L2Regularization', 1e-4);

            % train
            net_task2 = trainNetwork(imdsTrain50, layers_task2, options_task2);

            % evaluate
            preds_task2 = classify(net_task2, imdsVal);
            truth_task2 = imdsVal.Labels;
            acc_task2 = mean(preds_task2 == truth_task2) * 100;
            cm_task2 = confusionmat(truth_task2, preds_task2);

            % save model
            modelName = sprintf('task2_bf%d_fs%d_nb%d', bf, fs, nb);
            save(fullfile(modelRoot, [modelName '.mat']), ...
                'net_task2','layers_task2','options_task2','acc_task2','cm_task2');

            % store results
            results(rowCounter).ModelName   = string(modelName);
            results(rowCounter).BaseFilters = bf;
            results(rowCounter).FilterSize  = fs;
            results(rowCounter).NumBlocks   = nb;
            results(rowCounter).ValAccuracy = acc_task2;

            fprintf('Validation Accuracy = %.2f%%\n', acc_task2);

            rowCounter = rowCounter + 1;
        end
    end
end

% --------- results table ----------
resultsTable = struct2table(results);
resultsTable = sortrows(resultsTable, 'ValAccuracy', 'descend');

disp('================ TASK 2 RESULTS TABLE ==================');
disp(resultsTable);

writetable(resultsTable, fullfile(resultsRoot, 'task2_search_results.csv'));

% --------- best model ----------
bestTask2Name = char(resultsTable.ModelName(1));
bestTask2Acc  = resultsTable.ValAccuracy(1);

fprintf('\nBest Task 2 model: %s\n', bestTask2Name);
fprintf('Best Task 2 validation accuracy: %.2f%%\n', bestTask2Acc);

load(fullfile(modelRoot, [bestTask2Name '.mat']), 'net_task2');
bestTask2Net = net_task2;

% evaluate best model again for confusion matrix
bestPreds = classify(bestTask2Net, imdsVal);
bestTruth = imdsVal.Labels;
bestAcc   = mean(bestPreds == bestTruth) * 100;

fig2 = figure('Name','Best Task 2 Confusion Matrix');
confusionchart(bestTruth, bestPreds, ...
    'Title', sprintf('Best Task 2 Confusion Matrix (Acc = %.2f%%)', bestAcc), ...
    'RowSummary','row-normalized', ...
    'ColumnSummary','column-normalized');

saveas(fig2, fullfile(figRoot, 'task2_best_confusion_matrix.png'));

fprintf('\nTask 2 completed.\n');
function layers = buildCustomSpeechCNN(inputSize, numClasses, baseFilters, filterSize, numBlocks, dropoutRate)
    currentFilters = baseFilters;
    timePoolSize = 12;

    layers = [
        imageInputLayer(inputSize, 'Name','input')
    ];

    for b = 1:numBlocks
        layers = [
            layers
            convolution2dLayer(filterSize, currentFilters, 'Padding','same', 'Name', sprintf('conv_%d',b))
            batchNormalizationLayer('Name', sprintf('bn_%d',b))
            reluLayer('Name', sprintf('relu_%d',b))
        ];

        if b == 1 || b == 3
            layers = [
                layers
                maxPooling2dLayer(2, 'Stride', 2, 'Name', sprintf('pool_%d',b))
            ];
            currentFilters = currentFilters * 2;
        end
    end

    layers = [
        layers
        maxPooling2dLayer([12 1], 'Stride', [12 1], 'Name','time_pool')
        dropoutLayer(dropoutRate, 'Name','dropout')
        fullyConnectedLayer(numClasses, 'Name','fc')
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classOutput')
    ];
end
