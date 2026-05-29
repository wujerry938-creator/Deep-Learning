%% ========================================================
% TASK 4 - DROPOUT + L2 SEARCH (Practical BO-style Search)
% =========================================================
disp('================ TASK 4: DROPOUT + L2 SEARCH ==========');

% 如果单独运行 task4，可手动补前面结果
if ~exist('baselineAcc','var')
    baselineAcc = 58.24;
end

if ~exist('bestTask2Acc','var')
    bestTask2Acc = 57.64;
end

if ~exist('transferAcc','var')
    transferAcc = 21.01;
end

% --------- fixed architecture from Task 2 insights ----------
fixedBaseFilters = 16;
fixedFilterSize  = 5;
fixedNumBlocks   = 5;

% --------- search space ----------
dropoutList = [0.2 0.3 0.5];
l2List      = [1e-4 5e-4 1e-3];

task4Epochs = 8;        % 可先用 6 测试，正式用 8
task4Batch  = 64;

resultsRoot = 'ELE456_results';
if ~exist(resultsRoot,'dir'); mkdir(resultsRoot); end

modelRoot = fullfile(resultsRoot,'saved_models_task4');
if ~exist(modelRoot,'dir'); mkdir(modelRoot); end

figRoot = fullfile(resultsRoot,'figures_task4');
if ~exist(figRoot,'dir'); mkdir(figRoot); end

results4 = struct([]);
rowCounter = 1;

for d = 1:numel(dropoutList)
    for r = 1:numel(l2List)

        dropoutRate = dropoutList(d);
        l2Value = l2List(r);

        fprintf('\nTraining Task 4 model %d/9: dropout=%.2f, L2=%g\n', ...
            rowCounter, dropoutRate, l2Value);

        % build network
        layers_task4 = buildCustomSpeechCNN(inputSize, numClasses, ...
            fixedBaseFilters, fixedFilterSize, fixedNumBlocks, dropoutRate);

        % options
        options_task4 = trainingOptions('adam', ...
            'InitialLearnRate', 1e-3, ...
            'MaxEpochs', task4Epochs, ...
            'MiniBatchSize', task4Batch, ...
            'Shuffle', 'every-epoch', ...
            'ValidationData', imdsVal, ...
            'ValidationFrequency', max(1, floor(numel(imdsTrain.Files)/task4Batch)), ...
            'Verbose', true, ...
            'Plots', 'none', ...
            'ExecutionEnvironment', 'auto', ...
            'L2Regularization', l2Value);

        % train
        net_task4 = trainNetwork(imdsTrain, layers_task4, options_task4);

        % evaluate
        preds_task4 = classify(net_task4, imdsVal);
        truth_task4 = imdsVal.Labels;
        acc_task4 = mean(preds_task4 == truth_task4) * 100;
        cm_task4 = confusionmat(truth_task4, preds_task4);

        modelName = sprintf('task4_drop%.2f_l2_%g', dropoutRate, l2Value);
        modelName = strrep(modelName,'.','p');  % 避免文件名小数点太多

        save(fullfile(modelRoot, [modelName '.mat']), ...
            'net_task4','layers_task4','options_task4','acc_task4','cm_task4', ...
            'dropoutRate','l2Value');

        results4(rowCounter).ModelName   = string(modelName);
        results4(rowCounter).Dropout     = dropoutRate;
        results4(rowCounter).L2          = l2Value;
        results4(rowCounter).ValAccuracy = acc_task4;

        fprintf('Validation Accuracy = %.2f%%\n', acc_task4);

        rowCounter = rowCounter + 1;
    end
end

% --------- results table ----------
resultsTable4 = struct2table(results4);
resultsTable4 = sortrows(resultsTable4, 'ValAccuracy', 'descend');

disp('================ TASK 4 RESULTS TABLE ==================');
disp(resultsTable4);

writetable(resultsTable4, fullfile(resultsRoot, 'task4_search_results.csv'));

% --------- best task 4 model ----------
bestTask4Name = char(resultsTable4.ModelName(1));
bestTask4Acc  = resultsTable4.ValAccuracy(1);

fprintf('\nBest Task 4 model: %s\n', bestTask4Name);
fprintf('Best Task 4 validation accuracy: %.2f%%\n', bestTask4Acc);

load(fullfile(modelRoot, [bestTask4Name '.mat']), 'net_task4');
bestTask4Net = net_task4;

bestPreds4 = classify(bestTask4Net, imdsVal);
bestTruth4 = imdsVal.Labels;

fig4 = figure('Name','Best Task 4 Confusion Matrix');
confusionchart(bestTruth4, bestPreds4, ...
    'Title', sprintf('Best Task 4 Confusion Matrix (Acc = %.2f%%)', bestTask4Acc), ...
    'RowSummary','row-normalized', ...
    'ColumnSummary','column-normalized');

saveas(fig4, fullfile(figRoot, 'task4_best_confusion_matrix.png'));

% --------- comparison table ----------
comparisonNames4 = ["Task1_Baseline"; "Task2_Best"; "Task3_Transfer"; "Task4_Best"];
comparisonAccs4  = [baselineAcc; bestTask2Acc; transferAcc; bestTask4Acc];

comparisonTable4 = table(comparisonNames4, comparisonAccs4, ...
    'VariableNames', {'Model','ValidationAccuracy'});

comparisonTable4 = sortrows(comparisonTable4, 'ValidationAccuracy', 'descend');

disp('================ TASK 1/2/3/4 COMPARISON ==============');
disp(comparisonTable4);

writetable(comparisonTable4, fullfile(resultsRoot,'task1_task2_task3_task4_comparison.csv'));

disp('Task 4 completed.');
clear; clc; close all;
rng(1,'twister');

trainFolder = 'TrainData';
valFolder   = 'ValData';

imdsTrain = imageDatastore(trainFolder, ...
    "IncludeSubfolders",true, ...
    "LabelSource","foldernames");

imdsVal = imageDatastore(valFolder, ...
    "IncludeSubfolders",true, ...
    "LabelSource","foldernames");

inputSize = [98 50 1];
numClasses = numel(categories(imdsTrain.Labels));
function layers = buildCustomSpeechCNN(inputSize, numClasses, baseFilters, filterSize, numBlocks, dropoutRate)

    currentFilters = baseFilters;

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