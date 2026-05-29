%% ========================================================
% TASK 5 - DATA AUGMENTATION EXTENSION
% =========================================================
disp('================ TASK 5: DATA AUGMENTATION EXTENSION ===');
bestDropoutTask5 = 0.2;
bestL2Task5      = 0.001;

% --------- if running separately, manually provide previous results ----------
if ~exist('baselineAcc','var')
    baselineAcc = 58.24;
end

if ~exist('bestTask2Acc','var')
    bestTask2Acc = 57.64;
end

if ~exist('transferAcc','var')
    transferAcc = 21.01;
end

if ~exist('bestTask4Acc','var')
    bestTask4Acc = 72.42;
end

% --------- dataset check ----------
if ~exist('imdsTrain','var') || ~exist('imdsVal','var')
    trainFolder = 'TrainData';
    valFolder   = 'ValData';

    imdsTrain = imageDatastore(trainFolder, ...
        "IncludeSubfolders", true, ...
        "LabelSource", "foldernames");

    imdsVal = imageDatastore(valFolder, ...
        "IncludeSubfolders", true, ...
        "LabelSource", "foldernames");
end

if ~exist('inputSize','var')
    inputSize = [98 50 1];
end

if ~exist('numClasses','var')
    numClasses = numel(categories(imdsTrain.Labels));
end

% --------- use the best Task 4 style architecture ----------
% If you know the exact best Task 4 parameters, set them here.
fixedBaseFilters = 16;
fixedFilterSize  = 5;
fixedNumBlocks   = 5;

% Best regularisation values from Task 4:
% Replace these with your real best Task 4 values if needed.
bestDropoutTask5 = 0.30;
bestL2Task5      = 1e-4;

% --------- custom augmented datastore ----------
augImdsTrain = augmentedImageDatastore( ...
    inputSize(1:2), ...
    imdsTrain, ...
    'DataAugmentation', imageDataAugmenter( ...
        'RandXTranslation', [-4 4], ...
        'RandYTranslation', [-2 2], ...
        'RandXScale', [0.95 1.05], ...
        'RandYScale', [0.95 1.05]), ...
    'ColorPreprocessing', 'none');

augImdsVal = augmentedImageDatastore( ...
    inputSize(1:2), ...
    imdsVal, ...
    'ColorPreprocessing', 'none');

% --------- build network ----------
layers_task5 = buildCustomSpeechCNN(inputSize, numClasses, ...
    fixedBaseFilters, fixedFilterSize, fixedNumBlocks, bestDropoutTask5);

% --------- training options ----------
task5Epochs = 10;
task5Batch  = 64;

options_task5 = trainingOptions('adam', ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs', task5Epochs, ...
    'MiniBatchSize', task5Batch, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augImdsVal, ...
    'ValidationFrequency', max(1, floor(numel(imdsTrain.Files)/task5Batch)), ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto', ...
    'L2Regularization', bestL2Task5);

% --------- train ----------
task5Net = trainNetwork(augImdsTrain, layers_task5, options_task5);

% --------- evaluate ----------
task5Preds = classify(task5Net, augImdsVal);
task5Truth = imdsVal.Labels;
task5Acc = mean(task5Preds == task5Truth) * 100;
task5CM = confusionmat(task5Truth, task5Preds);

fprintf('Task 5 validation accuracy = %.2f%%\n', task5Acc);

% --------- plot confusion matrix ----------
fig5 = figure('Name','Best Task 5 Confusion Matrix');
confusionchart(task5Truth, task5Preds, ...
    'Title', sprintf('Task 5 Augmentation Confusion Matrix (Acc = %.2f%%)', task5Acc), ...
    'RowSummary','row-normalized', ...
    'ColumnSummary','column-normalized');

% --------- save results ----------
resultsRoot = 'ELE456_results';
if ~exist(resultsRoot,'dir'); mkdir(resultsRoot); end

modelRoot = fullfile(resultsRoot,'saved_models_task5');
if ~exist(modelRoot,'dir'); mkdir(modelRoot); end

figRoot = fullfile(resultsRoot,'figures_task5');
if ~exist(figRoot,'dir'); mkdir(figRoot); end

save(fullfile(modelRoot,'task5_augmented_model.mat'), ...
    'task5Net','layers_task5','options_task5','task5Acc','task5CM', ...
    'bestDropoutTask5','bestL2Task5');

saveas(fig5, fullfile(figRoot,'task5_confusion_matrix.png'));

% --------- final comparison ----------
comparisonNames5 = ["Task1_Baseline"; "Task2_Best"; "Task3_Transfer"; "Task4_Best"; "Task5_Augmented"];
comparisonAccs5  = [baselineAcc; bestTask2Acc; transferAcc; bestTask4Acc; task5Acc];

comparisonTable5 = table(comparisonNames5, comparisonAccs5, ...
    'VariableNames', {'Model','ValidationAccuracy'});

comparisonTable5 = sortrows(comparisonTable5, 'ValidationAccuracy', 'descend');

disp('================ TASK 1/2/3/4/5 COMPARISON ============');
disp(comparisonTable5);

writetable(comparisonTable5, fullfile(resultsRoot,'task1_task2_task3_task4_task5_comparison.csv'));

disp('Task 5 completed.');
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