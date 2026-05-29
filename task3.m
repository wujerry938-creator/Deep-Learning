%% ========================================================
% TASK 3 - TRANSFER LEARNING WITH MOBILENETV2
% =========================================================
disp('================ TASK 3: TRANSFER LEARNING =============');
baselineAcc = 58.24;      % 你的 Task 1
bestTask2Acc = 57.64;     % 你的 Task 2
canRunTransfer = true;

try
    netTL = mobilenetv2;
catch
    warning('mobilenetv2 not found. Please install the MobileNetV2 support package.');
    canRunTransfer = false;
end

if canRunTransfer

    % --------- Input size for MobileNetV2 ----------
    inputSizeTL = netTL.Layers(1).InputSize;   % usually [224 224 3]

    % --------- Convert grayscale spectrograms to RGB and resize ----------
    augTrainTL = augmentedImageDatastore(inputSizeTL(1:2), imdsTrain, ...
        'ColorPreprocessing', 'gray2rgb');

    augValTL = augmentedImageDatastore(inputSizeTL(1:2), imdsVal, ...
        'ColorPreprocessing', 'gray2rgb');

    % --------- Create layer graph ----------
    lgraph = layerGraph(netTL);

    % Find layers to replace
    [learnableLayer, classLayer] = findLayersToReplaceCustom(lgraph);

    % Replace final fully connected layer
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);

    % Replace classification layer
    newClassLayer = classificationLayer('Name','new_classoutput');

    lgraph = replaceLayer(lgraph, learnableLayer.Name, newLearnableLayer);
    lgraph = replaceLayer(lgraph, classLayer.Name, newClassLayer);

    % --------- Freeze most early layers a bit ----------
    layersTL = lgraph.Layers;
    connectionsTL = lgraph.Connections;

    for i = 1:numel(layersTL)
        if isprop(layersTL(i), 'WeightLearnRateFactor')
            layersTL(i).WeightLearnRateFactor = 0.1;
        end
        if isprop(layersTL(i), 'BiasLearnRateFactor')
            layersTL(i).BiasLearnRateFactor = 0.1;
        end
    end

    % Keep the new last layer trainable faster
    for i = 1:numel(layersTL)
        if strcmp(layersTL(i).Name, 'new_fc')
            layersTL(i).WeightLearnRateFactor = 10;
            layersTL(i).BiasLearnRateFactor = 10;
        end
    end

    lgraph = createLgraphUsingConnectionsCustom(layersTL, connectionsTL);

    % --------- Training options ----------
    optionsTL = trainingOptions('adam', ...
        'MiniBatchSize', 32, ...
        'MaxEpochs', 8, ...
        'InitialLearnRate', 1e-4, ...
        'Shuffle', 'every-epoch', ...
        'ValidationData', augValTL, ...
        'ValidationFrequency', max(1, floor(numel(imdsTrain.Files)/32)), ...
        'Verbose', true, ...
        'Plots', 'training-progress', ...
        'ExecutionEnvironment', 'auto');

    % --------- Train ----------
    transferNet = trainNetwork(augTrainTL, lgraph, optionsTL);

    % --------- Evaluate ----------
    transferPreds = classify(transferNet, augValTL);
    transferTruth = imdsVal.Labels;
    transferAcc = mean(transferPreds == transferTruth) * 100;

    fprintf('Transfer learning validation accuracy = %.2f%%\n', transferAcc);

    % --------- Confusion matrix ----------
    fig3 = figure('Name','Task 3 Confusion Matrix');
    confusionchart(transferTruth, transferPreds, ...
        'Title', sprintf('Task 3 Transfer Learning Confusion Matrix (Acc = %.2f%%)', transferAcc), ...
        'RowSummary','row-normalized', ...
        'ColumnSummary','column-normalized');

    % --------- Save results ----------
    resultsRoot = 'ELE456_results';
    if ~exist(resultsRoot,'dir'); mkdir(resultsRoot); end

    modelRoot = fullfile(resultsRoot,'saved_models_task3');
    if ~exist(modelRoot,'dir'); mkdir(modelRoot); end

    figRoot = fullfile(resultsRoot,'figures_task3');
    if ~exist(figRoot,'dir'); mkdir(figRoot); end

    save(fullfile(modelRoot,'transferNet_mobilenetv2.mat'), ...
        'transferNet','lgraph','optionsTL','transferAcc');

    saveas(fig3, fullfile(figRoot,'task3_transfer_confusion_matrix.png'));

    % --------- Compare with Task 1 and Task 2 ----------
    comparisonNames = ["Task1_Baseline"; "Task2_Best"; "Task3_Transfer"];
    comparisonAccs  = [baselineAcc; bestTask2Acc; transferAcc];

    comparisonTable = table(comparisonNames, comparisonAccs, ...
        'VariableNames', {'Model','ValidationAccuracy'});

    comparisonTable = sortrows(comparisonTable, 'ValidationAccuracy', 'descend');

    disp('================ TASK 1/2/3 COMPARISON ================');
    disp(comparisonTable);

    writetable(comparisonTable, fullfile(resultsRoot,'task1_task2_task3_comparison.csv'));

end
function [learnableLayer, classLayer] = findLayersToReplaceCustom(lgraph)

    if ~isa(lgraph, 'nnet.cnn.LayerGraph')
        error('Input must be a layerGraph.');
    end

    layers = lgraph.Layers;
    connections = lgraph.Connections;

    src = string(connections.Source);
    dst = string(connections.Destination);

    classificationLayerIdx = [];
    for i = 1:numel(layers)
        if isa(layers(i), 'nnet.cnn.layer.ClassificationOutputLayer')
            classificationLayerIdx = i;
            break;
        end
    end

    if isempty(classificationLayerIdx)
        error('No classification layer found.');
    end

    classLayer = layers(classificationLayerIdx);
    className = string(classLayer.Name);

    sourceLayerName = src(dst == className);
    if isempty(sourceLayerName)
        error('Could not find layer feeding into classification layer.');
    end

    learnableLayerIdx = find(string({layers.Name}) == sourceLayerName(1), 1);
    if isempty(learnableLayerIdx)
        error('Could not find learnable layer.');
    end

    learnableLayer = layers(learnableLayerIdx);
end
function lgraph = createLgraphUsingConnectionsCustom(layers, connections)
    lgraph = layerGraph();
    for i = 1:numel(layers)
        lgraph = addLayers(lgraph, layers(i));
    end
    for c = 1:size(connections,1)
        lgraph = connectLayers(lgraph, connections.Source{c}, connections.Destination{c});
    end
end