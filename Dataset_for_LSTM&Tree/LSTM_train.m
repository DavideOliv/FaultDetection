%% Cleaning
close all; clear; clc;
%% Load data
% carico i dataset faultfree e faulty insieme (sono distinti dal fault number)
% ed elimino la prima colonna (indice generale già generato da MATLAB)
training_data = [readtable("TEP_FaultFree_Training.csv"); readtable("TEP_Faulty_Training.csv")];
training_data.(1) = [];
%% Pre Processing
% sappiamo che: nel dataset contenente i fault 
%   - per ogni fault code (0 to 20 con 0 fault free) abbiamo 500
%       simulazioni e per ogni simulazione 500 campioni (250'000),
%   - nel dataset faulty, nella prima ora (20 campioni, 1 campione ogni 3
%       minuti) non viene introdotto il fault, per cui per ogni simulazione
%       di ogni fault abbiamo 480 campioni e quindi per ogni fault da 1 a
%       20 abbiamo 500 * 480 campioni = 240'000,
%   - per fare il dataset bilanciato bisogna togliere 10'000 campioni dal
%       dataset fault free

% Elimino le prime 20 righe per ogni run di ogni fault number (compreso 0)
training_data = training_data(training_data.sample > 20, :);

%% Costruzione dataset di train
seq_len = 20;
n_features = 52;

% Creo il vettore delle labels
training_labels = categorical(training_data.faultNumber(mod(training_data.sample,seq_len)==1));

% Costruisco il tensore di training
training_data = squeeze(num2cell(reshape(table2array(training_data(:, 4:end))', 52 ,seq_len, []),[1 2]));

%% Costruzione rete neurale (LSTM model 3)
model = [
    sequenceInputLayer(n_features, "Normalization", 'zscore')    % rescale-zero-one, zscore
    lstmLayer(128,'OutputMode','sequence', 'GateActivationFunction','sigmoid')
    lstmLayer(128,'OutputMode','last', 'GateActivationFunction','sigmoid')
    fullyConnectedLayer(300)
    dropoutLayer(0.5)
    fullyConnectedLayer(128)
    dropoutLayer(0.8)
    batchNormalizationLayer
    fullyConnectedLayer(21)
    softmaxLayer
    classificationLayer];

%% Training rete
options = trainingOptions('adam', ...
    'GradientThreshold',1, ...
    'MiniBatchSize', 256, ...
    'MaxEpochs', 50, ...
    'Verbose',false, ...
    'Plots','training-progress');
trained_model = trainNetwork(training_data, training_labels, model, options);

%% Save network
save("tnet-zscore", "trained_model");