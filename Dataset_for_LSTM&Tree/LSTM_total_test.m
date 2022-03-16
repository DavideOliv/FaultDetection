%% Cleaning
close all; clear; clc;
%% carico i dati di test
test_data_FF = readtable("TEP_FaultFree_Testing.csv");
test_data_F = readtable("TEP_Faulty_Testing.csv");
test_data_FF.(1) = [];
test_data_F.(1) = [];
%% creo il test totale
test_data_F = test_data_F(test_data_F.sample > 160, :);
dataset_test = [test_data_FF; test_data_F];

%% dati e label
seq_len = 20;
n_features = 52;
labels_test = dataset_test.faultNumber(height(test_k1)/20);
data_test = squeeze(num2cell(reshape(table2array(dataset_test(:, 4:end))', 52 ,seq_len, []),[1 2]));
%% carico la rete
net = load("tnet-zscore.mat");
net = net.trained_model;
%% previsioni
pred = [];
real = [];

for i = 1:size(labels_test)
    prediction = predict(net, data_test(i));
    [val(i) temp(i)] = max(prediction);
    pred(i) = temp(i) - 1;
    real(i) = labels_test(i);
end

%% matrice di confusione
chart = confusionchart(real,pred)
chart.RowSummary = 'row-normalized'
