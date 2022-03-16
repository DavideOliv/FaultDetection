%% Test Datastore
% store test data in datastore variable
% access chunks of data and build test dataset
clc;clear;close all;

%% Data storing
% 1) store test csv data in datastore variable
num_samples_per_session = 960;

fault_free_test = tabularTextDatastore("TEP_FaultFree_Testing.csv");
fault_free_test.ReadSize = num_samples_per_session;

faulty_test = tabularTextDatastore("TEP_Faulty_Testing.csv");
faulty_test.ReadSize = num_samples_per_session;

var_names = faulty_test.VariableNames; % same for fault_free_test

%% Prepare data
% faulty
% 10 sumulation run for all faultNumber
% 19 fault number (1 to 20) => 190 iteration
num_el = 20;
num_det = 20;
step = num_det - 1;
tot_data_test = [];
tot_label_test = [];

for i = 1:10
    for j = 1:num_det
        data = read(faulty_test);
        if size(data, 1) ~= num_samples_per_session
            ss = num_samples_per_session - size(data, 1);
            faulty_test.ReadSize = ss;
            data = [data; read(faulty_test)];
            size(data)
            faulty_test.ReadSize = num_samples_per_session;
        end
        t = data(1,:);
        data = data(data.sample > 160, :);
        for k = 1:num_el:height(data)
            try
                tot_data_test = [tot_data_test; {table2array(data(k: k+step, 5:end))'}];   
                tot_label_test = [tot_label_test; {data.faultNumber(1)}];
            catch
                fprintf("\n\theight: %d, j: %d, i: %d\n", size(data,1), j, i)
            end
        end
    end
    data = read(fault_free_test);
    if size(data, 1) ~= num_samples_per_session
        ss = num_samples_per_session - size(data, 1);
        faulty_test.ReadSize = ss;
        data = [data; read(faulty_test)];
        size(data)
        faulty_test.ReadSize = num_samples_per_session;
    end
    data = data(data.sample > 160, :);
    for k = 1:num_el:height(data)
        try
            tot_data_test = [tot_data_test; {table2array(data(k: k+step, 5:end))'}];
            tot_label_test = [tot_label_test; {0}];
        catch
            fprintf("\n\theight: %d, j: %d, i: %d\n", size(data,1), j, i)
        end
    end
end
%% Saving dataset
dataset_test = [tot_data_test, tot_label_test];
save("dataset_test", "dataset_test");

%% Load networks
net = load("TNetworks/tnet-zscore.mat");
net = net.trained_model;
net.Layers


%% Make predictions and confusion chart
num_el = size(tot_data_test, 1);
predictions_rows = predict(net, tot_data_test);
predictions = zeros(num_el, 1);
for row_idx = 1:num_el
    [a, idx] = max(predictions_rows(row_idx, :));
    predictions(row_idx) = idx - 1;
end

%% Confusion chart
cm = confusionchart(predictions, table2array(cell2table(tot_label_test)));
cm.RowSummary = 'row-normalized';
