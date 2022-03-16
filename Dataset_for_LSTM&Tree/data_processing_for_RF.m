%% Data preprocessing for Random Forest
% the aim of this script is to preprocess the train_dataset.mat
% to be used with Classification Learner APP in order to 
% train a random forest.
clc; clear; close all;

%% Needed:
%   - a smaller dataset. Is not possible to use the CL App with this
%       amount of data. As show in medium, we are going to use 40
%       simulation run for fault free data and 25 simulation run for each
%       faulty data
%   - normalization needed.
load train_dataset.mat;

%% Filter dataset
condition = data.simulationRun <= 40 & data.faultNumber == 0 ...
    | data.simulationRun <= 25 & data.faultNumber ~= 0;
data = data(condition, :);

%% Normalize columns
data = normalize(data, "zscore", ...
    "DataVariables", data.Properties.VariableNames(4:end));
