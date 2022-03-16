%% Descrizione
% bisogna costruire un esempio per ogni feature del dataset,
% avremo quindi 52 matrici di esempio da cui calcolare le distanze.
% Come sono fatte queste matrici di esempio?
% Dovranno essere delle matrici di time series (tensori o cellArray quindi)
% Tutto questo dovrà poi essere fatto per ogni fault.

% Ricapitolando:

% per ogni colonna dovremmo costruire un esempio fatto di x campioni.
% una matrice di esempio sarà quindi nxm dove n è il numero di esempi
% che vogliamo considerare, mentre m è la lunghezza della time series.
clc;close all;clear;

data = [readtable("C:/Users/davide/Desktop/Manutenzione Robotica/Progetto/Dataset/TEP_FaultFree_Training.csv");...
    readtable("C:/Users/davide/Desktop/Manutenzione Robotica/Progetto/Dataset/TEP_Faulty_Training.csv")];
data.(1) = [];

%% Hyperparameters
window_len = 20;
window_num = 100;
col_beg    = 4;
col_end    = 55;
num_col    = col_end - col_beg + 1;
n_faults   = 21;
offset = 20;
test_num_for_fault_code = 10;
n_simul = 60;

%% Training
exampleImages = train(data, n_faults, window_num, window_len, col_beg, col_end, n_simul);

%% Testing to create a distance vector
clc

%prepare distance H,J,E e Labels
h_result = cell(n_faults, 1);
H = cell(n_simul, 1);

j_result = cell(n_faults, 1);
J = cell(n_simul, 1);

e_result = cell(n_faults, 1);
E = cell(n_simul, 1);

labels = cell(n_faults, 1);
L = cell(n_simul, 1);

for s = 1:n_simul
    for f = 1:n_faults
        datatest = data(data.simulationRun == s & data.faultNumber == f-1, :);
        Imagetest = cell(num_col, 1);
        for i = 1:num_col
            Imagetest{i,1} = zeros(window_num, window_len);
        end

        for i = 1:window_num
            dtest = table2array(datatest(i+offset:i+window_len-1+offset, col_beg:col_end));
            for j = 1:num_col
                x = dtest(:, j);
                Imagetest{j,1}(i,:) = x';
            end
        end

        vect_test = cell(num_col, 1);

        for j = 1:num_col
            Imagetest{j, 1} = Imagetest{j, 1}';
            vect_test{j, 1} = sort(Imagetest{j, 1}(:)', 'descend');
            vect_test{j, 1} = normalize(vect_test{j, 1}, 'norm',1);
            Imagetest{j, 1} = vect_test{j, 1};
        end
        clear datatest dtest i j x

        [a,b,c] = classify(exampleImages, Imagetest);
        h_result{f,1} = a;
        j_result{f,1} = b;
        e_result{f,1} = c;
        labels{f,1} = f-1;
    end
    H{s,1} = h_result;
    J{s, 1} = j_result;
    E{s, 1} = e_result;
    L{s, 1} = labels;
    
end
%% Confusion matrix and calculate JEFFREY DISTANCE

%create vectors prediciton e labels
prediction = zeros(n_simul, n_faults);
real = zeros(n_simul, n_faults);

for i=1:n_simul
    for j=1:n_faults
        prediction(i,j) = cell2mat(J{i,1}(j,1));
        real(i, j) = cell2mat(L{i,1}(j,1));
    end
end

real = real(:);
prediction = prediction(:);

%calculate PRECISION, RECALL, F1SCORE of JEFFREY
cmt1 = confusionmat(real, prediction);
%cmt1 = cmt1';

diagonal = diag(cmt1); %true positive
sum_of_rows = sum(cmt1,2); %true positive + false positive
sum_of_cols = sum(cmt1,1); %true positive + false negative

precision = diagonal ./ sum_of_rows;
recall = diagonal ./ sum_of_cols';

precision_total = mean(precision(~isnan(precision)));
recall_total = mean(recall(~isnan(recall)));
f1_score = 2*(precision_total*recall_total)/(precision_total+recall_total);

%Plot confusion matrix JEFFREY
cm1 = confusionchart(real, prediction);
cm1.RowSummary = 'row-normalized';
cm1.Title = sprintf("Jeffrey distance with %i Consecutive windows \n PRECISION = %f \n RECALL = %f \n F1SCORE = %f"...
    ,window_num, precision_total, recall_total, f1_score);
%% Confusion matrix and calculate HELLINGER DISTANCE

%create vectors prediciton e labels
prediction = zeros(n_simul, n_faults);
real = zeros(n_simul, n_faults);

for i=1:n_simul
    for j=1:n_faults
        prediction(i,j) = cell2mat(H{i,1}(j,1));
        real(i, j) = cell2mat(L{i,1}(j,1));
    end
end

%calculate PRECISION, RECALL, F1SCORE of JEFFREY
cmt1 = confusionmat(real(:), prediction(:));
%cmt1 = cmt1';

diagonal = diag(cmt1); %true positive
sum_of_rows = sum(cmt1,2); %true positive + false positive
sum_of_cols = sum(cmt1,1); %true positive + false negative

precision = diagonal ./ sum_of_rows;
recall = diagonal ./ sum_of_cols';

precision_total = mean(precision(~isnan(precision)));
recall_total = mean(recall(~isnan(recall)));
f1_score = 2*(precision_total*recall_total)/(precision_total+recall_total);

%Plot confusion matrix HELLINGER
cm1 = confusionchart(real(:), prediction(:));
cm1.RowSummary = 'row-normalized';
cm1.Title = sprintf("Hellinger distance with %i Consecutive windows \n PRECISION = %f \n RECALL = %f \n F1SCORE = %f"...
    ,window_num, precision_total, recall_total, f1_score);
%% Confusion matrix and calculate EUCLIDEAN DISTANCE

%create vectors prediciton e labels
prediction = zeros(n_simul, n_faults);
real = zeros(n_simul, n_faults);

for i=1:n_simul
    for j=1:n_faults
        prediction(i,j) = cell2mat(E{i,1}(j,1));
        real(i, j) = cell2mat(L{i,1}(j,1));
    end
end

%calculate PRECISION, RECALL, F1SCORE of JEFFREY
cmt1 = confusionmat(real(:), prediction(:));
%cmt1 = cmt1';

diagonal = diag(cmt1); %true positive
sum_of_rows = sum(cmt1,2); %true positive + false positive
sum_of_cols = sum(cmt1,1); %true positive + false negative

precision = diagonal ./ sum_of_rows;
recall = diagonal ./ sum_of_cols';

precision_total = mean(precision(~isnan(precision)));
recall_total = mean(recall(~isnan(recall)));
f1_score = 2*(precision_total*recall_total)/(precision_total+recall_total);

%Plot confusion matrix HELLINGER
cm1 = confusionchart(real(:), prediction(:));
cm1.RowSummary = 'row-normalized';
cm1.Title = sprintf("Euclidean distance with %i Consecutive windows \n PRECISION = %f \n RECALL = %f \n F1SCORE = %f"...
    ,window_num, precision_total, recall_total, f1_score);



%% FUNZIONI
%% Training function
function Trained = train(dataset, nfaults, nwindows, lwindows, col_begin, col_end, n_simul)
    Trained = cell(nfaults, 1);
    n_columns = col_end - col_begin;
    for i = 1:nfaults
        filter = dataset.simulationRun == n_simul+1 & ...
            dataset.faultNumber == i-1 & dataset.sample > 20;

        Trained{i,1} = cell(n_columns, 1);
        Trained{i,1} = createImage(table2array(...
            dataset(filter, :)), nwindows,...
            lwindows, col_begin, col_end,0,0);
    end
    
end

%% Funzione per la generazione delle "Immagini"
% In particolare per generare un immagine di esempio ho bisogno di:
%   - una sessione intera di simulazione,
%   - il numero di finestre,
%   - la lunghezza delle finestre
function Image = createImage(session, nwindows, lwindows, col_begin, col_end, offset, train_flag)
    n_columns = col_end - col_begin + 1;
    Imagetrain = cell(n_columns, 1);

    if nargin == 6
        offset = 0;
    end

    for i = 1:n_columns
        Imagetrain{i,1} = zeros(nwindows, lwindows);
    end
    for i = 1:nwindows
        
        % per ogni finestra bisogna appendere le colonne in Imagetrain
        if train_flag == 1
            rn = randi([1 size(session,1)-lwindows+1]); % choose randomly a window
            x = session(rn:rn+lwindows-1, col_begin:col_end); % get the subsession
        else
            x = session(i+offset:i+offset+lwindows-1, col_begin:col_end);
        end
        for j = 1:n_columns
            col = x(:, j);
            Imagetrain{j, 1}(i, :) = col';
        end
    end
    
    vect = cell(n_columns, 1);
    
    for i = 1:n_columns
        Imagetrain{i, 1} = Imagetrain{i,1}';
        vect{i, 1} = sort(Imagetrain{i,1}(:)','descend');
        vect{i, 1} = normalize(vect{i, 1}, 'norm',1);
        
    end
    Image = vect;
end

%% Funzione per il calcolo delle distanze
function [Jeffrey, Hellinger, Euclidean] = getSingleDistance(img1, img2)

    density_train=img1{1};
    %density_train(density_train<0)=0; % numerical issue
    density_train=density_train+realmin; % numerical issue
    density_train_sqrt=sqrt(density_train);
    density_train_log=log(density_train);

    density_test=img2;
    %density_test(density_test<0)=0;
    density_test=density_test+realmin;
    DELTA=density_train-density_test;
    DELTAsqrt=density_train_sqrt-sqrt(density_test);

    % Jeffreys divergence
    logd=density_train_log-log(density_test);
    temp1=density_train.*logd;
    temp2=density_test.*logd;
    Jeffrey=sum(sum(temp1-temp2,1));
    % Hellinger
    Hellinger=sum(sqrt(sum(DELTAsqrt.^2,1))/sqrt(2));
    % Euclidean
    Euclidean=sum(sqrt(sum(DELTA.^2,1)));

end

function [class_J, class_H, class_E] = classify(trained, test)
    n_faults  = size(trained, 1);
    n_columns = size(trained{1,1}, 1);
    assert(n_columns==size(test, 1),...
        "arguments must have the same columns");

    % matrice 21x52, ogni riga andrebbe mediata
    J_vect = zeros(n_faults, n_columns);
    H_vect = zeros(n_faults, n_columns);
    E_vect = zeros(n_faults, n_columns);

    for fc = 1:n_faults
        for nc = 1:n_columns
            [J, H, E] = getSingleDistance(...
                trained{fc}(nc),...
                test{nc});
            J_vect(fc, nc) = J;
            H_vect(fc, nc) = H;
            E_vect(fc, nc) = E;
        end

    end
    
    meaned_J = mean(J_vect, 2);
    meaned_H = mean(H_vect, 2);
    meaned_E = mean(E_vect, 2);

    [~, class_J] = min(meaned_J);
    [~, class_H] = min(meaned_H);
    [~, class_E] = min(meaned_E);
    class_J = class_J - 1;
    class_H = class_H - 1;
    class_E = class_E - 1;
    
end









