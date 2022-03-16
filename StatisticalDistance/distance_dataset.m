%% USARE SOLO LA DISTANZA EUCLIDEA PER CLASSIFICARE
% perchè le distanze sono state normalizzate


%% load data
clc;close all;clear;

data = [readtable("C:/Users/davide/Desktop/Manutenzione Robotica/Progetto/LSTM&Tree/TEP_FaultFree_Training.csv");...
    readtable("C:/Users/davide/Desktop/Manutenzione Robotica/Progetto/LSTM&Tree/TEP_Faulty_Training.csv")];
data.(1) = [];

%% Hyperparameters
window_len = 20;
window_num = 10;
col_beg    = 4;
col_end    = 55;
num_col    = col_end - col_beg + 1;
n_faults   = 21;
offset = 20;
n_simul = 2; %max 499
n_sub_simul = 22; %fixed
flag = 0; %create consecutive window


%% create train image
rif = train(data, n_faults, window_num, window_len, col_beg, col_end, flag);

%% generate distance dataset 
[jeffrey,hellinger,euclidean,labels] = genDataset(data, rif, n_simul, n_faults, window_len, n_sub_simul);
%% save
save("euclidea",'euclidean','-v7.3');
save('label', 'labels', '-v7.3');


%% FUNZIONI
% Training function
function Trained = train(dataset, nfaults, nwindows, lwindows, col_begin, col_end, train_flag)
    Trained = cell(nfaults, 1);
    n_columns = col_end - col_begin;
    for i = 1:nfaults
        filter = dataset.simulationRun == 10 & ...
            dataset.faultNumber == i-1 & dataset.sample > 20;

        Trained{i,1} = cell(n_columns, 1);

        Trained{i,1} = createImage(table2array(...
            dataset(filter, :)), nwindows,...
            lwindows, col_begin, col_end, train_flag);
    end
    
end

%% Funzione per la generazione delle "Immagini"
% In particolare per generare un immagine di esempio ho bisogno di:
%   - una sessione intera di simulazione,
%   - il numero di finestre,
%   - la lunghezza delle finestre
function Image = createImage(session, nwindows, lwindows, col_begin, col_end, train_flag, offset)
    n_columns = col_end - col_begin + 1;
    Imagetrain = cell(n_columns, 1);

    if nargin == 6
        offset = 0;
    end

    for i = 1:n_columns
        Imagetrain{i,1} = zeros(nwindows, lwindows);
    end
    
    for i = 1:nwindows
        
%         per ogni finestra bisogna appendere le colonne in Imagetrain
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
        vect{i, 1} = normalize(vect{i, 1}, 'norm');
        
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

function [J_vect, H_vect, E_vect] = getAllDistance(trained, test)
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
   
    J_vect = J_vect(:)';
    H_vect = H_vect(:)';
    E_vect = E_vect(:)';
    
end


function [J_dist, H_dist, E_dist, label] = genDataset(dataset, rif, n_simul, n_fault, offset, sub_simul)

    count = 1;
    
    for s=2:n_simul+1
        for fc=1:n_fault
            filter = dataset.simulationRun == s & dataset.faultNumber == fc-1;
            data = dataset(filter, :);
            for o=offset:20:(20*sub_simul)
                img = createImage(table2array(data), 10, 20, 4, 55, 0, o);
                [J, H, E] = getAllDistance(rif, img);
                if(~isnan(J) & ~isnan(H) & ~isnan(E))
                    J_dist(count, :) = J;
                    H_dist(count, :) = H;
                    E_dist(count, :) = E;
                    label(count, 1) = data.faultNumber(1);
                    count = count + 1;
                end
            end
        end
        
    end
 
end




