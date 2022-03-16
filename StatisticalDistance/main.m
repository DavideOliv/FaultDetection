clc;clear;close;
fds=fileDatastore('C:\Users\Frank\Desktop\IMS\1st_test','ReadFcn',@MyReadFcn);
dataarray=cell(numel(fds.Files),1);
i=1;
reset(fds);
while hasdata(fds)
    dataarray{i}=read(fds);
    i=i+1;
end
%load dataset1.mat;
c=size(dataarray,1);

Fs=20000; % signal frequency
mc=100; % windows
len=1024*2; % window length
window=hamming(len,'periodic'); %windowing
nfft=2^nextpow2(len); % fft points

%% Statistical Model -> Training
d=size(dataarray{1,1},2);
Imagetrain=zeros(mc,ceil((1+nfft)/2)*d-d);
for i=1:mc
    datatrain=dataarray{1,1};
    rn=randi([1 size(datatrain,1)-len+1]); % choose randomly a window
    x=datatrain(rn:rn+len-1,:);
    x=bsxfun(@minus,x,mean(x)); % remove mean
    [pxx,f]=periodogram(x,window,nfft,Fs); % power spectral density 
    Imagetrain(i,:)=reshape(pxx(2:end,:),[],1);
end
Imagetrain=sort(Imagetrain,1,'descend'); % Sorting operation to obtain CDF

density_train=Imagetrain;
density_train(density_train<0)=0; % numerical issue
density_train=density_train+realmin; % numerical issue
density_train_sqrt=sqrt(density_train);
density_train_log=log(density_train);

%% Distances calculation
distJ=zeros(c-1,1);
distH=zeros(c-1,1);
distE=zeros(c-1,1);
distT=zeros(c-1,1);
for k=2:c
    disp(k)

    %% Statistical Model -> Testing
    Imagetest=zeros(mc,ceil((1+nfft)/2)*d-d);
    for i=1:mc
        datatrain=dataarray{k,1};
        rn=randi([1 size(datatrain,1)-len+1]);
        x=datatrain(rn:rn+len-1,:);
        x=bsxfun(@minus,x,mean(x));
        [pxx,f]=periodogram(x,window,nfft,Fs);
        Imagetest(i,:)=reshape(pxx(2:end,:),[],1);
    end
    Imagetest=sort(Imagetest,1,'descend');
    
    density_test=Imagetest;
    density_test(density_test<0)=0;
    density_test=density_test+realmin;
    DELTA=density_train-density_test;
    DELTAsqrt=density_train_sqrt-sqrt(density_test);
    %% Jeffreys divergence
    logd=density_train_log-log(density_test);
    temp1=density_train.*logd;
    temp2=density_test.*logd;
    distJ(k)=sum(sum(temp1-temp2,1));
    %% Hellinger
    distH(k)=sum(sqrt(sum(DELTAsqrt.^2,1))/sqrt(2));
    %% Euclidean
    distE(k)=sum(sqrt(sum(DELTA.^2,1)));
    %% Total Variation
    distT(k)=0.5*sum(sum(abs(DELTA).^2,1));
end


function data=MyReadFcn(filename)
opts=delimitedTextImportOptions("NumVariables", 8);
%opts=delimitedTextImportOptions("NumVariables", 4);

% Specify range and delimiter
opts.DataLines=[1,Inf];
opts.Delimiter="\t";

% Specify column names and types
opts.VariableNames=["VarName1","VarName2","VarName3","VarName4","VarName5","VarName6","VarName7","VarName8"];
opts.VariableTypes=["double","double","double","double","double","double","double","double"];
%opts.VariableNames=["VarName1","VarName2","VarName3","VarName4"];
%opts.VariableTypes=["double","double","double","double"];

% Specify file level properties
opts.ExtraColumnsRule="ignore";
opts.EmptyLineRule="read";

% Import the data
data=readtable(filename,opts);

%% Convert to output type
data=table2array(data);
data=data(:,8);
%% Clear temporary variables
clear opts
end