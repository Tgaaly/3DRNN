%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright by Richard Socher
% For questions, email richard @ socher .org
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath('main_functions/');
addpath(genpath('presegmentation/'));
addpath('primitiveBenchmark/');
addpath('estimation/');
addpath('mixtures/');
addpath('bow/');
addpath('matching/');
addpath('database/');
addpath('geometry/');
addpath('parameters/');
addpath('visualize/');
addpath('human_prior/');
addpath(genpath('geometry/kdtree'));
addpath('../drtoolbox/techniques/');
addpath('experiments/');
addpath('loaders/');
addpath('tools/');
addpath('tools/minFunc');
rmpath(genpath('C:\Users\tgaaly\Documents\MATLAB\vlfeat-0.9.17\toolbox'));
addpath(genpath('C:\Users\tgaaly\Documents\My Dropbox\PHD THESIS RGBD object segmentation and recognition\Matlab src\Libraries\libsvm-3.17'));
rmpath('C:\Program Files\MATLAB\R2014b\toolbox\stats\stats');
addpath(genpath('C:\Users\tgaaly\Documents\My Dropbox\PHD THESIS RGBD object segmentation and recognition\Matlab src\Libraries\libsvm-3.17\matlab'));

clear, close all
format compact
dbstop if error

% set to 1 if you have <5GB RAM or you just want to see what's going on for debugging/studying
flag_recompute=1;

tinyDatasetDebug = 1;
flag_autoencoder=1;
flag_dataaugment=1;
flag_linearsvm=1;
flag_nonlinearsvm=0;
flag_semanticspace = 1;
flag_validation=0;
flag_voxel01=0;

%%%%%%%%%%%%%%%%%%%%%%
% data set: stanford background data set from Gould et al.
mainDataSet = 'iccv09-1'
% setDataFolders
dataSet = 'train';

%%%%%%%%%%%%%%%%%%%%%%%
% minfunc options (not tuned)
options.Method = 'lbfgs';
options.MaxIter = 300;
optionsPT=options;
options.TolX = 1e-4;


%%%%%%%%%%%%%%%%%%%%%%%
%iccv09: 0 void   1,1 sky  0,2 tree   2,3 road  1,4 grass  1,5 water  1,6 building  2,7 mountain 2,8 foreground
set(0,'RecursionLimit',1000);
params.numLabels = 8; % we never predict 0 (void)
params.numFeat = 119;

%%%%%%%%%%%%%%%%%%%%%%
% model parameters (should be ok, found via CV)
params.numHid = 100;%50;
params.regPTC = 0.0001;
params.regC = params.regPTC;
params.LossPerError = 0.005;

%sigmoid activation function:
params.f = @(x) (1./(1 + exp(-x)));
params.df = @(z) (z .* (1 - z));


%%%%%%%%%%%%%%%%%%%%%%
% input and output file names
% neighNameStem = ['data/' mainDataSet '-allNeighborPairs'];
% if tinyDatasetDebug
%     neighName = [neighNameStem '_' dataSet '_tiny.mat'];
% else
%     neighName = [neighNameStem '_' dataSet '.mat'];
% end
% neighNameEval = [neighNameStem '_' dataSetEval '.mat'];

paramString = ['_hid' num2str(params.numHid ) '_PTC' num2str(params.regPTC)];
fullParamNameBegin = ['output/' mainDataSet '_fullParams'];
paramString = [paramString '_fullC' num2str(params.regC) '_L' num2str(params.LossPerError)];
fullTrainParamName = 'output/3DRNN.mat';%[fullParamNameBegin paramString '.mat'];
disp(['fullTrainParamName=' fullTrainParamName ])


%%%%%%%%%%%%%%%%%%%%%%
%load and pre-process training and testing dataset
%the resulting files are provided
% if ~exist(neighName,'file')
%     %%% first run preProData once for both train and eval!
%     dataSet='train';
%     preProSegFeatsAndSave(dataFolder,neighNameStem,trainList, neighName, dataSet, params,mainDataSet)
%     dataSet='eval';
%     preProSegFeatsAndSave(dataFolder,neighNameStem,evalList, neighNameEval, dataSet, params,mainDataSet)
% end

% if ~exist('allData','var')
%     load(neighName,'allData','goodPairsL','goodPairsR','badPairsL','badPairsR','onlyGoodL','onlyGoodR','onlyGoodLabels','allSegs','allSegLabels');
%     evalSet=load(neighNameEval,'allData','goodPairsL','goodPairsR','badPairsL','badPairsR','onlyGoodL','onlyGoodR','onlyGoodLabels','allSegs','allSegLabels');
% end

% start a matlab pool to use all CPU cores for full tree training
if isunix && matlabpool('size') == 0
    numCores = feature('numCores')
    if numCores==16
        numCores=8
    end
    matlabpool('open',numCores);
end


%%%%%%%%%%%%%%%%%%%%%%
% initialize parameters
initParams

%%%%%%%%%%%%%%%%%%%%%%
% TRAINING

% train Wbot layer and first RNN collapsing decisions with all possible correct and incorrect segment pairs
% this uses the training data more efficiently than the purely greedy full parser training that only looks at some pairs
% both could have been combined into one training as well.
[X decodeInfo] = param2stack(Wbot,W,Wcat);

% goodPairsL --> 120 x 1455
% goodPairsR --> 120 x 1455
% badPairsL --> 120 x 1455
% badPairsR --> 120 x 1455
% onlyGoodL --> 120 x 986
% onlyGoodL --> 120 x 986
% onlyGoodLabels --> 1 x 986

flag_runmycode=1;
if flag_runmycode==1
    params.numLabels = 2;%17; % we never predict 0 (void)
    params.numFeat = 1331;%9261;%2744;%119;
    params.numHid = 1332;%9262;
    cat_end_idx=30;
    subsample_models=1;
    gt_subsample=1;
    %load(['good_bad_pairs_' num2str(cat_end_idx) '_' num2str(subsample_models) '_' num2str(gt_subsample) '.mat']);
    %load(['good_bad_pairs_all_levels_11bins.mat']);
    load('D:/Datasets/good_bad_pairs__train.mat');
    if flag_semanticspace==1
        initParamsWithSemantic
    else
        initParams
    end
    [X decodeInfo] = param2stack(Wbot,W,Wcat);
end

%% split into training, validation and testing
if flag_validation==1
    numValidation = 5000;
    numTraining = size(goodPairsL,2) - numValidation;


    %numTesting = 1000;
    indices = [1:size(goodPairsL,2)];
    sel = randperm(length(goodPairsL),numTraining);%+numTesting);
    sel_tr = sel(1:numTraining);
    indices(sel_tr)=[];
    %sel = randperm(length(indices),numValidation);%+numTesting);
    sel_val = indices;%(sel(1:numValidation));

    assert(isempty(intersect(sel_val,sel_tr)))


    goodPairsL_tr = goodPairsL(:,sel_tr);
    goodPairsR_tr = goodPairsR(:,sel_tr);
    badPairsL_tr = badPairsL(:,sel_tr);
    badPairsR_tr = badPairsR(:,sel_tr);

    goodPairsL_val = goodPairsL(:,sel_val);
    goodPairsR_val = goodPairsR(:,sel_val);
    badPairsL_val = badPairsL(:,sel_val);
    badPairsR_val = badPairsR(:,sel_val);
    % all = [1:length(goodPairsL)];
    % all(sel_tr)=[];
    % sel_te = all;%sel(numTraining+1:end);
else
    sel_tr = 1:length(goodPairsL);
    numTraining = size(goodPairsL,2);
end

numGood=size(goodPairsL,2);
numBad=size(badPairsL,2);

layers = [500];
lambda=2;
if flag_autoencoder==1
    if flag_voxel01==1
        load(['D:/Datasets/autoencoder_' num2str(layers(1)) '_' num2str(length(layers)) '_' ...
            num2str(lambda) '_nodt.mat'],'model','goodPairsL_dr','goodPairsR_dr','badPairsL_dr','badPairsR_dr');
    else
           load(['D:/Datasets/autoencoder_' num2str(layers(1)) '_' num2str(length(layers)) '_' ...
            num2str(lambda) '_dt.mat'],'model','goodPairsL_dr','goodPairsR_dr','badPairsL_dr','badPairsR_dr');
    end
    goodPairsL = [goodPairsL_dr  ones(numGood,1)]';%[mappedRep(1:numGood,:)  ones(numGood,1)]';
    goodPairsR = [goodPairsR_dr  ones(numGood,1)]';%[mappedRep(numGood+1:2*numGood,:) ones(numGood,1)]';
    badPairsL = [badPairsL_dr  ones(numBad,1)]';%[mappedRep(2*numGood+1:2*numGood+numBad,:) ones(numBad,1)]';
    badPairsR = [badPairsR_dr  ones(numBad,1)]';%[mappedRep(2*numGood+numBad+1:end,:) ones(numBad,1)]';
%     training.goodPairsL = goodPairsL(:,sel_tr);
%     training.goodPairsR = goodPairsR(:,sel_tr);
%     training.badPairsL = badPairsL(:,sel_tr);
%     training.badPairsR = badPairsR(:,sel_tr);
%     validation.goodPairsL = goodPairsL(:,sel_val);
%     validation.goodPairsR = goodPairsR(:,sel_val);
%     validation.badPairsL = badPairsL(:,sel_val);
%     validation.badPairsR = badPairsR(:,sel_val);
    params.numFeat = 500;%600
    params.numHid = size(goodPairsL,1);%50;
    if flag_semanticspace==1
        initParamsWithSemantic
    else
        initParams
    end
    [X decodeInfo] = param2stack(Wbot,W,Wcat);
end


training.goodPairsL = goodPairsL(:,sel_tr);
training.goodPairsR = goodPairsR(:,sel_tr);
training.badPairsL = badPairsL(:,sel_tr);
training.badPairsR = badPairsR(:,sel_tr);

if flag_validation==1
    validation.goodPairsL = goodPairsL(:,sel_val);
    validation.goodPairsR = goodPairsR(:,sel_val);
    validation.badPairsL = badPairsL(:,sel_val);
    validation.badPairsR = badPairsR(:,sel_val);
end

clear badPairsR goodPairsL goodPairsR badPairsL

%% data augment
if flag_dataaugment==1
    goodPairsL2 = training.goodPairsR; % swap
    goodPairsR2 = training.goodPairsL;
    badPairsL2 = training.badPairsR;
    badPairsR2 = training.badPairsL;
    training.goodPairsL = [training.goodPairsL  goodPairsL2];
    clear goodPairsL2
    training.goodPairsR = [training.goodPairsR  goodPairsR2];
    clear goodPairsR2
    training.badPairsL = [training.badPairsL  badPairsL2];
    clear badPairsL2
    training.badPairsR = [training.badPairsR  badPairsR2];
    clear badPairsR2
end

%% train
if flag_validation==1  && flag_linearsvm==0 && flag_nonlinearsvm==0
   
    acc_val=0;
    all_acc_val=[];
    all_acc_tr=[];
    optionsPT.MaxIter=20;
    theindex=1;
    X_save=[];
    while theindex < 50% acc_val<99.99
        rand_ids = randperm(size(training.goodPairsL,2),size(training.goodPairsL,2));
        training.goodPairsL = training.goodPairsL(:,rand_ids);
        training.goodPairsR = training.goodPairsR(:,rand_ids);
        training.badPairsL = training.badPairsL(:,rand_ids);
        training.badPairsR = training.badPairsR(:,rand_ids);
        assert(length(rand_ids)==length(unique(rand_ids)))
        X = minFunc(@costFctInitWithCatWithSemantic,X,optionsPT,decodeInfo,training.goodPairsL,training.goodPairsR,...
            training.badPairsL,training.badPairsR,[],[],[],params);
        
        [Wbot,W,Wcat] = stack2param(X, decodeInfo);
        X_save{theindex}=X;    
        
        numGood = size(training.goodPairsL,2);%length(onlyGoodLabels);
        goodBotL= params.f(Wbot* training.goodPairsL);
        goodBotR= params.f(Wbot* training.goodPairsR);
        goodHid = params.f(W * [goodBotL; goodBotR; ones(1,numGood)]);

        % apply Wcat
        catHid = Wcat * [goodHid ; ones(1,numGood)];

        % for goot should all be [1 0]
        numGood = size(catHid,2);
        catOutGood = softmax(catHid);
        catOutGood_classIndex = find(catOutGood(1,:)>catOutGood(2,:));

        numBad = size(training.badPairsL,2);%length(onlyGoodLabels);
        badBotL= params.f(Wbot* training.badPairsL);
        badBotR= params.f(Wbot* training.badPairsR);
        badHid = params.f(W * [badBotL; badBotR; ones(1,numBad)]);

        % apply Wcat
        catHid = Wcat * [badHid ; ones(1,numBad)];

        numBad = size(catHid,2);
        catOutBad = softmax(catHid);
        catOutBad_classIndex = find(catOutBad(1,:)<catOutBad(2,:));
        acc_tr = (length(catOutGood_classIndex)+length(catOutBad_classIndex))/(numGood+numBad);
        all_acc_tr = [all_acc_tr acc_tr];

        numGood = size(validation.goodPairsL,2);%length(onlyGoodLabels);
        goodBotL= params.f(Wbot* validation.goodPairsL);
        goodBotR= params.f(Wbot* validation.goodPairsR);
        goodHid = params.f(W * [goodBotL; goodBotR; ones(1,numGood)]);

        % apply Wcat
        catHid = Wcat * [goodHid ; ones(1,numGood)];

        % for goot should all be [1 0]
        numGood = size(catHid,2);
        catOutGood = softmax(catHid);
        catOutGood_classIndex = find(catOutGood(1,:)>catOutGood(2,:));

        numBad = size(validation.badPairsL,2);%length(onlyGoodLabels);
        badBotL= params.f(Wbot* validation.badPairsL);
        badBotR= params.f(Wbot* validation.badPairsR);
        badHid = params.f(W * [badBotL; badBotR; ones(1,numBad)]);

        % apply Wcat
        catHid = Wcat * [badHid ; ones(1,numBad)];

        numBad = size(catHid,2);
        catOutBad = softmax(catHid);
        catOutBad_classIndex = find(catOutBad(1,:)<catOutBad(2,:));

        acc_val = (length(catOutGood_classIndex)+length(catOutBad_classIndex))/(numGood+numBad);
        disp(['over all accuracy = ' num2str((length(catOutGood_classIndex)+length(catOutBad_classIndex))/(numGood+numBad))]);
        all_acc_val = [all_acc_val acc_val];
        
        theindex = theindex+1;
    end
    
    figure(1), clf, hold on, plot(1-all_acc_val, 'r-'), plot(1-all_acc_tr, 'b-'), hold off
    X=X_save{30};
    
elseif flag_semanticspace==1 && flag_linearsvm==0 && flag_nonlinearsvm==0
    
    rand_ids = randperm(size(training.goodPairsL,2),size(training.goodPairsL,2));
    training.goodPairsL = training.goodPairsL(:,rand_ids);
    training.goodPairsR = training.goodPairsR(:,rand_ids);
    rand_ids = randperm(size(training.badPairsL,2),size(training.badPairsL,2));
    training.badPairsL = training.badPairsL(:,rand_ids);
    training.badPairsR = training.badPairsR(:,rand_ids);
    assert(length(rand_ids)==length(unique(rand_ids)))

    X = minFunc(@costFctInitWithCatWithSemantic,X,optionsPT,decodeInfo,training.goodPairsL,training.goodPairsR,...
        training.badPairsL,training.badPairsR,[],[],[],params);
    
    save(['trained_RNN_' num2str(numTraining) '_ae_' num2str(flag_autoencoder) '_da_' num2str(flag_dataaugment) '_semantic.mat']);
elseif flag_recompute==1  && flag_linearsvm==0 && flag_nonlinearsvm==0
    X = minFunc(@costFctInitWithCat,X,optionsPT,decodeInfo,training.goodPairsL,training.goodPairsR,...
        training.badPairsL,training.badPairsR,[],[],[],params);
    save(['trained_RNN_' num2str(numTraining) '_ae_' num2str(flag_autoencoder) '_da_' num2str(flag_dataaugment) '.mat']);
elseif flag_linearsvm==0 && flag_nonlinearsvm==0
    if exist(['trained_RNN_' num2str(numTraining) '_ae_' num2str(flag_autoencoder) '_da_' num2str(flag_dataaugment) '.mat'])
        load(['trained_RNN_' num2str(numTraining) '_ae_' num2str(flag_autoencoder) '_da_' num2str(flag_dataaugment) '.mat'],'X','decodeInfo');
    elseif flag_autoencoder==1 && flag_nonlinearsvm==0
        load('final_workspace_rnn_withAE_1.mat','X','Wbot','W','Wcat');
    else
        load('final_workspace_rnn_93.mat','X','Wbot','W','Wcat');
    end
end

[Wbot,W,Wcat] = stack2param(X, decodeInfo);
save(fullTrainParamName,'Wbot','W','Wout','Wcat','params','options')

%% test on training set just to verify
numGood = size(training.goodPairsL,2);%length(onlyGoodLabels);
goodBotL= params.f(Wbot* training.goodPairsL);
goodBotR= params.f(Wbot* training.goodPairsR);
goodHid = params.f(W * [goodBotL; goodBotR; ones(1,numGood)]);

% apply Wcat
catHid = Wcat * [goodHid ; ones(1,numGood)];

% for goot should all be [1 0]
catOutGood = softmax(catHid);
catOutGood_classIndex = find(catOutGood(1,:)>catOutGood(2,:));
disp([num2str(length(catOutGood_classIndex)) '/' num2str(size(catHid,2)) ' good correct --> ' num2str(length(catOutGood_classIndex)/size(catHid,2))]);

numBad = size(training.badPairsL,2);%length(onlyGoodLabels);
badBotL= params.f(Wbot* training.badPairsL);
badBotR= params.f(Wbot* training.badPairsR);
badHid = params.f(W * [badBotL; badBotR; ones(1,numBad)]);

% apply Wcat
catHid = Wcat * [badHid ; ones(1,numBad)];

catOutBad = softmax(catHid);
catOutBad_classIndex = find(catOutBad(1,:)<catOutBad(2,:));
disp([num2str(length(catOutBad_classIndex)) '/' num2str(size(catHid,2)) ' bad correct --> ' num2str(length(catOutBad_classIndex)/size(catHid,2))]);

disp('complete');

%% try linear SVM
if flag_linearsvm==1
    
    rand_ids = randperm(size(training.goodPairsL,2),size(training.goodPairsL,2));
    training.goodPairsL = training.goodPairsL(:,rand_ids);
    training.goodPairsR = training.goodPairsR(:,rand_ids);
    training.badPairsL = training.badPairsL(:,rand_ids);
    training.badPairsR = training.badPairsR(:,rand_ids);
    assert(length(rand_ids)==length(unique(rand_ids)))
    
    good = [training.goodPairsL(1:end-1,:)'  training.goodPairsR(1:end-1,:)']';
    bad = [training.badPairsL(1:end-1,:)'  training.badPairsR(1:end-1,:)']';
    training_data = [good  bad];
    rand_ids = randperm(size(training_data,2),2000);%,size(training_data,2));%size(training_data,2),size(training_data,2));
    training_data=training_data(:,rand_ids);
    actuallabels = [ones(size(good,2),1)*1; ones(size(bad,2),1)*-1];
    actuallabels = actuallabels(rand_ids);
    
    C=[0.05 0.1 0.5 1 1.5 2];
    accuracy = zeros(length(C),1)
    for cc=1:length(C)
        model_svm{cc} = svmtrain2(actuallabels, training_data', ['-t 0 -c ' num2str(C(cc))]);
        [~, acc, ~] = ...
            svmpredict2(actuallabels, training_data', model_svm{cc});
        accuracy(cc)=acc(1);
        cc
    end
    [~,idxx] = max(accuracy);
    good = [training.goodPairsL(1:end-1,:)'  training.goodPairsR(1:end-1,:)']';
    bad = [training.badPairsL(1:end-1,:)'  training.badPairsR(1:end-1,:)']';
    training_data = [good  bad];
    rand_ids = randperm(size(training_data,2),10000);%,size(training_data,2));%size(training_data,2),size(training_data,2));
    training_data=training_data(:,rand_ids);
    actuallabels = [ones(size(good,2),1)*1; ones(size(bad,2),1)*-1];
    actuallabels = actuallabels(rand_ids);
 
    model_svm = svmtrain2(actuallabels, training_data', ['-t 0 -c ' num2str(C(idxx))]);
    [predicted_label, acc, prob_estimates] = ...
        svmpredict2(actuallabels, training_data', model_svm);
    accuracy = acc
    
    %     LAMBDA=[0.001 0.01 0.05 0.07 0.09 0.1 0.5 0.9 1.3];
    %     acc_svm=zeros(length(LAMBDA),1);
    %     for ll=1:length(LAMBDA)
    %         [w_svm{ll}, b_svm{ll}, ~, ~] = svmtrain(training_data,actuallabels, LAMBDA(ll));
    %         [~,~,~, scores] = svmtrain(training_data, actuallabels, 0, 'model', w_svm{ll}, 'bias', b_svm{ll}, 'solver', 'none') ;
    %         scores(scores>0)=1;
    %         scores(scores<0)=-1;
    %         tmp = (scores' - actuallabels);
    %         acc_svm(ll) = length(find(tmp==0))/length(scores)
    %     end
    %     [~,idxx] = max(acc_svm);
    %     w_svm = w_svm{idxx};
    %     b_svm = b_svm{idxx};
    %SVMStruct = svmtrain(training_data',actuallabels,'boxconstraint',0.8,'tolkkt',1e-2,'kktviolationlevel',0.5)
elseif flag_nonlinearsvm==1
    good = [training.goodPairsL'  training.goodPairsR']';
    bad = [training.badPairsL'  training.badPairsR']';
    training_data = [good  bad];
    rand_ids = randperm(size(training_data,2),size(training_data,2));
    training_data=training_data(:,rand_ids);
    actuallabels = [ones(size(good,2),1)*1; ones(size(bad,2),1)*2];
    actuallabels = actuallabels(rand_ids);

    C=[0.05 0.1 0.5 1 1.5 2];
    accuracy = zeros(length(C),1)
    for cc=1:length(C)
        model_svm{cc} = svmtrain2(actuallabels, training_data', ['-t 2 -c ' num2str(C(cc))]);
        [~, acc, ~] = ...
            svmpredict2(actuallabels, training_data', model_svm{cc});
        accuracy(cc)=acc(1);
        cc
    end
    [~,idxx] = max(accuracy);
    good = [training.goodPairsL(1:end-1,:)'  training.goodPairsR(1:end-1,:)']';
    bad = [training.badPairsL(1:end-1,:)'  training.badPairsR(1:end-1,:)']';
    training_data = [good  bad];
    rand_ids = randperm(size(training_data,2),10000);%,size(training_data,2));%size(training_data,2),size(training_data,2));
    training_data=training_data(:,rand_ids);
    actuallabels = [ones(size(good,2),1)*1; ones(size(bad,2),1)*-1];
    actuallabels = actuallabels(rand_ids);
 
    model_svm = svmtrain2(actuallabels, training_data', ['-t 2 -c ' num2str(C(idxx))]);
    [predicted_label, acc, prob_estimates] = ...
        svmpredict2(actuallabels, training_data', model_svm);
    accuracy = acc
end
    
%% test on large test set

% load('good_bad_pairs_all_levels_11bins_72.mat','goodPairsL', 'goodPairsR', 'badPairsL', 'badPairsR');


%% test on completely new model
cat_end_idx=20;
clear training goodPairsL goodPairsR badPairsL badPairsR
% load(['good_bad_pairs_' num2str(cat_end_idx) '_' num2str(subsample_models) '_' num2str(gt_subsample) '_test.mat']);
% load(['good_bad_pairs_all_levels_11bins_72.mat'],'goodPairsL','goodPairsR','badPairsL','badPairsR');
load('D:/Datasets/good_bad_pairs__test.mat','goodPairsL','goodPairsR','badPairsL','badPairsR');

if flag_voxel01==1
    for i=1:size(goodPairsL,2)
       ids = find(goodPairsL(:,i)==0);
       goodPairsL_tmp=zeros(size(goodPairsL,1),1);
       goodPairsL_tmp(ids)=1;
       goodPairsL(:,i) = goodPairsL_tmp;
       
       ids = find(goodPairsR(:,i)==0);
       goodPairsR_tmp=zeros(size(goodPairsR,1),1);
       goodPairsR_tmp(ids)=1;
       goodPairsR(:,i) = goodPairsR_tmp;
    end
    
    for i=1:size(badPairsL,2)
       ids = find(badPairsL(:,i)==0);
       badPairsL_tmp=zeros(size(badPairsL,1),1);
       badPairsL_tmp(ids)=1;
       badPairsL(:,i)=badPairsL_tmp;
       
       ids = find(badPairsR(:,i)==0);
       badPairsR_tmp=zeros(size(badPairsR,1),1);
       badPairsR_tmp(ids)=1;
       badPairsR(:,i) = badPairsR_tmp;
    end
end

testing.goodPairsL = goodPairsL;%(:,sel_te);
testing.goodPairsR = goodPairsR;%(:,sel_te);
testing.badPairsL = badPairsL;%(:,sel_te);
testing.badPairsR = badPairsR;%(:,sel_te);


if flag_autoencoder==1
    [~, mappedFeat] = run_data_through_autoenc(model,testing.goodPairsL');
    testing.goodPairsL = [mappedFeat  ones(size(mappedFeat,1),1)]';
    [~, mappedFeat] = run_data_through_autoenc(model,testing.goodPairsR');
    testing.goodPairsR = [mappedFeat  ones(size(mappedFeat,1),1)]';
    [~, mappedFeat] = run_data_through_autoenc(model,testing.badPairsL');
    testing.badPairsL = [mappedFeat  ones(size(mappedFeat,1),1)]';
    [~, mappedFeat] = run_data_through_autoenc(model,testing.badPairsR');
    testing.badPairsR = [mappedFeat  ones(size(mappedFeat,1),1)]';
end

%random shuffle just to be sure
ri_good = randperm(size(testing.goodPairsL,2),size(testing.goodPairsL,2));
ri_bad = randperm(size(testing.badPairsL,2),size(testing.badPairsL,2));
testing.goodPairsL = testing.goodPairsL(:,ri_good);
testing.goodPairsR = testing.goodPairsR(:,ri_good);
testing.badPairsL = testing.badPairsL(:,ri_bad);
testing.badPairsR = testing.badPairsR(:,ri_bad);

%% test linear svm 
if flag_linearsvm==1
    good = [testing.goodPairsL'  testing.goodPairsR']';
    bad = [testing.badPairsL'  testing.badPairsR']';
    testing_data = [good  bad];
    actuallabels = [ones(size(good,2),1)*1; ones(size(bad,2),1)*2];
    rand_ids = randperm(size(testing_data,2),size(testing_data,2));%,size(training_data,2));%size(training_data,2),size(training_data,2));
    testing_data=testing_data(:,rand_ids);
    actuallabels = actuallabels(rand_ids);
    
    model_svm = svmtrain2(actuallabels, testing_data', ['-t 0 -c ' num2str(C(idxx))]);
    [predicted_label, acc, prob_estimates] = ...
        svmpredict2(actuallabels, training_data', model_svm);
    lin_svm_accuracy = acc
    
    return;
elseif flag_nonlinearsvm==1
    good = [testing.goodPairsL'  testing.goodPairsR']';
    bad = [testing.badPairsL'  testing.badPairsR']';
    testing_data = [good  bad];
    actuallabels = [ones(size(good,2),1)*1; ones(size(bad,2),1)*2];
    rand_ids = randperm(size(testing_data,2),size(testing_data,2));%,size(training_data,2));%size(training_data,2),size(training_data,2));
    testing_data=testing_data(:,rand_ids);
    actuallabels = actuallabels(rand_ids);
    
    model_svm = svmtrain2(actuallabels, testing_data', ['-t 2 -c ' num2str(C(idxx))]);
    [predicted_label, acc, prob_estimates] = ...
        svmpredict2(actuallabels, training_data', model_svm);
    nonlin_svm_accuracy = acc
    
    return;
end

%%
numGood = size(testing.goodPairsL,2);%length(onlyGoodLabels);
goodBotL= params.f(Wbot* testing.goodPairsL);
goodBotR= params.f(Wbot* testing.goodPairsR);
goodHid = params.f(W * [goodBotL; goodBotR; ones(1,numGood)]);

% apply Wcat
catHid = Wcat * [goodHid ; ones(1,numGood)];

% for goot should all be [1 0]
numGood = size(catHid,2);
catOutGood = softmax(catHid);
catOutGood_classIndex = find(catOutGood(1,:)>catOutGood(2,:));
disp([num2str(length(catOutGood_classIndex)) '/' num2str(size(catHid,2)) ' good correct --> ' num2str(length(catOutGood_classIndex)/size(catHid,2))]);

numBad = size(testing.badPairsL,2);%length(onlyGoodLabels);
badBotL= params.f(Wbot* testing.badPairsL);
badBotR= params.f(Wbot* testing.badPairsR);
badHid = params.f(W * [badBotL; badBotR; ones(1,numBad)]);

% apply Wcat
catHid = Wcat * [badHid ; ones(1,numBad)];

numBad = size(catHid,2);
catOutBad = softmax(catHid);
catOutBad_classIndex = find(catOutBad(1,:)<catOutBad(2,:));
disp([num2str(length(catOutBad_classIndex)) '/' num2str(size(catHid,2)) ' bad correct --> ' num2str(length(catOutBad_classIndex)/size(catHid,2))]);


disp(['over all accuracy = ' num2str((length(catOutGood_classIndex)+length(catOutBad_classIndex))/(numGood+numBad))]);

disp('complete');


