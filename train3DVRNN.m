%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright by Richard Socher
% For questions, email richard @ socher .org
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
close all
dbstop if error

addpath(genpath('tools/'));

% set to 1 if you have <5GB RAM or you just want to see what's going on for debugging/studying
tinyDatasetDebug = 1;
flag_autoencoder=1;
flag_dataaugment=0;
flag_linearsvm=1;
flag_nonlinearsvm=0;

%%%%%%%%%%%%%%%%%%%%%%
% data set: stanford background data set from Gould et al.
mainDataSet = 'iccv09-1'
% setDataFolders
dataSet = 'train';

%%%%%%%%%%%%%%%%%%%%%%%
% minfunc options (not tuned)
options.Method = 'lbfgs';
options.MaxIter = 200;
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
    load(['good_bad_pairs_all_levels_11bins.mat']);
    initParams
    [X decodeInfo] = param2stack(Wbot,W,Wcat);
end

numTraining = 1000;
numTesting = 1000;
sel = randperm(length(goodPairsL),numTraining+numTesting);
sel_tr = sel(1:numTraining);
sel_te = sel(numTraining+1:end);

numGood=size(goodPairsL,2);
numBad=size(badPairsL,2);

layers = [500];
lambda=2;
if flag_autoencoder==1
    load(['data/autoencoder_' num2str(layers(1)) '_' num2str(length(layers)) '_' ...
        num2str(lambda) '_11bins2.mat'],'model','goodPairsL_dr','goodPairsR_dr','badPairsL_dr','badPairsR_dr');
    goodPairsL = [goodPairsL_dr  ones(numGood,1)]';%[mappedRep(1:numGood,:)  ones(numGood,1)]';
    goodPairsR = [goodPairsR_dr  ones(numGood,1)]';%[mappedRep(numGood+1:2*numGood,:) ones(numGood,1)]';
    badPairsL = [badPairsL_dr  ones(numBad,1)]';%[mappedRep(2*numGood+1:2*numGood+numBad,:) ones(numBad,1)]';
    badPairsR = [badPairsR_dr  ones(numBad,1)]';%[mappedRep(2*numGood+numBad+1:end,:) ones(numBad,1)]';
    training.goodPairsL = goodPairsL;
    training.goodPairsR = goodPairsR;
    training.badPairsL = badPairsL;
    training.badPairsR = badPairsR;
    params.numFeat = 500;
    params.numHid = size(goodPairsL,1);%50;
    initParams
    [X decodeInfo] = param2stack(Wbot,W,Wcat);
end


training.goodPairsL = goodPairsL(:,sel_tr);
clear goodPairsL
training.goodPairsR = goodPairsR(:,sel_tr);
clear goodPairsR
training.badPairsL = badPairsL(:,sel_tr);
clear badPairsL
training.badPairsR = badPairsR(:,sel_tr);
clear badPairsR

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
% X = minFunc(@costFctInitWithCat,X,optionsPT,decodeInfo,training.goodPairsL,training.goodPairsR,...
%     training.badPairsL,training.badPairsR,[],[],[],params);
if flag_autoencoder==1
    load('final_workspace_rnn_withAE_1.mat','X','Wbot','W','Wcat');
else
    load('final_workspace_rnn_93.mat','X','Wbot','W','Wcat');
end

% [Wbot,W,Wcat] = stack2param(X, decodeInfo);

% save(fullTrainParamName,'Wbot','W','Wout','Wcat','params','options')

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


%% testing on test set to verify

% numGood = size(testing.goodPairsL,2);%length(onlyGoodLabels);
% goodBotL= params.f(Wbot* testing.goodPairsL);
% goodBotR= params.f(Wbot* testing.goodPairsR);
% goodHid = params.f(W * [goodBotL; goodBotR; ones(1,numGood)]);
%
% % apply Wcat
% catHid = Wcat * [goodHid ; ones(1,numGood)];
%
% % for goot should all be [1 0]
% catOutGood = softmax(catHid);
% catOutGood_classIndex = find(catOutGood(1,:)>catOutGood(2,:));
% disp([num2str(length(catOutGood_classIndex)) '/' num2str(size(catHid,2)) ' good correct --> ' num2str(length(catOutGood_classIndex)/size(catHid,2))]);
%
% numBad = size(testing.badPairsL,2);%length(onlyGoodLabels);
% badBotL= params.f(Wbot* testing.badPairsL);
% badBotR= params.f(Wbot* testing.badPairsR);
% badHid = params.f(W * [badBotL; badBotR; ones(1,numBad)]);
%
% % apply Wcat
% catHid = Wcat * [badHid ; ones(1,numBad)];
%
% catOutBad = softmax(catHid);
% catOutBad_classIndex = find(catOutBad(1,:)<catOutBad(2,:));
% disp([num2str(length(catOutBad_classIndex)) '/' num2str(size(catHid,2)) ' bad correct --> ' num2str(length(catOutBad_classIndex)/size(catHid,2))]);
%
% disp('complete');

%% try linear SVM
if flag_linearsvm==1
    good = [training.goodPairsL'  training.goodPairsR']';
    bad = [training.badPairsL'  training.badPairsR']';
    training_data = [good  bad];
    actuallabels = [ones(size(good,2),1)*1; ones(size(bad,2),1)*2];
    SVMStruct = svmtrain(training_data',actuallabels)
elseif flag_nonlinearsvm==1
    good = [training.goodPairsL'  training.goodPairsR']';
    bad = [training.badPairsL'  training.badPairsR']';
    training_data = [good  bad];
    actuallabels = [ones(size(good,2),1)*1; ones(size(bad,2),1)*2];
    SVMStruct = svmtrain(training_data',actuallabels,'kernel_function','rbf')
end
    

%% try kernel SVM

%% test on completely new model
cat_end_idx=20;
clear training
% load(['good_bad_pairs_' num2str(cat_end_idx) '_' num2str(subsample_models) '_' num2str(gt_subsample) '_test.mat']);
load(['good_bad_pairs_all_levels_11bins.mat'],'goodPairsL','goodPairsR','badPairsL','badPairsR');
testing.goodPairsL = goodPairsL(:,sel_te);
testing.goodPairsR = goodPairsR(:,sel_te);
testing.badPairsL = badPairsL(:,sel_te);
testing.badPairsR = badPairsR(:,sel_te);


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

%% test linear svm 
if flag_linearsvm==1
    good = [testing.goodPairsL'  testing.goodPairsR']';
    bad = [testing.badPairsL'  testing.badPairsR']';
    testing_data = [good  bad];
    actuallabels = [ones(size(good,2),1)*1; ones(size(bad,2),1)*2];
    predlabels = svmclassify(SVMStruct, testing_data');
    corr=0;
    for i=1:length(actuallabels)
        if actuallabels(i)==predlabels(i)
            corr=corr+1;
        end
    end
    linear_svm_acc = corr / length(actuallabels)
elseif flag_nonlinearsvm==1
    good = [testing.goodPairsL'  testing.goodPairsR']';
    bad = [testing.badPairsL'  testing.badPairsR']';
    testing_data = [good  bad];
    actuallabels = [ones(size(good,2),1)*1; ones(size(bad,2),1)*2];
    predlabels = svmclassify(SVMStruct, testing_data');
    corr=0;
    for i=1:length(actuallabels)
        if actuallabels(i)==predlabels(i)
            corr=corr+1;
        end
    end
    nonlinear_svm_acc = corr / length(actuallabels)
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


%% test on full models
