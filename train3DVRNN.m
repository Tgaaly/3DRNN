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

%%%%%%%%%%%%%%%%%%%%%%
% data set: stanford background data set from Gould et al.
mainDataSet = 'iccv09-1'
setDataFolders

%%%%%%%%%%%%%%%%%%%%%%%
% minfunc options (not tuned)
options.Method = 'lbfgs';
options.MaxIter = 1000;
optionsPT=options;
options.TolX = 1e-4;


%%%%%%%%%%%%%%%%%%%%%%%
%iccv09: 0 void   1,1 sky  0,2 tree   2,3 road  1,4 grass  1,5 water  1,6 building  2,7 mountain 2,8 foreground
set(0,'RecursionLimit',1000);
params.numLabels = 8; % we never predict 0 (void)
params.numFeat = 119;

%%%%%%%%%%%%%%%%%%%%%%
% model parameters (should be ok, found via CV)
params.numHid = 50;
params.regPTC = 0.0001;
params.regC = params.regPTC;
params.LossPerError = 0.05;

%sigmoid activation function:
params.f = @(x) (1./(1 + exp(-x)));
params.df = @(z) (z .* (1 - z));


%%%%%%%%%%%%%%%%%%%%%%
% input and output file names
neighNameStem = ['data/' mainDataSet '-allNeighborPairs'];
if tinyDatasetDebug
    neighName = [neighNameStem '_' dataSet '_tiny.mat'];
else
    neighName = [neighNameStem '_' dataSet '.mat'];
end
neighNameEval = [neighNameStem '_' dataSetEval '.mat'];

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
[X decodeInfo] = param2stack(Wbot,W,Wout,Wcat);

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
    params.numFeat = 2744;%119;
    cat_end_idx=15;
    subsample_models=1;
    gt_subsample=1;
    load(['good_bad_pairs_' num2str(cat_end_idx) '_' num2str(subsample_models) '_' num2str(gt_subsample) '.mat']);
    initParams
    [X decodeInfo] = param2stack(Wbot,W,Wout,Wcat);
%     goodPairsL2 = zeros(params.numFeat+1,length(goodPairsL));
%     goodPairsR2 = zeros(params.numFeat+1,length(goodPairsR));
%     badPairsL2 = zeros(params.numFeat+1,length(badPairsL));
%     badPairsR2 = zeros(params.numFeat+1,length(badPairsR));
%     for i=1:length(goodPairsL)
%         goodPairsL2(:,i) = [ goodPairsL{i}(:)  ; 1];%ones(length(goodPairsL{i}(:)),1)];
%         goodPairsR2(:,i) = [ goodPairsR{i}(:)  ; 1];%goodPairsR{i}(:);
%         badPairsL2(:,i) = [ badPairsL{i}(:)  ; 1];%badPairsL{i}(:);
%         badPairsR2(:,i) = [ badPairsR{i}(:)  ; 1];%badPairsR{i}(:);
%     end
%     goodPairsL = goodPairsL2;
%     goodPairsR = goodPairsR2;
%     badPairsL = badPairsL2;
%     badPairsR = badPairsR2;
%     onlyGoodL = goodPairsL2;
%     onlyGoodR = goodPairsR2;
%     onlyGoodLabels = [2*ones(size(onlyGoodR,2),1)]';% zeros(size(onlyGoodR,2),1)];
%     allSegLabels=ones(1,length(cell2mat(allSegLabel)));
%     allSegs2 = zeros(params.numFeat+1,length(allSegs));
%     for i=1:length(allSegs)
%         allSegs2(:,i) = [allSegs(:,i) ; 1];
%     end
%     allSegs = allSegs2;
end


X = minFunc(@costFctInitWithCat,X,optionsPT,decodeInfo,goodPairsL,goodPairsR,...
    badPairsL,badPairsR,[],[],allSegs,params);


%X = minFunc(@costFctFull,X,options,decodeInfo,allData,params);
[Wbot,W,Wout,Wcat] = stack2param(X, decodeInfo);

save(fullTrainParamName,'Wbot','W','Wout','Wcat','params','options')

%% test on training set just to verify
numGood = size(goodPairsL,2);%length(onlyGoodLabels);
goodBotL= params.f(Wbot* goodPairsL);
goodBotR= params.f(Wbot* goodPairsR);
goodHid = params.f(W * [goodBotL; goodBotR; ones(1,numGood)]);

% apply Wcat
catHid = Wcat * [goodHid ; ones(1,numGood)];

% for goot should all be [1 0]
catOutGood = softmax(catHid);
catOutGood_classIndex = find(catOutGood(1,:)>catOutGood(2,:));
disp([num2str(length(catOutGood_classIndex)) '/' num2str(size(catHid,2)) ' good correct --> ' num2str(length(catOutGood_classIndex)/size(catHid,2))]);

numBad = size(badPairsL,2);%length(onlyGoodLabels);
badBotL= params.f(Wbot* badPairsL);
badBotR= params.f(Wbot* badPairsR);
badHid = params.f(W * [badBotL; badBotR; ones(1,numBad)]);

% apply Wcat
catHid = Wcat * [badHid ; ones(1,numBad)];

catOutBad = softmax(catHid);
catOutBad_classIndex = find(catOutBad(1,:)<catOutBad(2,:));
disp([num2str(length(catOutBad_classIndex)) '/' num2str(size(catHid,2)) ' bad correct --> ' num2str(length(catOutBad_classIndex)/size(catHid,2))]);

disp('complete');




%%%%%%%%%%%%%%%%%%%%%
% run analysis
% test3DVRNN

% visualize trees
% visualizeImageTrees
% 