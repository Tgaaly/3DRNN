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


clear, close all
format compact
dbstop if error

% set to 1 if you have <5GB RAM or you just want to see what's going on for debugging/studying
flag_recompute=1;

tinyDatasetDebug = 1;
flag_autoencoder=1;
flag_dataaugment=1;
flag_linearsvm=0;
flag_nonlinearsvm=0;
flag_semanticspace = 1;
flag_validation=1;

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
%     training.goodPairsL = goodPairsL(:,sel_tr);
%     training.goodPairsR = goodPairsR(:,sel_tr);
%     training.badPairsL = badPairsL(:,sel_tr);
%     training.badPairsR = badPairsR(:,sel_tr);
%     validation.goodPairsL = goodPairsL(:,sel_val);
%     validation.goodPairsR = goodPairsR(:,sel_val);
%     validation.badPairsL = badPairsL(:,sel_val);
%     validation.badPairsR = badPairsR(:,sel_val);
    params.numFeat = 500;
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

validation.goodPairsL = goodPairsL(:,sel_val);
validation.goodPairsR = goodPairsR(:,sel_val);
validation.badPairsL = badPairsL(:,sel_val);
validation.badPairsR = badPairsR(:,sel_val);

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
if flag_validation==1
   
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
    X=X_save{25};
    
elseif flag_semanticspace==1
    
    X = minFunc(@costFctInitWithCatWithSemantic,X,optionsPT,decodeInfo,training.goodPairsL,training.goodPairsR,...
        training.badPairsL,training.badPairsR,[],[],[],params);
    
    save(['trained_RNN_' num2str(numTraining) '_ae_' num2str(flag_autoencoder) '_da_' num2str(flag_dataaugment) '_semantic.mat']);
elseif flag_recompute==1
    X = minFunc(@costFctInitWithCat,X,optionsPT,decodeInfo,training.goodPairsL,training.goodPairsR,...
        training.badPairsL,training.badPairsR,[],[],[],params);
    save(['trained_RNN_' num2str(numTraining) '_ae_' num2str(flag_autoencoder) '_da_' num2str(flag_dataaugment) '.mat']);
else
    if exist(['trained_RNN_' num2str(numTraining) '_ae_' num2str(flag_autoencoder) '_da_' num2str(flag_dataaugment) '.mat'])
        load(['trained_RNN_' num2str(numTraining) '_ae_' num2str(flag_autoencoder) '_da_' num2str(flag_dataaugment) '.mat'],'X','decodeInfo');
    elseif flag_autoencoder==1
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
    
%% test on large test set

load('good_bad_pairs_all_levels_11bins_72.mat','goodPairsL', 'goodPairsR', 'badPairsL', 'badPairsR');


%% test on completely new model
cat_end_idx=20;
clear training
% load(['good_bad_pairs_' num2str(cat_end_idx) '_' num2str(subsample_models) '_' num2str(gt_subsample) '_test.mat']);
load(['good_bad_pairs_all_levels_11bins_72.mat'],'goodPairsL','goodPairsR','badPairsL','badPairsR');
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


