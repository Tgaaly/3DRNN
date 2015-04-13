
addpath(genpath('../drtoolbox/'));

tag='lowlevel'

cat_end_idx=30;
subsample_models=1;
gt_subsample=1;
% load(['good_bad_pairs_' num2str(cat_end_idx) '_' num2str(subsample_models)  ...
%     '_' num2str(gt_subsample) '.mat'],'goodPairsL','goodPairsR','badPairsL',...
%     'badPairsR','allSegs');
% load(['good_bad_pairs_all_levels_11bins.mat'],'goodPairsL','goodPairsR','badPairsL',...
%     'badPairsR','allSegs');
% load(['D:/Datasets/good_bad_pairs_' tag '.mat'],'goodPairsL', 'goodPairsR', 'badPairsL', 'badPairsR');
load(['D:/Datasets/good_bad_pairs__train.mat'],'goodPairsL', 'goodPairsR', 'badPairsL', 'badPairsR');

%load('final_workspace_rnn_93.mat','sel_tr');
% load('merges_20.mat','countmerges','countbadmerges');
% numGoodPairs20 = sum(cell2mat(countmerges));
% numBadPairs20 = sum(cell2mat(countbadmerges));
% selgood_tr = round(linspace(1,numGoodPairs20,4000));
% selbad_tr = round(linspace(1,numGoodPairs20,4000));

%%  randomly select some data to encode with
% goodPairsL = goodPairsL(:,selgood_tr);
% goodPairsR = goodPairsR(:,selgood_tr);
% badPairsL = badPairsL(:,selbad_tr);
% badPairsR = badPairsR(:,selbad_tr);

allPairs = [goodPairsL' ; goodPairsR' ; badPairsL' ; badPairsR'];

%%
layers = [500];
lambda=2;
[model, mappedRep] = train_autoencoder(allPairs, layers, 0.0001, 100 );   
    
% load(['data/autoencoder_' num2str(layers(1)) '_' num2str(length(layers)) '_' num2str(lambda) '_11bins.mat'],'model','mappedRep');
%load(['good_bad_pairs_all_levels_11bins.mat'],'goodPairsL','goodPairsR','badPairsL',...
%    'badPairsR');
load(['D:/Datasets/autoencoder__train.mat'],'goodPairsL','goodPairsR','badPairsL',...
   'badPairsR');
[reconX, goodPairsL_dr] = run_data_through_autoenc(model, goodPairsL');
[reconX, goodPairsR_dr] = run_data_through_autoenc(model, goodPairsR');
[reconX, badPairsL_dr] = run_data_through_autoenc(model, badPairsL');
[reconX, badPairsR_dr] = run_data_through_autoenc(model, badPairsR');
    
save(['data/autoencoder_' num2str(layers(1)) '_' num2str(length(layers)) '_' num2str(lambda) '_11bins3.mat']);
