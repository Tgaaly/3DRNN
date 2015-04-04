
addpath(genpath('../drtoolbox/'));

cat_end_idx=30;
subsample_models=1;
gt_subsample=1;
% load(['good_bad_pairs_' num2str(cat_end_idx) '_' num2str(subsample_models)  ...
%     '_' num2str(gt_subsample) '.mat'],'goodPairsL','goodPairsR','badPairsL',...
%     'badPairsR','allSegs');
load(['good_bad_pairs_all_levels_11bins.mat'],'goodPairsL','goodPairsR','badPairsL',...
    'badPairsR','allSegs');
load('final_workspace_rnn_93.mat','sel_tr');

sel_tr2=sel_tr(1:5:end);
goodPairsL = goodPairsL(:,sel_tr2);
goodPairsR = goodPairsR(:,sel_tr2);
badPairsL = badPairsL(:,sel_tr2);
badPairsR = badPairsR(:,sel_tr2);

allPairs = [goodPairsL' ; goodPairsR' ; badPairsL' ; badPairsR'];

layers = [500];
lambda=2;
[model, mappedRep] = train_autoencoder(allPairs, layers, 0.0, 100 );   
    
% load(['data/autoencoder_' num2str(layers(1)) '_' num2str(length(layers)) '_' num2str(lambda) '_11bins.mat'],'model','mappedRep');
load(['good_bad_pairs_all_levels_11bins.mat'],'goodPairsL','goodPairsR','badPairsL',...
    'badPairsR','allSegs');
[reconX, goodPairsL_dr] = run_data_through_autoenc(model, goodPairsL');
[reconX, goodPairsR_dr] = run_data_through_autoenc(model, goodPairsR');
[reconX, badPairsL_dr] = run_data_through_autoenc(model, badPairsL');
[reconX, badPairsR_dr] = run_data_through_autoenc(model, badPairsR');
    
save(['data/autoencoder_' num2str(layers(1)) '_' num2str(length(layers)) '_' num2str(lambda) '_11bins2.mat']);
