
addpath(genpath('../drtoolbox/'));

cat_end_idx=30;
subsample_models=1;
gt_subsample=1;
load(['good_bad_pairs_' num2str(cat_end_idx) '_' num2str(subsample_models)  ...
    '_' num2str(gt_subsample) '.mat'],'goodPairsL','goodPairsR','badPairsL',...
    'badPairsR','allSegs');

allPairs = [goodPairsL' ; goodPairsR' ; badPairsL' ; badPairsR'];

layers = [500];
lambda=2;
[model, mappedRep] = train_autoencoder(allPairs, layers, 0.0, 100 );   
    
    
    
    
save(['../data/data_human/autoencoder_' num2str(layers(1)) '_' num2str(length(layers)) '_' num2str(lambda) '.mat'],'model','mappedX');
