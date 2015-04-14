
addpath(genpath('../drtoolbox/'));

%tag='lowlevel'
flag_voxel01=1;

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

% load('final_workspace_rnn_93.mat','sel_tr');
% load('merges_20.mat','countmerges','countbadmerges');
% numGoodPairs20 = sum(cell2mat(countmerges));
% numBadPairs20 = sum(cell2mat(countbadmerges));
% selgood_tr = round(linspace(1,numGoodPairs20,4000));
% selbad_tr = round(linspace(1,numGoodPairs20,4000));

%%  randomly select some data to encode with
sel_tr  = randperm(size(goodPairsL,2),2000);%1000);
goodPairsL_randsel = goodPairsL(:,sel_tr);
goodPairsR_randsel = goodPairsR(:,sel_tr);
badPairsL_randsel = badPairsL(:,sel_tr);
badPairsR_randsel = badPairsR(:,sel_tr);

allPairs = [goodPairsL_randsel' ; goodPairsR_randsel' ; ...
    badPairsL_randsel' ; badPairsR_randsel'];

%%
layers = [500];
lambda=2;
[model, mappedRep] = train_autoencoder(allPairs, layers, 0.0001, 100 );   
    
% load(['data/autoencoder_' num2str(layers(1)) '_' num2str(length(layers)) '_' num2str(lambda) '_11bins.mat'],'model','mappedRep');
%load(['good_bad_pairs_all_levels_11bins.mat'],'goodPairsL','goodPairsR','badPairsL',...
%    'badPairsR');
%load(['D:/Datasets/good_bad_pairs__train.mat'],'goodPairsL', 'goodPairsR', 'badPairsL', 'badPairsR');

[reconX, goodPairsL_dr] = run_data_through_autoenc(model, goodPairsL');
[reconX, goodPairsR_dr] = run_data_through_autoenc(model, goodPairsR');
[reconX, badPairsL_dr] = run_data_through_autoenc(model, badPairsL');
[reconX, badPairsR_dr] = run_data_through_autoenc(model, badPairsR');
    
if flag_voxel01==1
    save(['D:/Datasets/autoencoder_' num2str(layers(1)) '_' num2str(length(layers)) '_' num2str(lambda) '_' 'nodt' '.mat']);
else
    save(['D:/Datasets/autoencoder_' num2str(layers(1)) '_' num2str(length(layers)) '_' num2str(lambda) '_' 'dt' '.mat']);
end
