clear, close all
dbstop if error
format compact

%%
tag1=[];%'alllevel';%[];
tag2='train';%[];

%% add paths
addpath('../3d_recog_by_parts_humanprior/main_functions/');
addpath(genpath('../3d_recog_by_parts_humanprior/presegmentation/'));
addpath('../3d_recog_by_parts_humanprior/parameters/');
addpath('../3d_recog_by_parts_humanprior/');
addpath('../3d_recog_by_parts_humanprior/geometry');
addpath('../3d_recog_by_parts_humanprior/matching');
addpath('../3d_recog_by_parts_humanprior/human_prior');
addpath('../3d_recog_by_parts_humanprior/loaders');
addpath('../3d_recog_by_parts_humanprior/bow');
addpath('../3d_recog_by_parts_humanprior/visualize');

dirpath = 'data/merges/';
mergefiles = dir(dirpath);

bin_sz=0.2;%0.1, 0.15 - good, 0.2 - bad
dimFeat = 1332;%2744+1;%additional 1 for bias in network

allGoodMerges=[];
allBadMerges=[];
goodPairsL = zeros(dimFeat,50000);
goodPairsR = zeros(dimFeat,50000);
badPairsL = zeros(dimFeat,50000);
badPairsR = zeros(dimFeat,50000);

if strcmp(tag2,'train')==1
    load('../3d_recog_by_parts_humanprior/trainIDs.mat');
    train_IDs = [train_IDs 151 161 171 181 191];% 201];% 211 221 231 241 251 281 291 301 311];
    all_IDs = train_IDs;
    subsamp=1;
else
    load('../3d_recog_by_parts_humanprior/testIDs.mat');
    all_IDs = test_IDs;
    subsamp=5;
end

startGoodIdx = 1;
startBadIdx = 1;

countmerges = [];
countbadmerges = [];
counter=0;

for i=1:subsamp:length(all_IDs)%mergefiles)-10
%for i=3:length(mergefiles)-10
    ID = all_IDs(i);
    %load([dirpath mergefiles(i).name])
    load(['D:/Datasets/merges/merges_' num2str(ID) '.mat'],'merges','badmerges');
    
    % compute voxels for good merges
    for m=1:length(merges)
        seg1 = merges{m}.pts1;
        seg2 = merges{m}.pts2;
        
        [seg_voxel1, ~] = fn_voxelize_shapes_unitcube_DT(seg1, bin_sz, 0);
        goodPairsL(:,startGoodIdx:startGoodIdx)= [seg_voxel1{1} ; 1];
        
        [seg_voxel2, ~] = fn_voxelize_shapes_unitcube_DT(seg2, bin_sz, 0);
        goodPairsR(:,startGoodIdx:startGoodIdx)= [seg_voxel2{1} ; 1];
        
        startGoodIdx = startGoodIdx + 1;
        %m
    end
    %countmerges{i} = length(merges);
    
    % compute voxels for bad merges
    if exist('badmerges')
        for m=1:length(badmerges)
            seg1 = badmerges{m}.pts1;
            seg2 = badmerges{m}.pts2;
            
            [seg_voxel1, ~] = fn_voxelize_shapes_unitcube_DT(seg1, bin_sz, 0);
            badPairsL(:,startBadIdx:startBadIdx)= [seg_voxel1{1} ; 1];
            
            [seg_voxel2, ~] = fn_voxelize_shapes_unitcube_DT(seg2, bin_sz, 0);
            badPairsR(:,startBadIdx:startBadIdx)= [seg_voxel2{1} ; 1];
            
            startBadIdx = startBadIdx + 1;
            %m
        end
        %countbadmerges{i} = length(badmerges);

    end
    clear merges badmerges
    i
    counter=counter+1;
    if mod(counter,20)==0 && counter~=0
        save(['D:/data/temp_merges_' num2str(counter) '.mat']);
    end
end

%save(['data/merges_' tag1 '_' tag2 '_' num2str(length(all_IDs)) '.mat']);


goodPairsL = goodPairsL(:,1:startGoodIdx-1);
goodPairsR = goodPairsR(:,1:startGoodIdx-1);
badPairsL = badPairsL(:,1:startBadIdx-1);
badPairsR = badPairsR(:,1:startBadIdx-1);

save(['D:/Datasets/good_bad_pairs_' tag1 '_' tag2 '.mat'],'goodPairsL', 'goodPairsR', 'badPairsL', 'badPairsR');

disp('done')
