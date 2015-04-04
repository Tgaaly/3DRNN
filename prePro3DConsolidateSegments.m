clear, close all
dbstop if error

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
goodPairsL = zeros(dimFeat,20000);
goodPairsR = zeros(dimFeat,20000);
badPairsL = zeros(dimFeat,20000);
badPairsR = zeros(dimFeat,20000);


startGoodIdx = 1;
startBadIdx = 1;

for i=3:length(mergefiles)
    load([dirpath mergefiles(i).name])
    
    % compute voxels for good merges
    for m=1:length(merges)
        seg1 = merges{m}.pts1;
        seg2 = merges{m}.pts2;
        
        [seg_voxel1, ~] = fn_voxelize_shapes(seg1, bin_sz, 0);
        goodPairsL(:,startGoodIdx:startGoodIdx)= [seg_voxel1{1} ; 1];
        
        [seg_voxel2, ~] = fn_voxelize_shapes(seg2, bin_sz, 0);
        goodPairsR(:,startGoodIdx:startGoodIdx)= [seg_voxel2{1} ; 1];
        
        startGoodIdx = startGoodIdx + 1;
    end
    
    % compute voxels for bad merges
    if exist('badmerges')
        for m=1:length(badmerges)
            seg1 = badmerges{m}.pts1;
            seg2 = badmerges{m}.pts2;
            
            [seg_voxel1, ~] = fn_voxelize_shapes(seg1, bin_sz, 0);
            badPairsL(:,startBadIdx:startBadIdx)= [seg_voxel1{1} ; 1];
            
            [seg_voxel2, ~] = fn_voxelize_shapes(seg2, bin_sz, 0);
            badPairsR(:,startBadIdx:startBadIdx)= [seg_voxel2{1} ; 1];
            
            startBadIdx = startBadIdx + 1;
        end
    end
    clear merges badmerges
    i
end



goodPairsL = goodPairsL(:,1:startGoodIdx-1);
goodPairsR = goodPairsR(:,1:startGoodIdx-1);
badPairsL = badPairsL(:,1:startBadIdx-1);
badPairsR = badPairsR(:,1:startBadIdx-1);

save('good_bad_pairs_all_levels_11bins.mat','goodPairsL', 'goodPairsR', 'badPairsL', 'badPairsR');

disp('done')
