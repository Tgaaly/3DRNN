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

%% parameters
k=22;
cat_end_idx = 15;%20;
gt_subsample = 1;%2;
subsample_models = 1;

%% get model IDs
load('../3d_recog_by_parts_humanprior/trainIDs.mat');

%% load segments for each model
FLAGS=[];
FLAGS.flag_display=0;
FLAGS.flag_reSegment=0;
FLAGS.flag_save=1;
FLAGS.flag_showgraph=0;
FLAGS.flag_RunOrShowResults=1;
FLAGS.flag_redo=1;
FLAGS.flag_noise = 0;
FLAGS.flag_missingdata=0;
FLAGS.flag_humanprior=1;

idxGood=1;
idxBad=1;
idx_all=1;
startBoth=1;

bin_sz=0.15;%0.1, 0.15 - good, 0.2 - bad
id_k_mapping = [1:400 ; ones(1,400)*22]';% ; 103 21 ; 125 15];
% goodL = cell(10000,1);
% goodR = cell(10000,1);
% badL = cell(10000,1);
% badR = cell(10000,1);

cat_ID=[1:20:400 400];
idx_allData = 1;
startAllSegs = 1;
startOnlyGood=1;
startBoth = 1;
startBad = 1;
startGood = 1;
startAllSegs = 1;

for i=1:length(train_IDs)
    ID=train_IDs(i);
    
    %     [pcloud_raw, faces] = loadPcloud( ID );
    %     if pcloud_raw==-1
    %         continue;
    %     end
    %     pcloud_raw_facepts = tkConvertFaces2Points(pcloud_raw, faces);
    %     [normalv,normalf] = compute_normal(pcloud_raw',faces');
    %     [O1, Z1, G1, X1, clist1, options1, nbh1, segment_graph1, scale1, labeling_ours] = loadDecompositionFile(ID,k,id_k_mapping);
    
    [pcloud, faces] = loadPcloud( ID );
    pcloud_fcpts = tkConvertFaces2Points(pcloud, faces);
    
    load avg_gt_labels
    [gt] = loadGroundtruth_smart(ID,avg_gt_labels,cat_ID);
    
    [faces, ~, normals, segments, segments_sf, ~, ...
        saved_segments, labeling, pcloud_all, options] = fn_preseg(ID, k, 0.0, FLAGS);
    
    pcloud =[];
    for jj=1:length(segments)
        pcloud = [pcloud ; segments{jj}];
    end
    
    segment_graph = tkBuildSegmentGraph2(segments, pcloud, 0);
    
    for gg=1:length(gt)
        gt_label = gt{gg}+1;
        [gtlabels{gg},~]=tkMapToOurPointCloud(pcloud, pcloud_fcpts, gt_label);
    end
    
    for s=1:length(segments)
        [seg_voxels, ~] = fn_voxelize_shapes(segments{s}, bin_sz, 0);
        allSegs(:,startAllSegs) = [seg_voxels{1} ; 1];
        startAllSegs=startAllSegs+1;
        
        %get neighbors (adjacent to this one)
        neighbors_idx = find(segment_graph(:,1)==s | segment_graph(:,2)==s);
        %segment_graph(neighbors_idx,:)
        neighbors=[];
        for n=1:length(neighbors_idx)
            if  segment_graph(neighbors_idx(n),1)==s
                neighbors = [neighbors ; segment_graph(neighbors_idx(n),2)];
            else
                neighbors = [neighbors ; segment_graph(neighbors_idx(n),1)];
            end
        end
        neighbors = unique(neighbors);
        
        %get label of segment s
        for gg=1:length(gt)
            gtlabel_s{gg} = fn_getMajorityGroundTruthLabel(pcloud, gtlabels{gg}, segments{s});
        end
        gtlabel_s_mat = cell2mat(gtlabel_s);
        
        goodNeighbors = [];
        badNeighbors = [];
        for n=1:length(neighbors)
            
            figure(1), clf, hold on, plot3Dpoints(segments{s},'r.'), 
            plot3Dpoints(segments{neighbors(n)},'b.')
        
            
            for gg=1:length(gt)
                gtlabel_sn{gg} = fn_getMajorityGroundTruthLabel(pcloud, gtlabels{gg}, segments{neighbors(n)});
            end
            gtlabel_sn_mat = cell2mat(gtlabel_sn);
            
            lenSamePart = length(find(gtlabel_s_mat==gtlabel_sn_mat));
            percentageSamePart = lenSamePart / length(gtlabel_s_mat);
            
            if percentageSamePart >= 0.5
                goodNeighbors = [goodNeighbors ; neighbors(n)];
            else
                badNeighbors = [badNeighbors ; neighbors(n)];
            end
        end
        numGood = length(unique(goodNeighbors));
        numBad = length(unique(badNeighbors));
        numGBPairs = numGood * numBad;
        
%         for g = 1:numGood % ADDS THIS g TIMES!!!!!!!!!!!!!!?????????
%             % repeat the feature of this good segment (on the lefT)
%             [seg_voxel, ~] = fn_voxelize_shapes(segments{s}, bin_sz, 0);
%             onlyGoodL(:,startOnlyGood:startOnlyGood+numGood-1)= [repmat(seg_voxel{1},1,numGood ) ; ones(1,numGood)];
%             % put the features of the good neighboring segments here
%             [seg_voxel, ~] = fn_voxelize_shapes(segments(goodNeighbors), bin_sz, 0);
%             seg_voxel_mat = cell2mat(seg_voxel');
%             onlyGoodR(:,startOnlyGood:startOnlyGood+numGood-1)= [seg_voxel_mat ; ones(1,numGood)];
%             % add the labels for all these good pairs
%             %onlyGoodLabels(startOnlyGood:startOnlyGood+numGood-1) = segLabels(s);
%         end
%         startOnlyGood = startOnlyGood + numGood; %this does it for ALL SEGMENTS~!!!!!!!
%         
        % if there are good and bad adjacent neighboring segments!!
        if numGood>0
            
            % Carteian Product for [1 2 3] and [5 6] is [1 5 ; 2 5 ; 3 5 ;
            % 1 6 ; 2 6 ...etc.
            %gbPairNums = cartprod(goodNeighbors,badNeighbors);
            
            % these are the inputs to Wbot --> DOES THIS MEAN W BOTTOM???
            % NOW IT MAKES SENSE - these are the pairs that go in through
            % the bottom (at higher levels in the RNN tree we have to
            % consider different options for merging)
            
            % ADD GOOD PAIRS FEATURES
            [seg_voxel, ~] = fn_voxelize_shapes(segments{s}, bin_sz, 0);
            goodPairsL(:,startGood:startGood+length(goodNeighbors)-1)= [repmat(seg_voxel{1},1,length(goodNeighbors)) ; ones(1,length(goodNeighbors))];
            
            [seg_voxel, ~] = fn_voxelize_shapes(segments(goodNeighbors), bin_sz, 0);
            seg_voxel_mat = cell2mat(seg_voxel');
            goodPairsR(:,startGood:startGood+length(goodNeighbors)-1)= [seg_voxel_mat ; ones(1,length(goodNeighbors))];
        
            startGood = startGood + length(goodNeighbors);
        end
        if numBad>0
            % ADD BAD PAIRS FEATURES (NOTICE THE gbPairNums(:,2) indexing)
            % - cartesian product repeats the segments that are adjacent
            % but bad neighbors in the second column of gbPairNums. So
            % since its already repeated we just need to add these to
            % badPairsL
            [seg_voxel, ~] = fn_voxelize_shapes(segments{s}, bin_sz, 0);
            badPairsL(:,startBad:startBad+length(badNeighbors)-1)= [repmat(seg_voxel{1},1,length(badNeighbors)) ; ones(1,length(badNeighbors))];
            
            [seg_voxel, ~] = fn_voxelize_shapes(segments(badNeighbors), bin_sz, 0);
            seg_voxel_mat = cell2mat(seg_voxel');
            badPairsR(:,startBad:startBad+length(badNeighbors)-1)= [seg_voxel_mat ; ones(1,length(badNeighbors))];
            
            %startBoth = startBoth+numGBPairs;
            startBad = startBad + length(badNeighbors);
        end
        s
    end
    
    
    
%     for edge=1:size(segment_graph,1)
%         
%         for gg=1:length(gt)
%             gtlabel1{gg} = fn_getMajorityGroundTruthLabel(pcloud, gtlabels{gg}, segments{segment_graph(edge,1)});
%             gtlabel2{gg} = fn_getMajorityGroundTruthLabel(pcloud, gtlabels{gg}, segments{segment_graph(edge,2)});
%         end
%         gtlabel1_mat = cell2mat(gtlabel1);
%         gtlabel2_mat = cell2mat(gtlabel2);
%         
%         lenSamePart = length(find(gtlabel1_mat==gtlabel2_mat));
%         percentageSamePart = lenSamePart / length(gtlabel1_mat);
%         
%         if percentageSamePart >= 0.5
%             [voxelized_descrL, ~] = fn_voxelize_shapes(segments{segment_graph(edge,1)}, bin_sz, 0);
%             [voxelized_descrR, ~] = fn_voxelize_shapes(segments{segment_graph(edge,2)}, bin_sz, 0);
%             goodL{idxGood} = cell2mat(voxelized_descrL);
%             goodR{idxGood} = cell2mat(voxelized_descrR);
%             idxGood=idxGood+1;
%             
%             %                 figure(2), clf,  plot3Dpoints(segments{segment_graph(edge,1)},'r.'), hold on,
%             %                 plot3Dpoints(segments{segment_graph(edge,2)},'b.'), hold off
%             %                 disp('good');
%         else
%             [voxelized_descrL, ~] = fn_voxelize_shapes(segments{segment_graph(edge,1)}, bin_sz, 0);
%             [voxelized_descrR, ~] = fn_voxelize_shapes(segments{segment_graph(edge,2)}, bin_sz, 0);
%             badL{idxBad} = cell2mat(voxelized_descrL);
%             badR{idxBad} = cell2mat(voxelized_descrR);
%             idxBad=idxBad+1;
%             
%             %                 figure(3), clf, plot3Dpoints(segments{segment_graph(edge,1)},'r.'), hold on,
%             %                 plot3Dpoints(segments{segment_graph(edge,2)},'b.'), hold off
%             %                 disp('bad');
%         end
%     end
    i
end

numAllSegs = startAllSegs-1;
allSegs= allSegs(:,1:numAllSegs);
% allSegLabels= allSegLabels(1:numAllSegs);

% KEEP ONLY THE GOOD SEGMENTS IN SEPARATE DATA STRUCTURE - I GUESS FOR
% CONVENIENCE
% numOnlyGood = startOnlyGood-1;
% onlyGoodL = onlyGoodL(:,1:numOnlyGood);
% onlyGoodR = onlyGoodR(:,1:numOnlyGood);
%onlyGoodLabels= onlyGoodLabels(1:numOnlyGood);

% numGBPairsAll = startBoth-1;
% NOW THESE CONTAIN ALL POSSIBLE ADJACENCIES (PAIRS) OF GOOD AND BAD PAIRS
% OF NEIGHBORING ADJACENT SEGMENTS
% delete trailing zeros in pre-allocated matrix
goodPairsL = goodPairsL(:,1:startGood-1);
goodPairsR = goodPairsR(:,1:startGood-1);
badPairsL = badPairsL(:,1:startBad-1);
badPairsR = badPairsR(:,1:startBad-1);

save(['good_bad_pairs_' num2str(cat_end_idx) '_' num2str(subsample_models)  ...
    '_' num2str(gt_subsample) '.mat'],'goodPairsL','goodPairsR','badPairsL',...
    'badPairsR','allSegs');

disp('done preprocessing');
