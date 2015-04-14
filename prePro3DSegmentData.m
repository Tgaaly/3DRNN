clear, close all
dbstop if error
format compact

%%
tag1 = 'lowlevel'
tag2 = 'test'

addpath(genpath('matching'));
addpath(genpath('testing'));
addpath(genpath('estimation'));
addpath(genpath('mixtures'));
addpath(genpath('presegmentation'));
addpath('loaders');
addpath('main_functions');

%flags
FLAGS.flag_display=0;
FLAGS.flag_reSegment=0;
FLAGS.flag_save=1;
FLAGS.flag_showgraph=0;
FLAGS.flag_RunOrShowResults=1;
FLAGS.flag_redo=1;
FLAGS.flag_noise = 0;
FLAGS.flag_missingdata=0;
FLAGS.flag_doDecomposition=1;

% %% get model IDs
cat_ID=[1:20:400 400];
if strcmp(tag2,'train')==1
    load('../3d_recog_by_parts_humanprior/trainIDs.mat');
    train_IDs = [train_IDs 151 161 171 181 191];% 201];% 211 221 231 241 251 281 291 301 311];
    all_IDs = train_IDs;
    subsample=1;
else
    load('../3d_recog_by_parts_humanprior/testIDs.mat');
    all_IDs = test_IDs;
    subsample=5;
end

k=22;%32%14;%35;%30%Number of segments (12-good for ant)
save('k.mat','k');
id_k_mapping = [1:400 ; ones(1,400)*22]';% ; 103 21 ; 125 15];

% subsample_models = 10;
% all_ID=[1:subsample_models:360 381:subsample_models:400];


for ii=1:subsample:length(all_IDs)
    tic
    ID=all_IDs(ii);
    
    [pcloud, faces] = loadPcloud( ID );
    pcloud_fcpts = tkConvertFaces2Points(pcloud, faces);
    
    load avg_gt_labels
    [gt] = loadGroundtruth_smart(ID,avg_gt_labels,cat_ID);
    
    [faces, ~, normals, segments, segments_sf, ~, ...
        saved_segments, labeling, pcloud_all, options] = fn_preseg(ID, k, 0.0, FLAGS);
    
    [~, ~, ~, X1, ~, ~, ~, ~, scale1, ~] = loadDecompositionFile(ID,k,id_k_mapping);
    
    
    pcloud =[];
    pcloud2segments = [];
    for jj=1:length(segments)
        pcloud = [pcloud ; segments{jj}];
        pcloud2segments = [pcloud2segments ; ones(length(segments{jj}),1)*jj];
    end
    
    minDiff=Inf;
    if ~exist(['../data/scales_' num2str(ID) '.mat'])
        for scale2=0.001:0.01:1
            pcloud_ours2 = ((X1./1000).*scale1).*scale2;
            [~,~,err]=icp(pcloud_ours2',pcloud_raw_facepts',1);
            diff=err;%sum(sum(abs(pcloud_ours2-pcloud_raw)))
            if scale2==0.001
                diff_last = diff;
            end
            if diff < minDiff
                minDiff=diff;
            end
            if diff > diff_last
                break;
            end
            diff_last = diff;
        end
        save(['../data/scales_' num2str(ID) '.mat'], 'scale2');
    else
        load(['../data/scales_' num2str(ID) '.mat'], 'scale2');
    end
    
    segment_graph = tkBuildSegmentGraph2(segments, pcloud, 0);
    
    %pcloud_ours=((X1./1000).*scale1).*scale2;
    pcloud2 = ((pcloud).*scale1).*scale2;
    
    %figure, hold on, plot3Dpoints(pcloud, 'r.'),plot3Dpoints(pcloud_fcpts,'b.');
    gtlabels=[];
    if ~exist(['../data/gtlabels_' num2str(ID) '.mat'])
        gt_label = gt{1}+1;
        [~,mapping_raw2ours]=tkMapToOurPointCloud(pcloud2, pcloud_fcpts, gt_label);
        
        %learnt mapping, now just get labels for out point clouds according to each groundtruth
        for gg=1:length(gt)
            gt_label = gt{gg}+1;
            tmp_gt = zeros(length(pcloud),1);
            for po=1:length(pcloud2)
                tmp_gt(po) = gt_label(mapping_raw2ours(po,2));
            end
            %gt_label = gt{gg}+1;
            %[gtlabels{gg},mapping_raw2ours]=tkMapToOurPointCloud(pcloud2, pcloud_fcpts(1:2:end,:), gt_label(1:2:end));
            gtlabels{gg} = tmp_gt;
            
            %[gtlabels2{gg},~]=tkMapToOurPointCloud(pcloud2, pcloud_fcpts(1:2:end,:), gt_label(1:2:end));
            %figure, clf, fn_plot_ground_truth(pcloud,gtlabels{gg},5)
        end
        save(['../data/gtlabels_' num2str(ID) '.mat'],'gtlabels');
    else
        load(['../data/gtlabels_' num2str(ID) '.mat'],'gtlabels');
    end
    %pcloud = ((pcloud)./scale2)./scale1;
    N = length(segments);
    
    nbh = zeros(length(segments),length(segments));
    for pp=1:length(segment_graph)
        nbh(segment_graph(pp,1),segment_graph(pp,2))=1;
    end
    
    %hierarchally go up the tree to find valid segment merges
    %find adjacent segments from graph and then check if they should be
    %merged or not, if they should be merged then merge them and try
    %new adjacent neighbors
    idx=1;
    idxBad=1;
    merges=cell(10000,1);
    badmerges=cell(10000,1);
    for s1=1:N
        for s2=s1:N
            
            if s1==s2 || nbh(s1,s2)==0
                continue;
            end
            
            gtlabel_s1 = cell(length(gtlabels),1);
            for gg=1:length(gtlabels)
                %gtlabel_s1{gg} = fn_getMajorityGroundTruthLabel(pcloud, gtlabels{gg}, segments{s1});
                indices=find(pcloud2segments==s1);
                gtlabel_s1{gg} = mode(gtlabels{gg}(indices));
            end
            %fn_plot_ground_truth(pcloud,gtlabels{1},5)
            gtlabel_s1_mat = cell2mat(gtlabel_s1);
            
            gtlabel_s2 = cell(length(gtlabels),1);
            for gg=1:length(gtlabels)
                %gtlabel_s2{gg} = fn_getMajorityGroundTruthLabel(pcloud, gtlabels{gg}, segments{s2});
                indices=find(pcloud2segments==s2);
                gtlabel_s2{gg} = mode(gtlabels{gg}(indices));
            end
            %fn_plot_ground_truth(pcloud,gtlabels{1},5)
            gtlabel_s2_mat = cell2mat(gtlabel_s2);
            
            lenSamePart = length(find(gtlabel_s1_mat==gtlabel_s2_mat));
            percentageSamePart = lenSamePart / length(gtlabel_s1_mat);
            
            % if valid merge, record and continue
            if percentageSamePart >= 0.5
                %figure(1), clf, hold on, plot3Dpoints(segments{s1},'r.'),plot3Dpoints(segments{s2},'b.');
                %save merge information
                merges{idx}.ind = [s1 s2];% idx];
                merges{idx}.ind1 = [s1];% idx];
                merges{idx}.ind2 = [s2];% idx];
                merges{idx}.model_id = ID;
                merges{idx}.pts1 = segments{s1};
                merges{idx}.pts2 = segments{s2};
                %increment index
                idx=idx+1;
            else
                badmerges{idxBad}.ind = [s1 s2];% idx];
                badmerges{idxBad}.ind1 = [s1];% idx];
                badmerges{idxBad}.ind2 = [s2];% idx];
                badmerges{idxBad}.model_id = ID;
                badmerges{idxBad}.pts1 = segments{s1};
                badmerges{idxBad}.pts2 = segments{s2};
                idxBad = idxBad + 1;
            end
        end
    end
    merges = merges(1:idx-1);
    badmerges = badmerges(1:idxBad-1);
   
    save(['D:/Datasets/merges/merges_' tag1 '_' tag2 '_' num2str(ID) '.mat'],'merges','badmerges');
    

end




% clear, close all
% dbstop if error
%
% %% add paths
% addpath('../3d_recog_by_parts_humanprior/main_functions/');
% addpath(genpath('../3d_recog_by_parts_humanprior/presegmentation/'));
% addpath('../3d_recog_by_parts_humanprior/parameters/');
% addpath('../3d_recog_by_parts_humanprior/');
% addpath('../3d_recog_by_parts_humanprior/geometry');
% addpath('../3d_recog_by_parts_humanprior/matching');
% addpath('../3d_recog_by_parts_humanprior/human_prior');
% addpath('../3d_recog_by_parts_humanprior/loaders');
% addpath('../3d_recog_by_parts_humanprior/bow');
% addpath('../3d_recog_by_parts_humanprior/visualize');
%
% %% parameters
% k=22;
% cat_end_idx = 30;%15;%20;
% gt_subsample = 1;%dddd2;
% subsample_models = 1;
%
% %% get model IDs
% load('../3d_recog_by_parts_humanprior/trainIDs.mat');
% train_IDs = [train_IDs 151 161 171 181 191 201 211 221 231 241 251 281 291 301 311];
%
% %% load segments for each model
% FLAGS=[];
% FLAGS.flag_display=0;
% FLAGS.flag_reSegment=0;
% FLAGS.flag_save=1;
% FLAGS.flag_showgraph=0;
% FLAGS.flag_RunOrShowResults=1;
% FLAGS.flag_redo=1;
% FLAGS.flag_noise = 0;
% FLAGS.flag_missingdata=0;
% FLAGS.flag_humanprior=1;
%
% idxGood=1;
% idxBad=1;
% idx_all=1;
% startBoth=1;
%
% bin_sz=0.15;%0.1, 0.15 - good, 0.2 - bad
% id_k_mapping = [1:400 ; ones(1,400)*22]';% ; 103 21 ; 125 15];
%
% cat_ID=[1:20:400 400];
% idx_allData = 1;
% startAllSegs = 1;
% startOnlyGood=1;
% startBoth = 1;
% startBad = 1;
% startGood = 1;
% startAllSegs = 1;
%
% idx_newsegments=1;
%
% for i=1:1%length(train_IDs)
%     ID=train_IDs(i);
%
%     %     [pcloud_raw, faces] = loadPcloud( ID );
%     %     if pcloud_raw==-1
%     %         continue;
%     %     end
%     %     pcloud_raw_facepts = tkConvertFaces2Points(pcloud_raw, faces);
%     %     [normalv,normalf] = compute_normal(pcloud_raw',faces');
%     %     [O1, Z1, G1, X1, clist1, options1, nbh1, segment_graph1, scale1, labeling_ours] = loadDecompositionFile(ID,k,id_k_mapping);
%
%     [pcloud, faces] = loadPcloud( ID );
%     pcloud_fcpts = tkConvertFaces2Points(pcloud, faces);
%
%     load avg_gt_labels
%     [gt] = loadGroundtruth_smart(ID,avg_gt_labels,cat_ID);
%
%     [faces, ~, normals, segments, segments_sf, ~, ...
%         saved_segments, labeling, pcloud_all, options] = fn_preseg(ID, k, 0.0, FLAGS);
%
%     [~, ~, ~, X1, ~, ~, ~, ~, scale1, ~] = loadDecompositionFile(ID,k,id_k_mapping);
%
%
%     pcloud =[];
%     for jj=1:length(segments)
%         pcloud = [pcloud ; segments{jj}];
%     end
%
%     minDiff=Inf;
%     if ~exist(['../data/scales_' num2str(ID) '.mat'])
%         for scale2=0.001:0.01:1
%             pcloud_ours2 = ((X1./1000).*scale1).*scale2;
%             [~,~,err]=icp(pcloud_ours2',pcloud_raw_facepts',1);
%             diff=err;%sum(sum(abs(pcloud_ours2-pcloud_raw)))
%             if scale2==0.001
%                 diff_last = diff;
%             end
%             if diff < minDiff
%                 minDiff=diff;
%             end
%             if diff > diff_last
%                 break;
%             end
%             diff_last = diff;
%         end
%         save(['../data/scales_' num2str(ID) '.mat'], 'scale2');
%     else
%         load(['../data/scales_' num2str(ID) '.mat'], 'scale2');
%     end
%
%     segment_graph = tkBuildSegmentGraph2(segments, pcloud, 0);
%     pcloud_ours=((X1./1000).*scale1).*scale2;
%     pcloud2 = ((pcloud).*scale1).*scale2;
%
%     %figure, hold on, plot3Dpoints(pcloud, 'r.'),plot3Dpoints(pcloud_fcpts,'b.');
%
%     for gg=1:length(gt)
%         gt_label = gt{gg}+1;
%         [gtlabels{gg},~]=tkMapToOurPointCloud(pcloud2, pcloud_fcpts, gt_label);
%         %figure, clf, fn_plot_ground_truth(pcloud,gtlabels{gg},5)
%         %
%         %figure, clf,
%         %fn_plot_ground_truth(pcloud_fcpts,gt_label,5)
%     end
%     %pcloud = ((pcloud)./scale2)./scale1;
%
%     for s=1:length(segments)
%         [seg_voxels, ~] = fn_voxelize_shapes(segments{s}, bin_sz, 0);
%         allSegs(:,startAllSegs) = [seg_voxels{1} ; 1];
%         startAllSegs=startAllSegs+1;
%
%         neighbors=fn_getNeighbors(segment_graph, s)
%
%         %get label of segment s
%         for gg=1:length(gt)
%             gtlabel_s{gg} = fn_getMajorityGroundTruthLabel(pcloud, gtlabels{gg}, segments{s});
%         end
%         %fn_plot_ground_truth(pcloud,gtlabels{1},5)
%         gtlabel_s_mat = cell2mat(gtlabel_s);
%
%         goodNeighbors = [];
%         badNeighbors = [];
%         for n=1:length(neighbors)
%
%             %figure(1), clf, hold on, plot3Dpoints(segments{s},'r.'),
%             %plot3Dpoints(segments{neighbors(n)},'b.')
%
%
%             for gg=1:length(gt)
%                 gtlabel_sn{gg} = fn_getMajorityGroundTruthLabel(pcloud, gtlabels{gg}, segments{neighbors(n)});
%             end
%             gtlabel_sn_mat = cell2mat(gtlabel_sn);
%
%             lenSamePart = length(find(gtlabel_s_mat==gtlabel_sn_mat));
%             percentageSamePart = lenSamePart / length(gtlabel_s_mat);
%
%             if percentageSamePart >= 0.5
%                 newsegments{idx_newsegments}.segments = [s neighbors(n)];
%                 neighborsS = fn_getNeighbors(segment_graph, s);
%                 neighborsNeighborOfS = fn_getNeighbors(segment_graph, neighbors(n));
%                 newsegments{idx_newsegments}.neighbors = [neighborsS(:) neighborsNeighborOfS(:)];
%                 idx_newsegments=idx_newsegments+1;
%                 goodNeighbors = [goodNeighbors ; neighbors(n)];
%             else
%                 badNeighbors = [badNeighbors ; neighbors(n)];
%             end
%         end
%         numGood = length(unique(goodNeighbors));
%         numBad = length(unique(badNeighbors));
%         numGBPairs = numGood * numBad;
%
%         % if there are good and bad adjacent neighboring segments!!
%         if numGood>0
%
%             % Carteian Product for [1 2 3] and [5 6] is [1 5 ; 2 5 ; 3 5 ;
%             % 1 6 ; 2 6 ...etc.
%             %gbPairNums = cartprod(goodNeighbors,badNeighbors);
%
%             % these are the inputs to Wbot --> DOES THIS MEAN W BOTTOM???
%             % NOW IT MAKES SENSE - these are the pairs that go in through
%             % the bottom (at higher levels in the RNN tree we have to
%             % consider different options for merging)
%
%             % ADD GOOD PAIRS FEATURES
%             [seg_voxel, ~] = fn_voxelize_shapes(segments{s}, bin_sz, 0);
%             goodPairsL(:,startGood:startGood+length(goodNeighbors)-1)= [repmat(seg_voxel{1},1,length(goodNeighbors)) ; ones(1,length(goodNeighbors))];
%
%             [seg_voxel, ~] = fn_voxelize_shapes(segments(goodNeighbors), bin_sz, 0);
%             seg_voxel_mat = cell2mat(seg_voxel');
%             goodPairsR(:,startGood:startGood+length(goodNeighbors)-1)= [seg_voxel_mat ; ones(1,length(goodNeighbors))];
%
%             startGood = startGood + length(goodNeighbors);
%         end
%         if numBad>0
%             % ADD BAD PAIRS FEATURES (NOTICE THE gbPairNums(:,2) indexing)
%             % - cartesian product repeats the segments that are adjacent
%             % but bad neighbors in the second column of gbPairNums. So
%             % since its already repeated we just need to add these to
%             % badPairsL
%             [seg_voxel, ~] = fn_voxelize_shapes(segments{s}, bin_sz, 0);
%             badPairsL(:,startBad:startBad+length(badNeighbors)-1)= [repmat(seg_voxel{1},1,length(badNeighbors)) ; ones(1,length(badNeighbors))];
%
%             [seg_voxel, ~] = fn_voxelize_shapes(segments(badNeighbors), bin_sz, 0);
%             seg_voxel_mat = cell2mat(seg_voxel');
%             badPairsR(:,startBad:startBad+length(badNeighbors)-1)= [seg_voxel_mat ; ones(1,length(badNeighbors))];
%
%             %startBoth = startBoth+numGBPairs;
%             startBad = startBad + length(badNeighbors);
%         end
%         s
%     end
%
%     %for the rest of the levels
%     for s=1:length(newsegments)
%         segmenti =[];
%         for si=1:length(newsegments{s}.segments)
%             segmenti = [segmenti ; segments{newsegments{s}.segments(si)}];
%         end
%     end
%
%
%     i
% end
%
% numAllSegs = startAllSegs-1;
% allSegs= allSegs(:,1:numAllSegs);
% % allSegLabels= allSegLabels(1:numAllSegs);
%
% % KEEP ONLY THE GOOD SEGMENTS IN SEPARATE DATA STRUCTURE - I GUESS FOR
% % CONVENIENCE
% % numOnlyGood = startOnlyGood-1;
% % onlyGoodL = onlyGoodL(:,1:numOnlyGood);
% % onlyGoodR = onlyGoodR(:,1:numOnlyGood);
% %onlyGoodLabels= onlyGoodLabels(1:numOnlyGood);
%
% % numGBPairsAll = startBoth-1;
% % NOW THESE CONTAIN ALL POSSIBLE ADJACENCIES (PAIRS) OF GOOD AND BAD PAIRS
% % OF NEIGHBORING ADJACENT SEGMENTS
% % delete trailing zeros in pre-allocated matrix
% goodPairsL = goodPairsL(:,1:startGood-1);
% goodPairsR = goodPairsR(:,1:startGood-1);
% badPairsL = badPairsL(:,1:startBad-1);
% badPairsR = badPairsR(:,1:startBad-1);
%
% save(['good_bad_pairs_' num2str(cat_end_idx) '_' num2str(subsample_models)  ...
%     '_' num2str(gt_subsample) '_alllevels.mat'],'goodPairsL','goodPairsR','badPairsL',...
%     'badPairsR','allSegs');
%
% disp('done preprocessing');



% clear, close all
% dbstop if error
% 
% %% add paths
% addpath('../3d_recog_by_parts_humanprior/main_functions/');
% addpath(genpath('../3d_recog_by_parts_humanprior/presegmentation/'));
% addpath('../3d_recog_by_parts_humanprior/parameters/');
% addpath('../3d_recog_by_parts_humanprior/');
% addpath('../3d_recog_by_parts_humanprior/geometry');
% addpath('../3d_recog_by_parts_humanprior/matching');
% addpath('../3d_recog_by_parts_humanprior/human_prior');
% addpath('../3d_recog_by_parts_humanprior/loaders');
% addpath('../3d_recog_by_parts_humanprior/bow');
% addpath('../3d_recog_by_parts_humanprior/visualize');
% 
% %% parameters
% k=22;
% cat_end_idx = 20;%15;%20;
% gt_subsample = 1;%2;
% subsample_models = 1;
% 
% %% get model IDs
% load('../3d_recog_by_parts_humanprior/trainIDs.mat');
% train_IDs = [train_IDs 151 161 171 181 191];% 201 211 221 231 241 251 281 291 301 311];
% 
% %% load segments for each model
% FLAGS=[];
% FLAGS.flag_display=0;
% FLAGS.flag_reSegment=0;
% FLAGS.flag_save=1;
% FLAGS.flag_showgraph=0;
% FLAGS.flag_RunOrShowResults=1;
% FLAGS.flag_redo=1;
% FLAGS.flag_noise = 0;
% FLAGS.flag_missingdata=0;
% FLAGS.flag_humanprior=1;
% 
% idxGood=1;
% idxBad=1;
% idx_all=1;
% startBoth=1;
% 
% bin_sz=0.2;%0.1, 0.15 - good, 0.2 - bad
% id_k_mapping = [1:400 ; ones(1,400)*22]';% ; 103 21 ; 125 15];
% 
% cat_ID=[1:20:400 400];
% idx_allData = 1;
% startAllSegs = 1;
% startOnlyGood=1;
% startBoth = 1;
% startBad = 1;
% startGood = 1;
% startAllSegs = 1;
% 
% for i=1:length(train_IDs)
%     ID=train_IDs(i);
%     
%     %     [pcloud_raw, faces] = loadPcloud( ID );
%     %     if pcloud_raw==-1
%     %         continue;
%     %     end
%     %     pcloud_raw_facepts = tkConvertFaces2Points(pcloud_raw, faces);
%     %     [normalv,normalf] = compute_normal(pcloud_raw',faces');
%     %     [O1, Z1, G1, X1, clist1, options1, nbh1, segment_graph1, scale1, labeling_ours] = loadDecompositionFile(ID,k,id_k_mapping);
%     
%     [pcloud, faces] = loadPcloud( ID );
%     pcloud_fcpts = tkConvertFaces2Points(pcloud, faces);
%     
%     load avg_gt_labels
%     [gt] = loadGroundtruth_smart(ID,avg_gt_labels,cat_ID);
%     
%     [faces, ~, normals, segments, segments_sf, ~, ...
%         saved_segments, labeling, pcloud_all, options] = fn_preseg(ID, k, 0.0, FLAGS);
%     
%     [~, ~, ~, X1, ~, ~, ~, ~, scale1, ~] = loadDecompositionFile(ID,k,id_k_mapping);
% 
%     
%     pcloud =[];
%     for jj=1:length(segments)
%         pcloud = [pcloud ; segments{jj}];
%     end
% 
%     minDiff=Inf;
%     if ~exist(['../data/scales_' num2str(ID) '.mat'])
%         for scale2=0.001:0.01:1
%             pcloud_ours2 = ((X1./1000).*scale1).*scale2;
%             [~,~,err]=icp(pcloud_ours2',pcloud_raw_facepts',1);
%             diff=err;%sum(sum(abs(pcloud_ours2-pcloud_raw)))
%             if scale2==0.001
%                 diff_last = diff;
%             end
%             if diff < minDiff
%                 minDiff=diff;
%             end
%             if diff > diff_last
%                 break;
%             end
%             diff_last = diff;
%         end
%         save(['../data/scales_' num2str(ID) '.mat'], 'scale2');
%     else
%         load(['../data/scales_' num2str(ID) '.mat'], 'scale2');
%     end
%     
%     segment_graph = tkBuildSegmentGraph2(segments, pcloud, 0);
%     pcloud_ours=((X1./1000).*scale1).*scale2;
%     pcloud2 = ((pcloud).*scale1).*scale2;
% 
%     %figure, hold on, plot3Dpoints(pcloud, 'r.'),plot3Dpoints(pcloud_fcpts,'b.');
%     
%     for gg=1:length(gt)
%         gt_label = gt{gg}+1;
%         [gtlabels{gg},~]=tkMapToOurPointCloud(pcloud2, pcloud_fcpts, gt_label);
%         %figure, clf, fn_plot_ground_truth(pcloud,gtlabels{gg},5)
%         %
%         %figure, clf, 
%         %fn_plot_ground_truth(pcloud_fcpts,gt_label,5)
%     end
%     %pcloud = ((pcloud)./scale2)./scale1;
%     
%     for s=1:length(segments)
%         [seg_voxels, ~] = fn_voxelize_shapes(segments{s}, bin_sz, 0);
%         allSegs(:,startAllSegs) = [seg_voxels{1} ; 1];
%         startAllSegs=startAllSegs+1;
%         
%         %get neighbors (adjacent to this one)
%         neighbors_idx = find(segment_graph(:,1)==s | segment_graph(:,2)==s);
%         %segment_graph(neighbors_idx,:)
%         neighbors=[];
%         for n=1:length(neighbors_idx)
%             if  segment_graph(neighbors_idx(n),1)==s
%                 neighbors = [neighbors ; segment_graph(neighbors_idx(n),2)];
%             else
%                 neighbors = [neighbors ; segment_graph(neighbors_idx(n),1)];
%             end
%         end
%         neighbors = unique(neighbors);
%         
%         %get label of segment s
%         for gg=1:length(gt)
%             gtlabel_s{gg} = fn_getMajorityGroundTruthLabel(pcloud, gtlabels{gg}, segments{s});
%         end
%         %fn_plot_ground_truth(pcloud,gtlabels{1},5)
%         gtlabel_s_mat = cell2mat(gtlabel_s);
%         
%         goodNeighbors = [];
%         badNeighbors = [];
%         for n=1:length(neighbors)
%             
%             %figure(1), clf, hold on, plot3Dpoints(segments{s},'r.'),
%             %plot3Dpoints(segments{neighbors(n)},'b.')
%             
%             
%             for gg=1:length(gt)
%                 gtlabel_sn{gg} = fn_getMajorityGroundTruthLabel(pcloud, gtlabels{gg}, segments{neighbors(n)});
%             end
%             gtlabel_sn_mat = cell2mat(gtlabel_sn);
%             
%             lenSamePart = length(find(gtlabel_s_mat==gtlabel_sn_mat));
%             percentageSamePart = lenSamePart / length(gtlabel_s_mat);
%             
%             if percentageSamePart >= 0.5
%                 goodNeighbors = [goodNeighbors ; neighbors(n)];
%             else
%                 badNeighbors = [badNeighbors ; neighbors(n)];
%             end
%         end
%         numGood = length(unique(goodNeighbors));
%         numBad = length(unique(badNeighbors));
%         numGBPairs = numGood * numBad;
%         
%         %         for g = 1:numGood % ADDS THIS g TIMES!!!!!!!!!!!!!!?????????
%         %             % repeat the feature of this good segment (on the lefT)
%         %             [seg_voxel, ~] = fn_voxelize_shapes(segments{s}, bin_sz, 0);
%         %             onlyGoodL(:,startOnlyGood:startOnlyGood+numGood-1)= [repmat(seg_voxel{1},1,numGood ) ; ones(1,numGood)];
%         %             % put the features of the good neighboring segments here
%         %             [seg_voxel, ~] = fn_voxelize_shapes(segments(goodNeighbors), bin_sz, 0);
%         %             seg_voxel_mat = cell2mat(seg_voxel');
%         %             onlyGoodR(:,startOnlyGood:startOnlyGood+numGood-1)= [seg_voxel_mat ; ones(1,numGood)];
%         %             % add the labels for all these good pairs
%         %             %onlyGoodLabels(startOnlyGood:startOnlyGood+numGood-1) = segLabels(s);
%         %         end
%         %         startOnlyGood = startOnlyGood + numGood; %this does it for ALL SEGMENTS~!!!!!!!
%         %
%         % if there are good and bad adjacent neighboring segments!!
%         if numGood>0
%             
%             % Carteian Product for [1 2 3] and [5 6] is [1 5 ; 2 5 ; 3 5 ;
%             % 1 6 ; 2 6 ...etc.
%             %gbPairNums = cartprod(goodNeighbors,badNeighbors);
%             
%             % these are the inputs to Wbot --> DOES THIS MEAN W BOTTOM???
%             % NOW IT MAKES SENSE - these are the pairs that go in through
%             % the bottom (at higher levels in the RNN tree we have to
%             % consider different options for merging)
%             
%             % ADD GOOD PAIRS FEATURES
%             [seg_voxel, ~] = fn_voxelize_shapes(segments{s}, bin_sz, 0);
%             goodPairsL(:,startGood:startGood+length(goodNeighbors)-1)= [repmat(seg_voxel{1},1,length(goodNeighbors)) ; ones(1,length(goodNeighbors))];
%             
%             [seg_voxel, ~] = fn_voxelize_shapes(segments(goodNeighbors), bin_sz, 0);
%             seg_voxel_mat = cell2mat(seg_voxel');
%             goodPairsR(:,startGood:startGood+length(goodNeighbors)-1)= [seg_voxel_mat ; ones(1,length(goodNeighbors))];
%             
%             startGood = startGood + length(goodNeighbors);
%         end
%         if numBad>0
%             % ADD BAD PAIRS FEATURES (NOTICE THE gbPairNums(:,2) indexing)
%             % - cartesian product repeats the segments that are adjacent
%             % but bad neighbors in the second column of gbPairNums. So
%             % since its already repeated we just need to add these to
%             % badPairsL
%             [seg_voxel, ~] = fn_voxelize_shapes(segments{s}, bin_sz, 0);
%             badPairsL(:,startBad:startBad+length(badNeighbors)-1)= [repmat(seg_voxel{1},1,length(badNeighbors)) ; ones(1,length(badNeighbors))];
%             
%             [seg_voxel, ~] = fn_voxelize_shapes(segments(badNeighbors), bin_sz, 0);
%             seg_voxel_mat = cell2mat(seg_voxel');
%             badPairsR(:,startBad:startBad+length(badNeighbors)-1)= [seg_voxel_mat ; ones(1,length(badNeighbors))];
%             
%             %startBoth = startBoth+numGBPairs;
%             startBad = startBad + length(badNeighbors);
%         end
%         s
%     end
%     
%     
%     
%     %     for edge=1:size(segment_graph,1)
%     %
%     %         for gg=1:length(gt)
%     %             gtlabel1{gg} = fn_getMajorityGroundTruthLabel(pcloud, gtlabels{gg}, segments{segment_graph(edge,1)});
%     %             gtlabel2{gg} = fn_getMajorityGroundTruthLabel(pcloud, gtlabels{gg}, segments{segment_graph(edge,2)});
%     %         end
%     %         gtlabel1_mat = cell2mat(gtlabel1);
%     %         gtlabel2_mat = cell2mat(gtlabel2);
%     %
%     %         lenSamePart = length(find(gtlabel1_mat==gtlabel2_mat));
%     %         percentageSamePart = lenSamePart / length(gtlabel1_mat);
%     %
%     %         if percentageSamePart >= 0.5
%     %             [voxelized_descrL, ~] = fn_voxelize_shapes(segments{segment_graph(edge,1)}, bin_sz, 0);
%     %             [voxelized_descrR, ~] = fn_voxelize_shapes(segments{segment_graph(edge,2)}, bin_sz, 0);
%     %             goodL{idxGood} = cell2mat(voxelized_descrL);
%     %             goodR{idxGood} = cell2mat(voxelized_descrR);
%     %             idxGood=idxGood+1;
%     %
%     %             %                 figure(2), clf,  plot3Dpoints(segments{segment_graph(edge,1)},'r.'), hold on,
%     %             %                 plot3Dpoints(segments{segment_graph(edge,2)},'b.'), hold off
%     %             %                 disp('good');
%     %         else
%     %             [voxelized_descrL, ~] = fn_voxelize_shapes(segments{segment_graph(edge,1)}, bin_sz, 0);
%     %             [voxelized_descrR, ~] = fn_voxelize_shapes(segments{segment_graph(edge,2)}, bin_sz, 0);
%     %             badL{idxBad} = cell2mat(voxelized_descrL);
%     %             badR{idxBad} = cell2mat(voxelized_descrR);
%     %             idxBad=idxBad+1;
%     %
%     %             %                 figure(3), clf, plot3Dpoints(segments{segment_graph(edge,1)},'r.'), hold on,
%     %             %                 plot3Dpoints(segments{segment_graph(edge,2)},'b.'), hold off
%     %             %                 disp('bad');
%     %         end
%     %     end
%     i
% end
% 
% numAllSegs = startAllSegs-1;
% allSegs= allSegs(:,1:numAllSegs);
% % allSegLabels= allSegLabels(1:numAllSegs);
% 
% % KEEP ONLY THE GOOD SEGMENTS IN SEPARATE DATA STRUCTURE - I GUESS FOR
% % CONVENIENCE
% % numOnlyGood = startOnlyGood-1;
% % onlyGoodL = onlyGoodL(:,1:numOnlyGood);
% % onlyGoodR = onlyGoodR(:,1:numOnlyGood);
% %onlyGoodLabels= onlyGoodLabels(1:numOnlyGood);
% 
% % numGBPairsAll = startBoth-1;
% % NOW THESE CONTAIN ALL POSSIBLE ADJACENCIES (PAIRS) OF GOOD AND BAD PAIRS
% % OF NEIGHBORING ADJACENT SEGMENTS
% % delete trailing zeros in pre-allocated matrix
% goodPairsL = goodPairsL(:,1:startGood-1);
% goodPairsR = goodPairsR(:,1:startGood-1);
% badPairsL = badPairsL(:,1:startBad-1);
% badPairsR = badPairsR(:,1:startBad-1);
% 
% save(['good_bad_pairs_' num2str(cat_end_idx) '_' num2str(subsample_models)  ...
%     '_' num2str(gt_subsample) '.mat'],'goodPairsL','goodPairsR','badPairsL',...
%     'badPairsR','allSegs');
% 
% disp('done preprocessing');
