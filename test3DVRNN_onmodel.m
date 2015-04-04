clear, close all
dbstop if error
format compact

addpath(genpath('matching'));
addpath(genpath('testing'));
addpath(genpath('estimation'));
addpath(genpath('mixtures'));
addpath(genpath('presegmentation'));
addpath('loaders');
addpath('main_functions');
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
flag_autoencoder=0;

% %% get model IDs
cat_ID=[1:20:400 400];
load('../3d_recog_by_parts_humanprior/testIDs.mat');

k=22;%32%14;%35;%30%Number of segments (12-good for ant)
save('k.mat','k');
id_k_mapping = [1:400 ; ones(1,400)*22]';% ; 103 21 ; 125 15];

if flag_autoencoder==1
    load('final_workspace_rnn_withAE_1.mat','X','Wbot','W','Wcat');
else
    load('final_workspace_rnn_93.mat','X','Wbot','W','Wcat');
end

model_id = 1;
ID=test_IDs(model_id);

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


% test
bin_sz=0.2;
params.f = @(x) (1./(1 + exp(-x)));
merges = zeros(length(segments),length(segments));
for i=1:length(segments)
    neighbors1 = find(segment_graph(:,1)==i);%getallNeighbors(i,segment_graph, []);
    neighbors2 = find(segment_graph(:,2)==i);
    neighbors = unique([segment_graph(neighbors1,2) ; segment_graph(neighbors2,1)]);
    for j=1:length(segments)
        if ~isempty(find(neighbors==j))
            segi = fn_voxelize_shapes(segments{i},bin_sz,0);
            segj = fn_voxelize_shapes(segments{j},bin_sz,0);
            badBotL= params.f(Wbot* [segi{1} ; 1]);
            badBotR= params.f(Wbot* [segj{1} ; 1]);
            badHid = params.f(W * [badBotL; badBotR; 1]);
            
            % apply Wcat
            catHid = Wcat * [badHid ; 1];
            
            catOutBad = softmax(catHid);
            catOutBad_classIndex = find(catOutBad(1,:)>catOutBad(2,:));
            if isempty(catOutBad_classIndex)
                out = catOutBad(2);
            else
                out = catOutBad(1);
            end
            merges(i,j)=out;
        end
    end
end

% segment1 merge with anyone?
tomerge = [];
newmerges=[];
idx_newmerge=1;
alreadymerged = [];
for i=1:length(merges)
   [m,j] = max(merges(i,:));
   if isempty(find(alreadymerged==j))
       tomerge = [tomerge ; i j];
       newmerges{idx_newmerge}=[segments{i} ; segments{j}];
       alreadymerged = [alreadymerged ; j];
       idx_newmerge=idx_newmerge+1;
   end
end

% display first merges
figure(1),   clf, hold on, axis equal,
colors = parula(length(newmerges));
ri=randperm(length(newmerges),length(newmerges));
colors = colors(ri,:);
for i=1:length(newmerges)
   plot3( newmerges{i}(:,1), newmerges{i}(:,2), newmerges{i}(:,3), '.', 'Color', colors(i,:));
   %plot3Dpoints( segmentPairs{i}.actualseg2, 'b.');
   %segmentPairs{i}.mergeOrNot
end


% newsegments=[];
% idx_newsegments=1;
% for i=1:length(segments)
%
%     neighbors1 = find(segment_graph(:,1)==i);%getallNeighbors(i,segment_graph, []);
%     neighbors2 = find(segment_graph(:,2)==i);
%     neighbors = unique([segment_graph(neighbors1,2) ; segment_graph(neighbors2,1)]);
%
%     for j=1:length(neighbors)
%         segi = fn_voxelize_shapes(segments{i},bin_sz,0);
%         segj = fn_voxelize_shapes(segments{neighbors(j)},bin_sz,0);
%         badBotL= params.f(Wbot* [segi{1} ; 1]);
%         badBotR= params.f(Wbot* [segj{1} ; 1]);
%         badHid = params.f(W * [badBotL; badBotR; 1]);
%
%         % apply Wcat
%         catHid = Wcat * [badHid ; 1];
%
%         catOutBad = softmax(catHid);
%         catOutBad_classIndex = find(catOutBad(1,:)>catOutBad(2,:));
%         if isempty(catOutBad_classIndex)
%             out = 0;
%         else
%             out = catOutBad_classIndex;
%         end
%
%         if out==1
%            newsegments{idx_newsegments}=[segments{i} ; segments{neighbors(j)}];
%            idx_newsegments=idx_newsegments+1;
%         end
%     end
%     %     segmentPairs{i}.seg1 = fn_voxelize_shapes(segments{segment_graph(i,1)},bin_sz,0);
%     %     segmentPairs{i}.seg2 = fn_voxelize_shapes(segments{segment_graph(i,2)},bin_sz,0);%segments{segment_graph(i,2)};
%     %     segmentPairs{i}.actualseg1 =segments{segment_graph(i,1)};
%     %     segmentPairs{i}.actualseg2 =segments{segment_graph(i,2)};
%     %     %numBad = size(training.badPairsL,2);%length(onlyGoodLabels);
%     %
%     %     badBotL= params.f(Wbot* [segmentPairs{i}.seg1{1} ; 1]);
%     %     badBotR= params.f(Wbot* [segmentPairs{i}.seg2{1} ; 1]);
%     %     badHid = params.f(W * [badBotL; badBotR; 1]);
%     %
%     %     % apply Wcat
%     %     catHid = Wcat * [badHid ; 1];
%     %
%     %     catOutBad = softmax(catHid);
%     %     catOutBad_classIndex = find(catOutBad(1,:)>catOutBad(2,:));
%     %     %disp([num2str(length(catOutBad_classIndex)) '/' num2str(size(catHid,2)) ' bad correct --> ' num2str(length(catOutBad_classIndex)/size(catHid,2))]);
%     %     if isempty(catOutBad_classIndex)
%     %         segmentPairs{i}.mergeOrNot = 0;
%     %     else
%     %         segmentPairs{i}.mergeOrNot = catOutBad_classIndex;
%     %     end
% end

% display first merges
% figure(1),
% colors = parula(length(segments));
% for i=1:length(segmentPairs)
%    clf, hold on
%    plot3Dpoints( segmentPairs{i}.actualseg1, 'r.');
%    plot3Dpoints( segmentPairs{i}.actualseg2, 'b.');
%    segmentPairs{i}.mergeOrNot
% end


