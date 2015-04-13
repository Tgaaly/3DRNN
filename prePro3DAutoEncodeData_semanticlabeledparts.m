clear, close all

addpath(genpath('../drtoolbox/'));
addpath('C:\3D_Recog_by_parts_humanprior\3d_recog_by_parts_humanprior\geometry');
addpath('C:\3D_Recog_by_parts_humanprior\3d_recog_by_parts_humanprior\bow');
addpath('C:\3D_Recog_by_parts_humanprior\3d_recog_by_parts_humanprior\presegmentation\common');

load('D:/Datasets/all_semantic_labeled_parts_filtered.mat','all_segments','all_semantic_labels');

all_voxels=cell(3375,length(all_segments));
% all_segments_normalized=cell(length(all_segments),1);
for i=1:length(all_segments)
    [tmp, ~] = ...
        fn_voxelize_shapes_unitcube_DT(all_segments{i}, 0.15, 0);
    all_voxels{i}=tmp;
    i
%     if i==1000
%         save('temp_1000.mat');
%     elseif i==2255
%         save('temp_2255.mat');
%     end
end

save('D:/Datasets/autoencoder_semantic_labeled_parts.mat','all_voxels','all_semantic_labels');

disp('done');