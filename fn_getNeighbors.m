function neighbors = fn_getNeighbors(segment_graph, s)


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