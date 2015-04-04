
Code to form 3D object decomposition 

Code adapted from the paper:

Parsing Natural Scenes and Natural Language with Recursive Neural Networks, 
Richard Socher, Cliff Lin, Andrew Y. Ng, and Christopher D. Manning
The 28th International Conference on Machine Learning (ICML 2011)


Explanation of Code:
1. prePro3DSegmentData_alllevels.m        --> compute good and bad pairs of segments for all levels (including merges of over-segmented regions). This also voxelizes the segments

2. prePro3DAutoEncodeData.m               --> run the merge segements through an autoencoder to reduce dimensions

3. prePro3DConsolidateData.m              --> 

4. train3DVRNN.m                          --> train RNN on good and bad merges (currently this also does testing using trained model)

-------------------------------------------
Code
-------------------------------------------

For training and testing the full model run in matlab:

trainVRNN

For only testing with previously trained parameters (which doesn't require much RAM), run 

testVRNN

That should give an accuracy of 0.783473 on this fold.
Note that since this is a non-convex objective the final accuracy when you re-train the model may differ to that one.

The code is optimized for speed but uses a lot of RAM (especially the pre-training that looks at all possible pairs).
If you just want to run the code on a small machine for studying it, set tinyDatasetDebug = 1; in the top of trainVRNN

