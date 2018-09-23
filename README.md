# DBSCAN-Clustering-Algorithm-Implementation

For the purposes of this assignment, you will implement the DBSCAN clustering algorithm. You may not use libraries for this portion of your assignment. Additionally, you will gain experience with internal cluster evaluation metrics.
Input data (provided as training data) consists of 8580 text records in sparse format. No labels are provided.
For evaluation purposes (leaderboard ranking), we will use the Normalized Mutual Information Score (NMI), which is an external index metric for evaluating clustering solutions. Essentially, your task is to assign each of the instances in the input data to K clusters identified from 1 to K.
All objects in the training data set must be assigned to a cluster. Thus, you can either assign all noise points to cluster K+1 or apply post-processing after DBSCAN and assign noise points to the closest cluster.
