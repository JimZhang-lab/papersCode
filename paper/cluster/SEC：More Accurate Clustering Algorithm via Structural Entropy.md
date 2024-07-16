

## 1.  Info and Summary

This paper proposes a new clustering algorithm called SEC (More Accurate Clustering Algorithm via Structural Entropy).

This algorithm improves existing clustering methods by overcoming(maybe) limitations in relying on local neighborhood info and using Euclidean distances or densities for similarity.

I implemented this algorithm but it did not seem to perfome as well as it did in this paper. Of course, it is also possible that my coding skill not enough.

## 2.  Point view in this paper

works in this paper:

SEC Algorithm: It mainly consists of three parts: sparse graph embedding, structural entropy extraction, and iterative pre-deletion and reallocation. It captures the global structural features of clustering instances by integrating structural entropy.

Graph Embedding: It constructs a sparse graph using the k - NN and thresholding strategies to avoid the misguidance of the complete graph on the relationship between data points and better obtain the structural information.

Encoding Tree Construction: The encoding tree is a hierarchical abstract tree structure that can reflect the intrinsic and nonlinear patterns between data points. By iteratively merging leaf nodes to construct the encoding tree, and using the three operations of COMBINE, DROP, and SINGUP to limit the height of the tree, better clustering partitions can be obtained.

Iterative Pre-deletion and Reallocation: It defines fringe points and fringe sets through projective distance, and proposes an iterative pre-deletion and reallocation method to identify and exclude the fringe sets during the clustering process, avoiding significant deviations of

## 3.  Code

