# Network Measurement Lab HW3 README

## Overview
This README provides instructions and results for HW3 of the Network Measurement Lab at Politecnico di Milano (POLIMI). The objective of this assignment is to analyze data from lecture and challenge datasets using cluster methods.

## Dataset
- All CSV files required for the assignment are located in the "dataset" folder.

## Task 1: Determining Optimal N from Lecture Dataset
1. Begin by executing the [Lecture.py](Lecture.py) script to determine the optimal value of N.
2. The script utilizes data from the "lecture-dataset" directory.
3. Upon running, the script generates a performance plot saved as [Lecture_dataset_Performance.png](result_figure/Lecture_dataset_Performance.png).
4. The results for different N are as follows. We can see the optimal value of N is determined to be 7:

   * N=1, H=0.0, C=1.0, V=0.0, Error=-6
   * N=2, H=0.0, C=1.0, V=0.0, Error=-6
   * N=3, H=0.502, C=1.001, V=0.669, Error=-4
   * N=4, H=0.459, C=0.992, V=0.627, Error=-3
   * N=5, H=0.608, C=0.987, V=0.752, Error=-1
   * N=6, H=0.608, C=0.987, V=0.752, Error=-1
   * N=7, H=0.795, C=0.892, V=0.841, Error=1
   * N=8, H=0.802, C=0.838, V=0.820, Error=5
   * N=9, H=1.0, C=0.501, V=0.668, Error=29
   * N=10, H=1.0, C=0.237, V=0.384, Error=692

## Task 2: Analyzing Challenge Dataset
1. Execute the [Challenge.py](Challenge.py) script to perform the analysis on the challenge dataset.
2. The script utilizes data from the "challenge-dataset" directory.
3. The results of the analysis for different values of K are as follows:

   * K=2, Average H=1.0, Average C=0.7045206158343548, Average V=0.7913669176947777, Average Error=1.0
   * K=3, Average H=1.0, Average C=0.9092046198599684, Average V=0.9487343058921548, Average Error=0.4
   * K=4, Average H=1.0, Average C=0.9324160068172935, Average V=0.9638204441429764, Average Error=0.6
   * K=5, Average H=1.0, Average C=0.9312667246673367, Average V=0.9638058458642824, Average Error=0.8
   * K=6, Average H=1.0, Average C=0.9443515050002154, Average V=0.9713794060093172, Average Error=1.0
4. The script generates a performance plot with different K saved as [Challenge_dataset_Performance.png](result_figure/Challenge_dataset_Performance.png)

## Task 3: Analyzing Unlabeled Dataset
1. Execute the [Unlable.py](Unlable.py) script to estimate the number of different devices. **The Number of Devices is 93**.

## Conclusion
In this assignment, we employed cluster analysis techniques to analyze network measurement data. 

For Task 1, we utilized the lecture dataset to determine the optimal value of N, which represents the threshold for feature similarity. After analyzing the performance metrics for different N values, we found that N=7 yielded the best results. This indicates that when at least 7 features match between probes, the clustering performance is optimized. The generated performance plot [Lecture_dataset_Performance.png](result_figure/Lecture_dataset_Performance.png) illustrates the trend of performance metrics with varying N values.

For Task 2, we extended our analysis to the challenge dataset. By varying the number of clusters (K) from 2 to 6, we evaluated the clustering performance and computed average metrics including homogeneity, completeness, V-measure, and error. The performance plot [Challenge_dataset_Performance.png](result_figure/Challenge_dataset_Performance.png) demonstrates the change in these metrics as K increases. Notably, we observed high homogeneity and V-measure values across different K values, indicating strong clustering performance.

In Task 3, we applied our best-performing clustering technique to estimate the number of devices in the unlabeled dataset. By analyzing the unlabeled data, we determined that there are a total of 93 distinct devices present.

Overall, this assignment showcases the effectiveness of cluster analysis in identifying patterns within network measurement data. The results provide insights into the optimal parameters for clustering and demonstrate the utility of these techniques in network analysis applications.
