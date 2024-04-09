# Network Measurement Lab HW3 README

## Overview
This README provides instructions and results for HW3 of the Network Measurement Lab at Politecnico di Milano (POLIMI). The objective of this assignment is to analyze data from lecture and challenge datasets using Python scripts.

## Dataset
- All CSV files required for the assignment are located in the "dataset" folder.

## Task 1: Determining Optimal N from Lecture Dataset
1. Begin by executing the [Lecture.py](Lecture.py) script to determine the optimal value of N.
2. The script utilizes data from the "lecture-dataset" directory.
3. Upon running, the script generates a performance plot saved as [Lecture_dataset_Performance.png](result_figure/Lecture_dataset_Performance.png).
4. The optimal value of N is determined to be 7.
###### Lecture dataset Performance VS N:
![Lecture_dataset_Performance.png](result_figure%2FLecture_dataset_Performance.png)


## Task 2: Analyzing Challenge Dataset
1. Execute the [Challenge.py](Challenge.py) script to perform the analysis on the challenge dataset.
2. The script utilizes data from the "challenge-dataset" directory.
3. The results of the analysis for different values of K are as follows:

- **K=3:** Average Homogeneity = 1.0, Average Completeness = 0.9207704489070332, Average V-Measure = 0.9570309875905851, Average Error = 0.6
- **K=4:** Average Homogeneity = 1.0, Average Completeness = 0.8939475584314074, Average V-Measure = 0.9425396191559801, Average Error = 0.8
- **K=5:** Average Homogeneity = 1.0, Average Completeness = 0.9155182949377918, Average V-Measure = 0.9556124972392726, Average Error = 1.0

## Conclusion
This assignment demonstrates the use of Python scripts to analyze network measurement data, determine optimal parameters, and evaluate clustering performance.
