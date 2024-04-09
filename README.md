# Network Measurement Lab HW3 README

## Overview
This README provides instructions and results for HW3 of the Network Measurement Lab at Politecnico di Milano (POLIMI). The objective of this assignment is to analyze data from lecture and challenge datasets using cluster method.

## Dataset
- All CSV files required for the assignment are located in the "dataset" folder.

## Task 1: Determining Optimal N from Lecture Dataset
1. Begin by executing the [Lecture.py](Lecture.py) script to determine the optimal value of N.
2. The script utilizes data from the "lecture-dataset" directory.
3. Upon running, the script generates a performance plot saved as [Lecture_dataset_Performance.png](result_figure/Lecture_dataset_Performance.png).
4. The result for different N are as following. We can see the optimal value of N is determined to be 7.

* N=1, H=0.0, C=1.0, V=0.0, Error=-6
* N=2, H=0.0, C=1.0, V=0.0, Error=-6
* N=3, H=0.5021962586540387, C=1.0000000000000007, V=0.6686160423592115, Error=-4
* N=4, H=0.4588688834606525, C=0.9916439637908914, V=0.6274119657987576, Error=-3
* N=5, H=0.607679732390285, C=0.9873802840322383, V=0.7523365648822261, Error=-1
* N=6, H=0.607679732390285, C=0.9873802840322383, V=0.7523365648822261, Error=-1
* N=7, H=0.7954766125942974, C=0.891809370588526, V=0.8408930131185066, Error=1
* N=8, H=0.8018508833757472, C=0.8383051731751866, V=0.8196729097382783, Error=5
* N=9, H=1.0000000000000002, C=0.5009518338648059, V=0.6675122046720224, Error=29
* N=10, H=0.9999999999999999, C=0.23727030435398055, V=0.38353834811846904, Error=692

## Task 2: Analyzing Challenge Dataset
1. Execute the [Challenge.py](Challenge.py) script to perform the analysis on the challenge dataset.
2. The script utilizes data from the "challenge-dataset" directory.
3. The results of the analysis for different values of K are as follows:

- K=2, Average H=1.0, Average C=0.7500938098068041, Average V=0.8140675301959324, Average Error=0.8
- K=3, Average H=1.0, Average C=0.7391194306142497, Average V=0.8439108098280821, Average Error=0.8
- K=4, Average H=1.0, Average C=0.9242231535391128, Average V=0.9583631379721371, Average Error=0.6
- K=5, Average H=1.0, Average C=0.9186309969804329, Average V=0.9572811827068881, Average Error=1.0
- K=6, Average H=1.0, Average C=0.9443515050002154, Average V=0.9713794060093172, Average Error=1.0
4. The script generates a performance plot with different K saved as [Challenge_dataset_Performance.png](result_figure/Challenge_dataset_Performance.png)

## Task 3: Analyzing Unlabeled  Dataset
1. Execute the [Unlable.py](Unlable.py) script to estimate the number of different devices. **The Number of Devices is 93**.

## Conclusion
This assignment demonstrates the use of Python scripts to analyze network measurement data, determine optimal parameters, and evaluate clustering performance.
