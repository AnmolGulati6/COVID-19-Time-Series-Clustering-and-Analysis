# COVID-19 Time Series Clustering and Analysis

## Description

This project, done as part of the CS-540 Intro to AI course at UW Madison. Focuses on Wisconsin and California to gain insights into the COVID-19 death patterns in these states. Performs hierarchical clustering using both single linkage and complete linkage, along with K-means clustering to group states based on parameter estimates

The main steps of the analysis include:

1. Data Extraction: The project extracts COVID-19 death data for different states from the provided "time_series_covid19_deaths_US.csv" dataset.

2. Parameter Estimation: The project calculates statistical parameters, such as mean, standard deviation, median, beta, and pho, for each state's time series data. These parameters serve as features for clustering.

3. Data Normalization: To ensure fair comparison, the parameter values are normalized to a range between 0 and 1.

4. Hierarchical Clustering: The project performs hierarchical clustering using both single linkage and complete linkage methods to identify clusters of states with similar COVID-19 death trends.

5. K-means Clustering: The project applies K-means clustering to group states based on the parameter estimates. The cluster centers are updated iteratively to optimize the clustering result.

The results of the clustering algorithms are presented in various output files, including "singleLinkage.txt," "completeLinkage.txt," and "k-meanClusters.txt." These files contain information about the clustering results for further analysis and interpretation.

The goal of this project is to gain insights into the similarities and differences in COVID-19 death trends among states, which can aid in formulating targeted public health policies and interventions.

## Dataset

The project utilizes the "time_series_covid19_deaths_US.csv" dataset, which contains COVID-19 death data for different states in the US. The data is preprocessed to exclude territories and cruise ships, focusing solely on states.

## Files

The repository includes the following files:

1. **main.py**: The main Python script containing the implementation of the analysis and clustering algorithms.
2. **CumulativeTimeSeries.txt**: Output file containing cumulative time series death data for "Wisconsin" and "California".
3. **DailyAdditionalDeaths.txt**: Output file containing daily additional deaths data for "Wisconsin" and "California".
4. **parameterEstimates.txt**: Output file containing parameter estimates (mean, std, median, beta, pho) for each state.
5. **singleLinkage.txt**: Output file containing **single linkage hierarchical clustering** results.
6. **completeLinkage.txt**: Output file containing **complete linkage hierarchical clustering** results.
7. **k-meanClusters.txt**: Output file containing **K-means clustering** results.
8. **clusterCenters.txt**: Output file containing the cluster centers obtained from K-means clustering.
9. **totalDistortion.txt**: Output file containing the **total distortion** value obtained from K-means clustering.

## Getting Started

To run the analysis and clustering algorithms, follow these steps:

1. Clone this GitHub repository to your local machine.
2. Ensure you have Python installed along with the required dependencies (Pandas and NumPy).
3. Place the "time_series_covid19_deaths_US.csv" dataset in the same directory as the "main.py" script.
4. Run the "main.py" script, and it will generate the output files as mentioned above.

