🔹 Scenario 1 – Clustering using K-Means

Dataset (Kaggle – Public)
Mall Customer Segmentation Dataset
Dataset Link: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

In this scenario, customers are grouped into clusters based on similarity using the K-Means algorithm. The Mall Customer dataset is used with features like Annual Income, Spending Score, and Age. First, required libraries are imported and the dataset is loaded, followed by preprocessing steps such as handling missing values and scaling the data. The Elbow Method is used to determine the optimal number of clusters (K), after which K-Means is applied to assign cluster labels to each data point. The clusters are visualized using scatter plots along with centroids. The model is evaluated using Inertia and Silhouette Score to measure cluster quality. Analysis is performed to understand how different K values affect clustering and to identify customer segments like high-income high-spending groups. Overall, K-Means is simple and effective but assumes spherical clusters and does not handle overlapping data well.

🔹 Scenario 2 – Clustering using GMM

Dataset (Same / Alternative Dataset)
Same dataset (or any numerical dataset)

In this scenario, clustering is performed using the Gaussian Mixture Model, which provides probabilistic cluster assignments. The same dataset is used, and preprocessing steps such as cleaning and scaling are applied. GMM uses the Expectation-Maximization algorithm to fit the data, and the optimal number of components is selected using AIC and BIC metrics. The model predicts the probability of each data point belonging to different clusters, and final labels are assigned based on the highest probability. The performance is evaluated using Log-Likelihood, AIC, BIC, and Silhouette Score. Analysis includes comparing GMM with K-Means, understanding soft clustering, and observing overlapping and elliptical cluster shapes. Visualizations like probability distribution plots and contour plots help interpret the results. Overall, GMM is more flexible than K-Means and works better when clusters overlap.# 24ADI003_RAMYA-R_24BAD096_EX7
