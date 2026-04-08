print("Ramya R-24BAD096")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
df = pd.read_csv(r"C:\Users\Ramya.R\Downloads\archive (8).zip")
print(df.head())
df.dropna(inplace=True)
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
inertia = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
plt.figure(figsize=(8,5))
plt.plot(K_range, inertia, marker='o', label='Inertia (WCSS)')
plt.title("Elbow Method (K vs Inertia)")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.legend()
plt.show()
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
df['Cluster'] = clusters
print("\nCluster Means:\n")
print(df.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)', 'Age']].mean())
print("\nClustered Data:\n", df.head())
plt.figure(figsize=(8,6))
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=clusters)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=200, marker='X',label='Centroids')
plt.title("Customer Segments (K-Means)")
plt.xlabel("Annual Income (scaled)")
plt.ylabel("Spending Score (scaled)")
plt.legend()
plt.show()
print("\nEvaluation Metrics:")
print("Inertia:", kmeans.inertia_)
sil_score = silhouette_score(X_scaled, clusters)
print("Silhouette Score:", sil_score)
print("\nSilhouette Scores for different K:\n")
for k in range(2, 10):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f"K={k} -> Silhouette Score = {score:.4f}")
