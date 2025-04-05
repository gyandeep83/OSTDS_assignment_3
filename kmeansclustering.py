import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Step 1: Load the Dataset
df = pd.read_csv('/Users/gyandeep/OSTDS_assign_3/Datasets/Credit Card Customer Data.csv')
print("Initial Data Preview:")
print(df.head())

# Step 2: Drop unnecessary columns
df_clean = df.drop(columns=['Sl_No', 'Customer Key'])

# Step 3: Feature Scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_clean)

# Step 4: Elbow Method to determine optimal number of clusters
inertia = []
k_range = range(1, 11)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(scaled_data)
    inertia.append(km.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Inertia')
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 5: Apply KMeans with optimal number of clusters (let’s say K=3 for now)
k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(scaled_data)
df['Cluster'] = clusters

# Step 6: Analyze the clusters
cluster_summary = df.groupby('Cluster').mean()
print("\nCluster Summary:")
print(cluster_summary)

# Step 7a: Pairplot of key features by cluster
sns.pairplot(df, hue='Cluster', vars=[
    'Avg_Credit_Limit', 'Total_Credit_Cards',
    'Total_visits_online', 'Total_visits_bank', 'Total_calls_made'
], palette='Set2')
plt.suptitle("Customer Clusters Based on Credit Card Behavior (Pairplot)", y=1.02)
plt.tight_layout()
plt.show()

# Step 7b: PCA-based 2D Scatter Plot for Clusters
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_data)

df['PCA1'] = pca_components[:, 0]
df['PCA2'] = pca_components[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='Set2', s=60, edgecolor='k')

# Plot centroids
centroids = pca.transform(kmeans.cluster_centers_)
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=100, alpha=0.8, marker='X', label='Centroids')

plt.title('Clusters Visualization with PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.tight_layout()
plt.show()
