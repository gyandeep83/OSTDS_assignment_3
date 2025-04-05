import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load dataset
df = pd.read_csv('/Users/gyandeep/OSTDS_assign_3/Datasets/Customer Purchasing Behaviors.csv')  # Update with your correct path

# Preview
print("Original Data:")
print(df.head())

# Store region separately
regions = df['region']

# Drop non-numeric or identifier columns for PCA
df_numeric = df.drop(columns=['user_id', 'region'])

# Standardize features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_numeric)

# Apply PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

# Explained Variance
explained_variance = pca.explained_variance_ratio_
print(f"\nExplained Variance:\nPC1: {explained_variance[0]:.2f}, PC2: {explained_variance[1]:.2f}")

# Create DataFrame with PCA results + region
pca_df = pd.DataFrame(data=pca_data, columns=['PC1', 'PC2'])
pca_df['Region'] = regions.values

# PCA Plot colored by region
plt.figure(figsize=(9, 6))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Region', palette='Set2', s=80, edgecolor='k', alpha=0.8)
plt.title('Customer Segmentation by Region using PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Region')
plt.grid(True)
plt.tight_layout()
plt.show()
