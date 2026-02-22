#!/usr/bin/env python
# coding: utf-8

# In[2]:


"""
Customer Segmentation Mini Project
Using Real Dataset
"""

# 1. Import Libraries


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")


# 2. Load Dataset


file_path = "/mnt/data/customer segmentation.csv"

data = pd.read_csv('customer segmentation.csv')

print("Dataset Loaded Successfully!\n")
print("First 5 Rows:\n")
print(data.head())
print("\nDataset Shape:", data.shape)
print("\nColumns:", data.columns)


# 3. Data Cleaning

print("\nMissing Values:\n")
print(data.isnull().sum())

data.drop_duplicates(inplace=True)

# Encode Gender if exists
if "Gender" in data.columns:
    data["Gender"] = data["Gender"].map({"Male": 0, "Female": 1})


# 4. Feature Selection


# Automatically detect common column names
possible_income_cols = [col for col in data.columns if "Income" in col]
possible_spend_cols = [col for col in data.columns if "Spending" in col]

features = ["Age"] + possible_income_cols + possible_spend_cols

if "Gender" in data.columns:
    features = ["Gender"] + features

X = data[features]

print("\nSelected Features:", features)


# 5. Exploratory Data Analysis


print("\nBasic Statistics:\n")
print(X.describe())

plt.figure(figsize=(12,5))
for i, col in enumerate(features):
    plt.subplot(1, len(features), i+1)
    sns.histplot(data[col], kde=True)
    plt.title(col)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
sns.heatmap(X.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


# 6. Scaling

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 7. Find Optimal Clusters


inertia = []
silhouette_scores = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(range(2,11), inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("K")
plt.ylabel("Inertia")

plt.subplot(1,2,2)
plt.plot(range(2,11), silhouette_scores, marker='o')
plt.title("Silhouette Scores")
plt.xlabel("K")
plt.ylabel("Score")

plt.tight_layout()
plt.show()

optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
print("\nOptimal K selected based on highest Silhouette Score:", optimal_k)



# 8. Apply K-Means


kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

data["Cluster"] = clusters



# 9. PCA Visualization


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters)
plt.title("Customer Segmentation (PCA View)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.colorbar(scatter)
plt.show()


# 10. Cluster Profiling


print("\nCluster Mean Values:\n")
cluster_profile = data.groupby("Cluster")[features].mean()
print(cluster_profile)

print("\nCluster Sizes:\n")
print(data["Cluster"].value_counts())


# 11. Business Insights

print("\nBusiness Insights:")

for cluster in cluster_profile.index:
    print(f"\nCluster {cluster}:")
    
    avg_income = cluster_profile.iloc[cluster][possible_income_cols[0]]
    avg_spend = cluster_profile.iloc[cluster][possible_spend_cols[0]]
    
    if avg_income > data[possible_income_cols[0]].mean() and avg_spend < data[possible_spend_cols[0]].mean():
        print("High income but low spending → Strong upselling opportunity.")
    elif avg_spend > data[possible_spend_cols[0]].mean():
        print("High spending customers → Loyalty programs & premium offers.")
    else:
        print("Moderate/Low engagement → Retention & discount campaigns.")


# 12. Save Results


output_file = "segmented_customers.csv"
data.to_csv(output_file, index=False)

print("\nSegmented dataset saved as:", output_file)
print("\nProject Completed Successfully ")


# In[ ]:




