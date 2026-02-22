#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Customer Segmentation Mini Project
Author: mohammed shamsad
Description: End-to-end customer segmentation using K-Means clustering
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



# 2. Generate or Load Dataset


try:
    data = pd.read_csv("customer_data.csv")
    print("Dataset loaded successfully.\n")
except:
    print("Dataset not found. Generating synthetic dataset...\n")
    
    np.random.seed(42)
    n = 300
    
    data = pd.DataFrame({
        "CustomerID": range(1, n+1),
        "Age": np.random.randint(18, 70, n),
        "Annual Income": np.random.randint(15, 150, n),
        "Spending Score": np.random.randint(1, 100, n),
        "Recency": np.random.randint(1, 365, n),
        "Frequency": np.random.randint(1, 50, n),
        "Monetary Value": np.random.randint(100, 10000, n)
    })

print("First 5 Rows:")
print(data.head())



# 3. Data Cleaning


print("\nChecking Missing Values:")
print(data.isnull().sum())

data.drop_duplicates(inplace=True)


# 4. Feature Selection


features = ["Age", "Annual Income", "Spending Score"]
X = data[features]



# 5. Exploratory Data Analysis


print("\nBasic Statistics:")
print(X.describe())

plt.figure(figsize=(12,4))
for i, col in enumerate(features):
    plt.subplot(1,3,i+1)
    sns.histplot(data[col], kde=True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(5,4))
sns.heatmap(X.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


# 6. Scaling

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 7. Determine Optimal Clusters


inertia = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.show()


# Silhouette scores
print("\nSilhouette Scores:")
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f"K={k}, Score={score:.3f}")



# 8. Apply K-Means (Assume k=4)

optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

data["Cluster"] = clusters


# 9. Visualize Clusters (PCA)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters)
plt.title("Customer Segmentation (PCA View)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()


# 10. Cluster Profiling


print("\nCluster Mean Values:")
cluster_profile = data.groupby("Cluster")[features].mean()
print(cluster_profile)

print("\nCluster Sizes:")
print(data["Cluster"].value_counts())


# 11. Business Insights Generator

print("\nBusiness Insights:")

for cluster in cluster_profile.index:
    age = cluster_profile.loc[cluster, "Age"]
    income = cluster_profile.loc[cluster, "Annual Income"]
    spend = cluster_profile.loc[cluster, "Spending Score"]
    
    print(f"\nCluster {cluster}:")
    
    if income > 100 and spend < 50:
        print("High income but low spending → Upselling opportunity.")
    elif spend > 70:
        print("High spending customers → Loyalty & premium offers.")
    elif spend < 30:
        print("Low engagement customers → Discount campaigns.")
    else:
        print("Moderate customers → Retention strategy.")



# 12. Save Results

data.to_csv("segmented_customers.csv", index=False)
print("\nSegmented dataset saved as 'segmented_customers.csv'")

print("\nProject Completed Successfully")


# In[ ]:




