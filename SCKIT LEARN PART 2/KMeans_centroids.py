import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# Example data
data = {'Age': [23, 45, 30, 60, 25, 40, 35, 50, 29, 41],
        'Spending_Score': [55, 75, 35, 25, 60, 65, 40, 20, 45, 70]}
df = pd.DataFrame(data)

#Create model with 2 clusters
kmeans = KMeans(n_clusters=2, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[['Age', 'Spending_Score']])

#Plot the data Points
plt.figure(figsize=(8,6))
plt.scatter(df['Age'], df['Spending_Score'], c=df['Cluster'], cmap='coolwarm', s=70, alpha=0.7, edgecolors='black')


#Plot cluster centers
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='yellow', marker='X', s=200, edgecolors='black', label='Centroids')

# Labels and title
plt.title("Customer Segmentation using K-Means Clustering", fontsize=14, fontweight='bold')
plt.xlabel("Age", fontsize=12)
plt.ylabel("Spending Score", fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()