import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = {'Age': [23, 45, 30, 60, 25, 40, 35, 50, 29, 41],
        'Spending_Score': [55, 75, 35, 25, 60, 65, 40, 20, 45, 70]}
df = pd.DataFrame(data)

#Create model with 2 clusters
kmeans = KMeans(n_clusters=2, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[['Age', 'Spending_Score']])


print(df.head())
# #Plot
plt.scatter(df['Age'], df['Spending_Score'], c=df['Cluster'], cmap='coolwarm')
plt.xlabel("Age")
plt.ylabel("Spending Score")
plt.title("Customer Segementation")
plt.show()