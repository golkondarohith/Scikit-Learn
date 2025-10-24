import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd


data = {'Age': [23, 45, 30, 60, 25, 40, 35, 50, 29, 41],
        'Spending_Score': [55, 75, 35, 25, 60, 65, 40, 20, 45, 70]}
df = pd.DataFrame(data)

inertia = []
K = range(1, 10)

for k in K:
    model = KMeans(n_clusters=k, random_state=0)
    model.fit(df[['Age', 'Spending_Score']])
    inertia.append(model.inertia_)

plt.plot(K, inertia, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia (Within-cluster Sum of Squares)')
plt.title('Elbow Method for Optimal K')
plt.grid(True)
plt.show()
