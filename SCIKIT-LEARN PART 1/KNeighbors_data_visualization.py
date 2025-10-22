import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Training data
X = np.array([
    [180, 7], 
    [200, 7.5], 
    [250, 8], 
    [300, 8.5],
    [330, 9],
    [360, 9.5]
])
y = np.array([0, 0, 0, 1, 1, 1])  # 0 = Apple, 1 = Orange

# Train the model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# Create a grid for plotting the decision boundary
x_min, x_max = X[:, 0].min() - 20, X[:, 0].max() + 20
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

# Predict on the grid
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary and training points
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
plt.xlabel("Weight (grams)")
plt.ylabel("Size (cm)")
plt.title("KNN Decision Boundary for Fruit Classification")
plt.show()
