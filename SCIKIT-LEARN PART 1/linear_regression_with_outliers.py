from sklearn.linear_model import LinearRegression
import numpy as np

#Data with the outliers
X_out = np.array([[1],[2],[3],[4],[5],[6]])
y_out = np.array([40,45,50,55,60,95])

#Clean data(no Outliers)
X = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([40, 45, 50, 55, 60, 65])

m_out = LinearRegression().fit(X_out, y_out).coef_[0]
b_out = LinearRegression().fit(X_out, y_out).intercept_

m = LinearRegression().fit(X, y).coef_[0]
b = LinearRegression().fit(X, y).intercept_

print(f"With Outliers: slope={m_out:.2f}, intercept={b_out:.2f}")
print(f"Without Outliers: slope={m:.2f}, intercept={b:.2f}")