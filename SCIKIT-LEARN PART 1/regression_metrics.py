import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression


# Example data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([40, 45, 50, 55, 60])

# Train model
model = LinearRegression()
model.fit(X, y)

# Predictions (example)
y_pred = model.predict(X)

# Metrics
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")