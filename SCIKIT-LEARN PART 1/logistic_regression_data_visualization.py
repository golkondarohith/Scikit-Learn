import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Training data
X = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([0, 0, 0, 1, 1, 1])

# Train the logistic regression model
model = LogisticRegression()
model.fit(X, y)

#Create a range of study hours for smoother study
X_test  = np.linspace(0, 7, 100).reshape(-1, 1)

#Get predictied probabilities of passing (class 1)
y_prob = model.predict_proba(X_test)[:, 1]

#Plot
plt.plot(X_test, y_prob, label="Probability of Passing", linewidth = 2)
plt.scatter(X, y, color='red', label="Actual Data (0=Fail, 1=Pass)")
plt.xlabel("Hours Studied")
plt.ylabel("Probability of Passing")
plt.title("Logistic Regression: Study Hours vs. Pass Probability")
plt.legend()
plt.grid(True)
plt.show()