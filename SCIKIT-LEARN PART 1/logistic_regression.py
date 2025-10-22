from sklearn.linear_model import LogisticRegression


X = [[1], [2], [3], [4], [5], [6]]
y = [0, 0, 0, 1, 1, 1]


model = LogisticRegression()

model.fit(X, y)

hours = float(input("Enter the number of hours studied: "))

result = model.predict([[hours]])[0]

probabilities = model.predict_proba([[hours]])[0]

if result == 0:
    print(f"You are likely to FAIL. Probability of passing: {probabilities[1]*100:.2f}%")
else:
    print(f"You are likely to PASS. Probability of passing: {probabilities[1]*100:.2f}%")
