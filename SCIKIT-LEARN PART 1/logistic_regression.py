from sklearn.linear_model import LogisticRegression


X = [[1], [2], [3], [4], [5], [6]]
y = [0, 0, 0, 1, 1, 1]


model = LogisticRegression()

model.fit(X, y)

hours = float(input("Enter the number of hours studied: "))

result = model.predict([[hours]])[0]

if result == 0:
    print(f"Predicted result for the hours {hours}, you are likely to FAIL")
else: 
    print(f"Predicted result for the hours {hours}, you are likely to PASS")
