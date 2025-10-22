from sklearn.linear_model import LinearRegression

X = [[1], [2], [3], [4], [5]]
y = [40, 45, 50, 55, 60]

model = LinearRegression()

model.fit(X, y)

hours = float(input("Enter the number of hours studied: "))

predicted_marks = model.predict([[hours]])

print(f"Predicted marks for the hours {hours} are {predicted_marks}")
