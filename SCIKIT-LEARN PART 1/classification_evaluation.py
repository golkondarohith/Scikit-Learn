from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier


#predicting fruit bases on the weight and size
X = [
    [180, 7], 
    [200, 7.5], 
    [250, 8], 
    [300, 8.5],
    [330, 9],
    [360, 9.5]
]

# 0 = Apple, 1 = Orange
y = [0, 0, 0, 1, 1, 1,]

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)


#Suppose these are the true vs predicted lables for test data:
y_true = [0, 0, 1, 1, 1, 0]
y_pred = [0, 0, 1, 0, 1, 1]

#Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

print("Confusion Matrix:\n", cm)
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")