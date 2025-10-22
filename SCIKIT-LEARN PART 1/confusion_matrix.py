import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Example: True vs predicted labels
y_true = [0, 0, 1, 1, 1, 0]
y_pred = [0, 0, 1, 0, 1, 1]

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Define class labels
labels = ['Apple (0)', 'Orange (1)']

# Plot confusion matrix as heatmap
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm',
            xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.title('Confusion Matrix Visualization')
plt.show()
