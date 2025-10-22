from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd


#DataFrame
data = {
    'StudyHours': [1, 2, 3, 4, 5, 6, 7],
    'TestScore': [50, 55, 60, 65, 70, 75, 80]
}

df = pd.DataFrame(data)

#Standard Scaler
s_scaler = StandardScaler()
standard_scaler = s_scaler.fit_transform(df)
print("Standard Scaler Output:\n", standard_scaler)

#MinMax Scaler
m_scaler = MinMaxScaler()
minmax_scaler = m_scaler.fit_transform(df)
print(pd.DataFrame(minmax_scaler, columns=['StudyHours', 'TestScore']))


#Test data Splitting
X = df[['StudyHours']]
y = df[['TestScore']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=42)

