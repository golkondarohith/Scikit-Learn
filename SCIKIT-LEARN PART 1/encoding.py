import pandas as pd
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv("sample_data.csv")


df_label = df.copy()

le = LabelEncoder()
df_label['Gender_Encoded'] = le.fit_transform(df_label['Gender'])
df_label['Passed_Encoded'] = le.fit_transform(df_label['Passed'])


df_encoded = pd.get_dummies(df_label, columns=['City'])
print(df_encoded.head())