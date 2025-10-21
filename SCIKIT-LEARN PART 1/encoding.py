import pandas as pd
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv("sample_data.csv")


df_label = df.copy()

le = LabelEncoder()
df_label['Gender_Encoded'] = le.fit_transform(df_label['Gender'])
df_label['Passed_Encoded'] = le.fit_transform(df_label['Passed'])

print(df_label.head())