import pandas as pd

data = pd.read_csv("netflix_titles.csv")
df = pd.DataFrame(data)
# print(df.head())

print(df.shape)

print(df.isnull().sum())

df = df.dropna()
df.fillna(df.mean(numeric_only=True), inplace=True)
df = df.drop_duplicates()
print(df.shape)