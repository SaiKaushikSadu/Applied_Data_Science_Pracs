import pandas as pd

df = pd.read_csv('./travel-times.csv')

df.drop(columns=['Comments'] ,inplace=True)
df['FuelEconomy'] = df['FuelEconomy'].replace('-', pd.NA)
df.dropna(subset=['FuelEconomy'], inplace=True)

print(df.isnull().sum())
df.head(10).to_html('travel.html')