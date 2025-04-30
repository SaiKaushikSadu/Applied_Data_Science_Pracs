import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('./mushrooms.csv')

for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

df.head(10).to_html('mushroom.html')