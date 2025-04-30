import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./diamonds.csv')
df.drop(columns=['Unnamed: 0'], inplace=True)

sns.countplot(data=df, x='cut')
plt.show()

sns.barplot(x='color', y='price', data=df)
plt.show()

sns.heatmap(df[['x','y','z','carat']].corr(), annot=True)
plt.show()

sns.scatterplot(x='carat', y='price', data=df)
plt.show()

sns.pairplot(df[['x','y','z','carat']])
plt.tight_layout()
plt.show()

sns.boxplot(df[['x','y','z','carat']])
plt.show()

df.head(10).to_html('diamonds.html')