import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

df = pd.read_csv('./supermarket_sales.csv')

print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())

print(df.median(numeric_only=True))
print(df.mean(numeric_only=True))

print(df['City'].mode())

plt.scatter(df['Quantity'], df['Total'], c ="blue")
plt.show()

sns.scatterplot(data=df, x= df['Tax 5%'], y= df['Unit price'])
plt.show()

plt.boxplot(df['Tax 5%'])
plt.show()

print(stats.trim_mean(df['Tax 5%'], 0.1))

print(df.sum(numeric_only=True))
print(df.var(numeric_only=True))
print(df.corr(numeric_only=True))
print(df.sem(numeric_only=True))

sos = 0
for value in df['Tax 5%']:
    sos+= value*value

print(sos)

print(df['Tax 5%'].skew())
print(df['Tax 5%'].kurtosis())

sns.distplot(df['Total'])
plt.show()

df.head(10).to_html('supermarket.html')
