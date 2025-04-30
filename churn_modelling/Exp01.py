import pandas as pd
from scipy import stats
from scipy.stats import norm, poisson
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.stats.weightstats import ztest

df = pd.read_csv('./Churn_Modelling.csv', encoding='ansi')

print(df.shape)

print(df.isnull().sum())

print(df.describe())

print(df.info())

print(df.median(numeric_only=True))

print(df['Geography'].mode())
print(df['Age'].mode())
print(df['Gender'].mode())

print(df['EstimatedSalary'].sum())
print(df['Balance'].sum())

print(df['Geography'].value_counts())
print(df['Gender'].value_counts())

print(df.var(numeric_only=True))
print(df.corr(numeric_only=True))
# Std error of mean
print(df.sem(numeric_only=True))
print(df.skew(numeric_only=True))
print(df.kurtosis(numeric_only=True))

# Inter Quartile Range
q1 = df['EstimatedSalary'].quantile(0.25)
q3 = df['EstimatedSalary'].quantile(0.75)
print("IQR ->",q3-q1)

# Coefficient of variation
mean = df['EstimatedSalary'].mean()
sd = df['EstimatedSalary'].std()
cv = (sd / mean) * 100
print("CV -> ",cv)

# Sum of squares
sos = 0
for val in df['EstimatedSalary']:
    sos += val*val
print("Sum of Square -> ",sos)

# Trimmed Mean
tm = stats.trim_mean(df['EstimatedSalary'], proportiontocut=0.1)
print("Trimmed Mean -> ",tm)

# Percentage
print(df['Tenure'].value_counts(normalize=True) * 100)

# Box plot - do separate for each one of them
num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
plt.figure(figsize=(10,5))
sns.boxplot(df[num_cols])
plt.show()

# Scatter plot
plt.figure(figsize=(10,5))
sns.scatterplot(x='CreditScore', y='Age', hue='Tenure', data=df)
plt.show()

# Corr matrix
corr_cols = ['CreditScore', 'Age', 'Tenure', 'Exited']
plt.figure(figsize=(10,5))
sns.heatmap(df[corr_cols].corr(), annot=True)
plt.show()

# Normal distribution
sns.distplot(df['Age'])
plt.show()
# sns.histplot(df['Age'], kde=True, stat="density")
# plt.show()

# Sampling errors
pop_mean = df['Balance'].mean()
sample = df['Balance'].sample(100, random_state=42)
sample_mean = sample.mean()
sample_std = sample.std()
n = len(sample)

confidence_interval = stats.norm.interval(confidence=0.95, loc=sample_mean, scale=sample_std/np.sqrt(n))
print("Confidence Level -> ",confidence_interval)
print("Sample error -> ", abs(pop_mean - sample_mean))

def check(value, alpha=0.05):
    if value < alpha:
        print("Reject H0")
    else :
        print("Reject H1")

# Hypothesis Testing - Z test
exited = df[df['Exited'] == 1]['CreditScore']
z_stat , p_val = ztest(exited, df['CreditScore'])
print("Z-statistic -> ",z_stat)
print("P Val -> ",p_val)
check(p_val)

# Hypothesis Testing - T test - 2 groups
exited_true = df[df['Exited'] == 1]['Age']
exited_false = df[df['Exited'] == 0]['Age']
t_cal, p_val = stats.ttest_ind(exited_true, exited_false)
print("T-Cal -> ",t_cal)
print("P-value -> ",p_val)
check(p_val)

# Hypothesis Testing - ANOVA - 3 Groups
france = df[df['Geography'] == 'France']['Age']
spain = df[df['Geography'] == 'Spain']['Age']
germany = df[df['Geography'] == 'Germany']['Age']

anova_cal , p_val = stats.f_oneway(france, spain, germany)
print("T-Cal -> ",anova_cal)
print("P-value -> ",p_val)
check(p_val)



df.head(20).to_html('churn1.html')