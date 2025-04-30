import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

df = pd.read_csv('./loan_data_set.csv', encoding='ansi')

# Remove rows of missing values
# df = df.dropna(subset=['CoapplicantIncome'])

# Replace with mean
# df['LoanAmount'].fillna(df['LoanAmount'].mean(),inplace=True)

# Replace with median
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median(),inplace=True)

# Replace with mode
df['Gender'].fillna(df['Gender'].mode()[0],inplace=True) # [0] is important

# Arbitrary value
df['Self_Employed'].fillna('Yes',inplace=True)

# Random sampling
def random_sampling(series):
    random_val  = series.dropna().sample(series.isnull().sum(), replace=True, random_state=42)
    series[series.isnull()] = random_val.values
    return series
df['Credit_History'] = random_sampling(df['Credit_History'])

# Freq category
# df['Gender'].fillna(df['Gender'].mode()[0],inplace=True)

# Add new 'Missing' category
# df['Gender'].fillna('Missing', inplace=True)

# LR
df1 = df[["CoapplicantIncome", "LoanAmount"]].copy()

# Split data into training and testing
train = df1[df1['LoanAmount'].isnull() == False]
test = df1[df1['LoanAmount'].isnull() == True]

X_train = train[['CoapplicantIncome']]
y_train = train['LoanAmount']

lr = LinearRegression()
lr.fit(X_train, y_train)

predicted_values = lr.predict(test[['CoapplicantIncome']])
df.loc[df['LoanAmount'].isnull(), 'LoanAmount'] = predicted_values


print(df.isnull().sum())
df.head(10).to_html('loan2.html')
