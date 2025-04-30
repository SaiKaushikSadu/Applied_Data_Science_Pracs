import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

df = pd.read_csv('./Churn_Modelling.csv')

sns.scatterplot(data = df , x ='CreditScore', y = 'Age', hue = 'Exited')
plt.show()

print(df['Exited'].value_counts())

for col in df.columns:
    if df[col].dtype == 'O':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

X_train, X_test, y_train, y_test = train_test_split(df.drop('Exited', axis = 1), df['Exited'], test_size = 0.2, random_state = 101)
dt = DecisionTreeClassifier()

# Predict - 1
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
print(classification_report(y_test, y_pred))


smote = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=101)
X_oversample, y_oversample = smote.fit_resample(X_train, y_train)

# Predict - 2
dt.fit(X_oversample, y_oversample)
y_pred_smote = dt.predict(X_test)
print(classification_report(y_test, y_pred_smote))


df_smote = pd.DataFrame(X_oversample, columns=df.columns)
df_smote['Exited'] = y_oversample
sns.scatterplot(data=df_smote, x='CreditScore', y='Age', hue='Exited')
plt.show()