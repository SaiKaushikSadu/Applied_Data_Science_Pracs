# Unsupervised Learning - Regression and Classification
import math

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score, mean_squared_error, classification_report, confusion_matrix, roc_curve, roc_auc_score

df = pd.read_csv('./supermarket_sales.csv')

# Regression - LR, Decision Tree, Random Forest
X_train, X_test, y_train, y_test = train_test_split(df[['Unit price','Quantity', 'gross income']], df['Total'], test_size=0.2)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

rmse = root_mean_squared_error(y_test, y_pred)
print(rmse)
mse = mean_squared_error(y_test, y_pred)
print(mse)
mae = mean_absolute_error(y_test, y_pred)
print(mae)
r2 = r2_score(y_test, y_pred)
print(r2)

# Classification - Logistic Reg, Decision Tree, Random Forest, KNN
le = LabelEncoder()
df['Customer type'] = le.fit_transform(df['Customer type'])

X_train, X_test, y_train, y_test = train_test_split(df[['Unit price','Quantity', 'gross income']], df['Customer type'], test_size=0.2)
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

print(classification_report(y_test, y_pred))

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(tn)
print(fp)
print(fn)
print(tp)

accuracy = (tn + tp) / (tn + fp + tp + fn)
print(accuracy)

precision = tp / (tp + fp)
print(precision)

sns = tp/(tp + fn)
spc = tn/(tn + fp)
print(sns)
print(spc)

print("GM", math.sqrt(sns*spc))

f1 = (2*sns*precision) / (precision + sns)
print(f1)

fpr = 1-spc
fnr = 1 -sns
power = 1 - fnr
print("False positive Rate {}".format(fpr))
print("false negative Rate {}".format(fnr))
print("Power {}".format(power))

false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_test, y_pred)
print('roc_auc_score for DecisionTree: ', roc_auc_score(y_test, y_pred))

plt.plot(false_positive_rate1, true_positive_rate1)
plt.show()


X_train, X_test, y_train, y_test = train_test_split(df[['Unit price', 'Quantity', 'gross income']], df['Total'], test_size=0.2)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(mse)
print(mae)
print(r2)