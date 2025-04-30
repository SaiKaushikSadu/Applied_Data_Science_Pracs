# Time series
from cmath import sqrt

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_csv('./supermarket_sales.csv')

df['Date'] = pd.to_datetime(df['Date'])
df =df.sort_values('Date')

sns.lineplot(x='Date', y='Total', data=df, errorbar=None)
plt.show()

# Important
data = df[['Date','Total']]
data = data.sort_values('Date')
data.set_index('Date', inplace=True)
df1 = data
data = data['Total'].resample('D').mean()
print(data.head())

decompose = seasonal_decompose(data, model='multiplicative')
trend = decompose.trend
seasonal = decompose.seasonal
residual = decompose.resid
decompose.plot()

# ADF
result = adfuller(df['Total'])
print(result[0], result[1])

# ACF ,PACF
# acf = sm.graphics.tsa.plot_acf(data, lags=40)
# plt.show()
# pacf = sm.graphics.tsa.plot_pacf(data, lags=40)
# plt.show()

plot_acf(data, lags=40)
plt.show()
plot_pacf(data, lags=40)
plt.show()

# ARIMA
inputs = df1.index
target = df1['Total'].copy()
X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size=0.3, random_state=42, shuffle=False)

model = ARIMA(y_train, order=(5, 1, 2))
model_fit = model.fit()
predictions = model_fit.forecast(len(y_test))

mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = sqrt(mean_squared_error(y_test, predictions))
print(mse)
print(mae)
print(rmse)

df.head().to_html('supermarket6.html')