# Import necessary libraries
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib.pylab import rcParams
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings
rcParams['figure.figsize'] = 15,6
data = pd.read_csv(r'C:\Users\HP\Desktop\New ML\Arima\AirPassengers.csv')
print(data.head())
print("\nData Types: ")
print(data.dtypes)
# Load the dataset
data = pd.read_csv(r'C:\Users\HP\Desktop\New ML\Arima\AirPassengers.csv')
# Convert Month to datetime format
data['Month'] = pd.to_datetime(data['Month'], format='%Y-%m')
# Set Month as the index
data.set_index('Month', inplace=True)
# Display the data
print(data.head())
print(data.info())
# Convert to timeseries
ts = data['#Passengers']
print(ts.head())
print(ts[datetime(1949,1,1,)])
# Check stationarity
plt.plot(ts)
plt.show()
# Stationarity test
# H0 : Non-stationary
# H1 : Stationary
def test_stationarity(timeseries):
    # Determine rolling statistics
    rolmean = pd.Series(timeseries).rolling(window=12).mean()
    rolstd = pd.Series(timeseries).rolling(window=12).std()
    # Plot rolling statistics:
    orig = plt.plot(timeseries,color='blue',label='Original')
    mwan = plt.plot(rolmean,color='red',label='Rolling Mean')
    std = plt.plot(rolstd,color='black',label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title("Rolling Mean and Standard deviation")
    plt.show(block=False)
    # Perfom Dickey Fuller Test
    print("Results of Dickey Fuller Test: ")
    dftest = adfuller(timeseries,autolag='AIC')
    dfoutput = pd.Series(dftest[0:4],index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput[f'Critical Value ({key})'] = value
    print(dfoutput)
test_stationarity(ts)
# Making ts stationary
ts_log = np.log(ts)
test_stationarity(ts_log)
# Doublelog
ts_doublelog = np.log(ts_log)
test_stationarity(ts_doublelog)

# Smoothing
moving_avg = pd.Series(ts_log).rolling(window=12).mean()
ts_log_moving_avg_diff = (ts_log - moving_avg).dropna()
test_stationarity(ts_log_moving_avg_diff)

expweighted_avg = ts_log.ewm(span=12).mean()
ts_log_ewma_diff = (ts_log - expweighted_avg).dropna()
test_stationarity(ts_log_ewma_diff)

# DECOMPOSITION
decomposition =  seasonal_decompose(ts_log)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
plt.subplot(411)
plt.plot(ts_log,label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend,label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual,label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# GRID SEARCH FOR ARIMA
# Define ranges for p,q and d
ts_values = ts_log.values
X = ts_values
size  = int(len(X)*0.667)
train,test = X[0:size],X[size:len(X)]
p_values = [1,2]
d_values = [0,1]
q_values = [1,2]
best_rmse,best_p,best_d,best_q = np.inf,None,None,None
# Perform grid search
warnings.filterwarnings("ignore")

for p in p_values:
    for d in d_values:
        for q in q_values:
            history = [x for x in train]
            predictions = []
            try:
                for t in range(len(test)):
                    model = ARIMA(history, order=(p, d, q))
                    model_fit = model.fit()
                    yhat = model_fit.forecast()[0]
                    predictions.append(yhat)
                    # Add the actual observation to history
                    history.append(test[t])
                rmse = np.sqrt(mean_squared_error(test, predictions))
                print(f'ARIMA({p},{d},{q}) RMSE: {rmse:.4f}')
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_p, best_d, best_q = p, d, q
            except (ValueError, np.linalg.LinAlgError) as e:
                print(f'ARIMA({p},{d},{q}) failed: {e}')

print('\nBest ARIMA Model:')
print(f'Order: ({best_p},{best_d},{best_q})')
print(f'RMSE: {best_rmse:.4f}')

# FINAL MODEL BUILDING
history2 = [x for x in train]
predictions2 = list()
history2 = [x for x in train]
predictions2 = []
for t in range(len(test)):
    try:
        # Fit ARIMA(2,1,1)
        model2 = ARIMA(history2, order=(2, 1, 1))
        model_fit2 = model2.fit()
        # Forecast the next observation
        yhat2 = model_fit2.forecast()[0]
        predictions2.append(yhat2)
        # Get the actual observation
        obs = test[t]
        # Update history
        history2.append(obs)
        print(f'predicted={yhat2:.3f}, expected={obs:.3f}')
    except (ValueError, np.linalg.LinAlgError) as e:
        print(f'Model failed at step {t}: {e}')
        
rmse = mean_squared_error(test,predictions2) **0.5
print(f'Test RMSE: {rmse:.4f}')
# Plot
pyplot.plot(test,color='blue',label='test')
pyplot.plot(predictions2,color='red',label='preds')
pyplot.show()
# Converting to original format
pyplot.plot(np.exp(test),color='blue',label='test')
pyplot.plot(np.exp(predictions2),color='red',label='preds')
pyplot.show()