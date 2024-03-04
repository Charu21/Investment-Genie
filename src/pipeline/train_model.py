import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import math

from sklearn.metrics import mean_squared_error
from dataclasses import dataclass
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.ar_model import AutoReg
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler

from src.exceptions import CustomException
from src.logger import logging

from src.utils import plot_graph

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("model","model.pkl")
    
class ModelConfig:
    def __init__(self, model_name, duration):
        self.model_trainer_config=ModelTrainerConfig()
        self.model_name = model_name
        duration_in_int = 0
        if duration == "Daily":
            duration_in_int = 3
        elif duration == "Weekly":
            duration_in_int = 7
        else: 
            duration_in_int = 30
        self.duration = duration_in_int

    def predict(self,train,test):

        try:
            logging.info("Split training and test input data")
            train_index = train.shape[0]
            test_index = test.shape[0]
            if self.model_name == 'lstm':
                logging.info(f' LSTM has been called!')
                if self.duration == 7:
                    result_lstm = create_lstm(train_data, test_data, 7)
                    testPredictPlot = prediction_report(result_lstm, 7, "Weekly")                    
                elif self.duration == 30:
                    
            elif self.model_name == 'AR':
                ar_model = apply_autoreg(train, test)
                return model_predict("AR", ar_model, train, test)
            elif self.model_name == 'ARIMA':
                arima = apply_arima(train)
                return model_predict("ARIMA", arima, train, test)
            elif self.model_name == 'SARIMAX':
                sarima = apply_sarimax(train, test)
                return model_predict("SARIMA", sarima, train, test)
            elif self.model_name == 'SES':
                ses = apply_ses(train)
                return model_predict("SES", ses, train, test)
            elif self.model_name == 'DES':
                des = apply_des(train)
                return model_predict("DES", des, train, test)
            elif self.model_name == 'TES':
                tes = apply_tes(train, self.duration)
                return model_predict("TES", tes, train, test)
            elif self.model_name == 'Prophet':
                return apply_prophet_and_predict(train, test, self.duration)
        
        except Exception as e:
            raise CustomException(e,sys)
        
def apply_autoreg(train_data, test_data):
    df = pd.concat([train_data, test_data])

    plot_acf(df['Close'], lags=20)
    plt.title('Autocorrelation Function (ACF)')
    plt.show()

    plot_pacf(df['Close'], lags=20)
    plt.title('Partial Autocorrelation Function (PACF)')
    plt.show()

    # Fit models with different lag orders
    for p in range(1, 20):
        model = AutoReg(df['Close'], lags=p)
        model_fit = model.fit()
        print(f'Lag Order {p}: AIC={model_fit.aic}, BIC={model_fit.bic}')

    # Taking lag order as 10 as proper AIC and BIC value
    final_ar_model = AutoReg(train_data['Close'], lags=10)
    ar_model_fit = final_ar_model.fit()

    print(ar_model_fit.summary())
    return ar_model_fit 
    
def apply_arima(train_data):
    arima_model = ARIMA(train_data['Close'], order = (1,1,1))
    model_fit = arima_model.fit()
    print(model_fit.summary())
    model_fit.plot_diagnostics()
    # plt.show()
    
    return model_fit 

def apply_sarimax(train_data, test_data):
    df = pd.concat([train_data, test_data])
    decomp_results = seasonal_decompose(df['Close'], period=25)
    decomp_results.plot()
    plt.show()

    # Use auto_arima to find the best SARIMA model
    auto_model = auto_arima(train_data['Close'], seasonal=True, m=10, trace=True, suppress_warnings=True)
    print(auto_model.summary())
    
    sarima = SARIMAX(train_data['Close'], order=(0,1,1), seasonal_order=(0,0,1,10))
    sarima_fit = sarima.fit()
    print(sarima_fit.summary())

    # Now plotting common diagnostics
    sarima_fit.plot_diagnostics()
    plt.show()

    return sarima_fit 

def apply_ses(train_data):
    exponential_smoothing_model = ExponentialSmoothing(train_data['Close'], seasonal=None, trend=None)
    es_model_fit = exponential_smoothing_model.fit(smoothing_level=0.2)
    return es_model_fit 

def apply_des(train_data):
    exponential_smoothing_model = ExponentialSmoothing(train_data['Close'], seasonal=None, trend='add')
    es_model_fit = exponential_smoothing_model.fit(smoothing_level=0.2)
    return es_model_fit 

def apply_tes(train_data, period):
    exponential_smoothing_model = ExponentialSmoothing(train_data['Close'], trend='add', seasonal='add', seasonal_periods=period)
    es_model_fit = exponential_smoothing_model.fit(smoothing_level=0.2)
    return es_model_fit 

def apply_prophet_and_predict(train, test, period):
    # Make input readable by FB Prophet
    # ds: date and y: price/close
    logging.info("Started applying Prophet model!")
    train_prophet = pd.DataFrame({'ds': train.index, 'y': train.Close})
    test_prophet = pd.DataFrame({'ds': test.index, 'y':test.Close})    
    logging.info(train_prophet.head())
    prophet_model = Prophet()
    prophet_model.fit(train_prophet)
    # Make future dataframe for predictions
    future = prophet_model.make_future_dataframe(periods=len(test))
    forecast = prophet_model.predict(future)

    test_final = prophet_model.predict(test_prophet)
    
    # Plotting graph here
    f, ax = plt.subplots(figsize=(14,5))
    train_prophet.plot(kind='line',x='ds', y='y', label='Actual', ax=ax)
    test_prophet.plot(kind='line',x='ds',y='y',label='Test', ax=ax)
    test_final.plot(kind='line',x='ds',y='yhat',label='Forecast', ax=ax)
    plt.title('Forecast vs Actuals')
    plt.show()

    mse = mean_squared_error(test_prophet['y'], test_final[-len(test_prophet):]['yhat'])
    print(f"Mean Squared Error: {mse}")
    logging.info(f'Value of mse = {mse}')
    return mse

def model_predict(model_name: str, model_fit: any, train: pd.DataFrame, test: pd.DataFrame) -> float:
    train_index = train.shape[0]
    test_index = test.shape[0]
    predictions = model_fit.predict(start=train_index, end=train_index + test_index -1)
    predictions_df = plot_graph(predictions, train, test, model_name)
    mse = mean_squared_error(test['Close'], predictions_df['Close'])
    print(f'Mean Squared Error (MSE) for {model_name}: {mse}')
    return mse

# preparing data for LSTM
def prepare_data(timeseries_data, n_features):
	X, y =[],[]
	for i in range(len(timeseries_data)):
		# find the end of this pattern
		end_ix = i + n_features
		# check if we are beyond the sequence
		if end_ix > len(timeseries_data)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = timeseries_data[i:end_ix], timeseries_data[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten

def create_lstm(train_data, test_data, n_steps) -> Result:
    n_features = 1
    X, y = prepare_data(train_data['Close'], n_steps)

    # reshape from [samples, timesteps] into [samples, timesteps, features]
    X = X.reshape((X.shape[0], X.shape[1], n_features))

    X_test, y_test = prepare_data(test_data['Close'], n_steps)

    # define model
    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error',  run_eagerly=True)
    # fit model
    history_lstm = model.fit(X, y, epochs=200, validation_data=(X_test, y_test), batch_size=64, verbose=1)

    res = Result(model, X, y, X_test, y_test, history_lstm)
    return res


