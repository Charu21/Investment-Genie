import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import math
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

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

from src.pipeline.result import Result
from src.exceptions import CustomException
from src.logger import logging

from src.utils import plot_graph, prediction_report, load_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("model","model.pkl")
    
class ModelConfig:
    def __init__(self, model_name, duration):
        self.model_trainer_config=ModelTrainerConfig()
        self.model_name = model_name
        self.duration = duration

    def evaluate(self,train,test):

        try:
            logging.info("Split training and test input data")
            train_index = train.shape[0]
            test_index = test.shape[0]
            logging.info(f"Size of data frame ={train_index} vs {test_index}")
            if self.model_name == 'LSTM':
                logging.info(f' LSTM has been called with value = {self.duration}')
                if self.duration == '7':
                    model_path = 'model\lstm_weekly_model.pkl'
                    model = load_object(file_path = model_path)
                    X, y, X_test, y_test = lstm_model_data_transform(train, test, 7)
                    _, ans = prediction_report(model, train, test, X, y, X_test, y_test, 7, "Weekly")
                    return ans
                elif self.duration == '30':
                    model_path = 'model\lstm_monthly_model.pkl'
                    X, y, X_test, y_test = lstm_model_data_transform(train, test, 30)
                    model = load_object(file_path = model_path)
                    _, ans = prediction_report(model, train, test, X, y, X_test, y_test, 30, "Monthly")
                    return ans
                elif self.duration == '1':
                    model_path = 'model\lstm_daily_model.pkl'
                    model = load_object(file_path = model_path)
                    X, y, X_test, y_test = lstm_model_data_transform(train, test, 1)
                    _, ans = prediction_report(model, train, test, X, y, X_test, y_test, 1, "Daily")
                    return ans
                else: 
                    return "No such option exists"
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
            else :
                return "Not Yet!"
        except Exception as e:
            raise CustomException(e,sys)
        

    def predict(self,train,test):

        try:
            logging.info("Computing Bollinger Bands")
            train_index = train.shape[0]
            test_index = test.shape[0]
            data = pd.concat([train, test])
            logging.info(f"Size of data frame ={train_index} vs {test_index}")
            if self.model_name == 'LSTM':
                logging.info(f' LSTM has been called with value = {self.duration}')
                if self.duration == '7':
                    model_path = 'model\lstm_weekly_model.pkl'
                    model = load_object(file_path = model_path)
                    X, y, X_test, y_test = lstm_model_data_transform(train, test, 7)
                    testPredictPlot, _ = prediction_report(model, train, test, X, y, X_test, y_test, 7, "Weekly")
                    weekly_data = data.iloc[0:]
                    weekly_data['lstm_weekly'] = testPredictPlot.flatten()
                    weekly_ans, plt = add_signal(weekly_data, False, 20, 2, 'lstm_weekly')
                    fig_name = "lstm_" + self.duration + "_days.png"
                    file_path = "C:/Users/CHARU/Pictures/ML/MLProjects/static/image/"
                    final_path = os.path.join(file_path, fig_name) 
                    if os.path.isfile(final_path):
                        os.remove(final_path)  
                    plt.savefig(final_path)
                    weekly_return = money_made(weekly_ans, "Weekly")
                    return weekly_return, fig_name
                    
                elif self.duration == '30':
                    model_path = 'model\lstm_monthly_model.pkl'
                    model = load_object(file_path = model_path)
                    X, y, X_test, y_test = lstm_model_data_transform(train, test, 30)
                    testPredictPlot, _ = prediction_report(model, train, test, X, y, X_test, y_test, 30, "Monthly")
                    monthly_data = data.iloc[1:]
                    monthly_data['lstm_monthly'] = testPredictPlot.flatten()
                    monthly_ans, plt = add_signal(monthly_data, False, 20, 2, 'lstm_monthly')
                    fig_name = "lstm_" + self.duration + "_days.png"
                    file_path = "C:/Users/CHARU/Pictures/ML/MLProjects/static/image/"
                    final_path = os.path.join(file_path, fig_name) 
                    if os.path.isfile(final_path):
                        os.remove(final_path)  
                    plt.savefig(final_path)
                    monthly_return = money_made(monthly_ans, "Monthly")
                    return monthly_return, fig_name
                
                elif self.duration == '1':
                    model_path = 'model\lstm_daily_model.pkl'
                    model = load_object(file_path = model_path)
                    X, y, X_test, y_test = lstm_model_data_transform(train, test, 1)
                    testPredictPlot, _ = prediction_report(model, train, test, X, y, X_test, y_test, 1, "Daily")
                    daily_data = data.iloc[1:]
                    daily_data['lstm_daily'] = testPredictPlot.flatten()
                    daily_ans, plt = add_signal(daily_data, False, 20, 2, 'lstm_daily')
                    fig_name = "lstm_" + self.duration + "_days.png"
                    file_path = "C:/Users/CHARU/Pictures/ML/MLProjects/static/image/"
                    final_path = os.path.join(file_path, fig_name) 
                    if os.path.isfile(final_path):
                        os.remove(final_path)  
                    plt.savefig(final_path)
                    daily_return = money_made(daily_ans, "Daily")
                    return daily_return, fig_name
                
                else: 
                    return "No such option exists"
            else :
                return "Not Yet!"
        except Exception as e:
            raise CustomException(e,sys)
        
def apply_autoreg(train_data, test_data):
    df = pd.concat([train_data, test_data])

    plot_acf(df['Close'], lags=20)
    plt.title('Autocorrelation Function (ACF)')
    #plt.show()

    plot_pacf(df['Close'], lags=20)
    plt.title('Partial Autocorrelation Function (PACF)')
    #plt.show()

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
    # #plt.show()
    
    return model_fit 

def apply_sarimax(train_data, test_data):
    df = pd.concat([train_data, test_data])
    decomp_results = seasonal_decompose(df['Close'], period=25)
    decomp_results.plot()
    #plt.show()

    # Use auto_arima to find the best SARIMA model
    auto_model = auto_arima(train_data['Close'], seasonal=True, m=10, trace=True, suppress_warnings=True)
    print(auto_model.summary())
    
    sarima = SARIMAX(train_data['Close'], order=(0,1,1), seasonal_order=(0,0,1,10))
    sarima_fit = sarima.fit()
    print(sarima_fit.summary())

    # Now plotting common diagnostics
    sarima_fit.plot_diagnostics()
    #plt.show()

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
    #plt.show()

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

def lstm_model_data_transform(train_data: pd.DataFrame, test_data:pd.DataFrame, n_steps):

    n_features = 1
    X, y = prepare_data(train_data['Close'], n_steps)

    # reshape from [samples, timesteps] into [samples, timesteps, features]
    X = X.reshape((X.shape[0], X.shape[1], n_features))

    X_test, y_test = prepare_data(test_data['Close'], n_steps)

    return X, y, X_test, y_test

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


def plot_bollinger_bands(df: pd.DataFrame, buy_signal: list, sell_signal: list, hold_signal: list, model_name: str):
    fig, ax = plt.subplots(figsize=(16, 5))
    df['predictions'].plot(label='Predicted prices', linewidth=1.5)
    df['BU'].plot(label='Upper BB', linestyle='--', linewidth=1.5)
    df['B_MA'].plot(label='Middle BB', linestyle='--', linewidth=1.5)
    df['BL'].plot(label='Lower BB', linestyle='--', linewidth=1.5)
    
    plt.scatter(df.index, buy_signal, marker='^', color='tab:green', label='Buy', s=100)
    plt.scatter(df.index, np.absolute(sell_signal), marker='v', color='tab:red', label='Sell', s=100)
    plt.scatter(df.index, hold_signal, marker='*', color='tab:blue', label='Hold', s=100)

    plt.title(f'Bollinger Bands Strategy for {model_name} - Trading Signals', fontsize=20)
    plt.legend(loc='upper left')
    return plt


## Bollinger bands with ticks/ indications as to when to buy, hold or sell
def add_signal(data, is_daily, n, m, column):
     # takes dataframe on input
    # n = smoothing length
    # m = number of standard deviations away from MA
    
     # adds two columns to dataframe with buy, hold and sell signals
    buy_list = []
    sell_list = []
    hold_list = []

    df = pd.DataFrame({"close": data['Close'][1:], "predictions": data[column][1:]})

    # df['predictions']= data[column]
    
    # takes one column from dataframe
    df['B_MA'] = df['predictions'].rolling(n).mean()
    df['B_STD'] = df['predictions'].rolling(n).std() # Rolling Standard deviation for intermediate calculation
    df['BU']  = df['B_MA'] + (df['B_STD'] * 2)
    df['BL']  = df['B_MA'] - (df['B_STD'] * 2)
    df = df.dropna()
    df = df.reset_index(drop = True)

    if is_daily:
        print("inside daily loop")
        for i in df.index:
            if df.loc[i, 'predictions'] > df.loc[i, 'BU']:         
                buy_list.append(np.nan)
                sell_list.append(df.loc[i, 'predictions'])
                hold_list.append(np.nan)

            elif df.loc[i, 'predictions'] < df.loc[i, 'BL']:        
                buy_list.append(df.loc[i, 'predictions'])
                sell_list.append(np.nan)  
                hold_list.append(np.nan)
            else:
                buy_list.append(np.nan)
                sell_list.append(np.nan)
                hold_list.append(df.loc[i, 'predictions'])
    else:
        print("inside non daily loop")         
        for i in df.index:
            if df.loc[i, 'predictions'] > df.loc[i, 'BU']:         
                buy_list.append(np.nan)
                sell_list.append(df.loc[i, 'predictions'])
                hold_list.append(np.nan)

            elif df.loc[i, 'predictions'] < df.loc[i, 'BL']:        
                buy_list.append(df.loc[i, 'predictions'])
                sell_list.append(np.nan)
                hold_list.append(np.nan)
  
            else:
                buy_list.append(np.nan)
                sell_list.append(np.nan)
                hold_list.append(df.loc[i, 'predictions'])

         
    buy_list = pd.Series(buy_list, name='Buy')
    sell_list = pd.Series(sell_list, name='Sell')
    hold_list = pd.Series(hold_list, name = 'Hold')
        
    df = df.join(buy_list)
    df = df.join(sell_list)        
    df = df.join(hold_list)


    plt = plot_bollinger_bands(df, buy_list, sell_list, hold_list, column)
    return df, plt

def money_made(df: pd.DataFrame, duration: str):

    profit_list = []
    index_list = [0]

    [index_list.append(idx) for idx in df.index if df.loc[idx, 'Buy'] or df.loc[idx, 'Sell']];
    index_list.append(df.shape[0]-1)


    if df[df['Buy'] == True].shape[0] != 0 or df[df['Sell'] == True].shape[0] != 0:

        for j in range(len(index_list)):
            if j == len(index_list) - 1: break
            if j == len(index_list) - 2:
                start = index_list[j+1]
                end = index_list[-1]
            else:
                start = index_list[j]
                end = index_list[j+1]
                if j == len(index_list) - 2: break

            difference = df.loc[end, 'close'] - df.loc[start, 'close'] # last value - first value
            percent = np.round((difference/df.loc[start, 'close']) * 100, 2)
            profit_list.append(percent)

    # If only hold signal exists
    else:
        difference = df.loc[df.shape[0]-1, 'close'] - df.loc[0, 'close'] # last value - first value
        percent = np.round((difference/df.loc[0, 'close']) * 100, 2)
        profit_list.append(percent)

    logging.info(f'The recommendations from the LSTM model on {duration} basis helped us in making the following percentage of money: {profit_list}')

    return profit_list