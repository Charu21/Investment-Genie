import os
import sys

import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import seaborn as sns
import pickle
from statsmodels.tsa.stattools import adfuller

from src.exceptions import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def plot_time_series(timeseries: pd.DataFrame, y: list=[], labels: list=[], colors: list=[]):
    
    sns.set()
    fig, ax = plt.subplots(1, 1, figsize = (10,3))
    
    for i in range(len(y)):
        ax.plot(timeseries['Date'], timeseries[y[i]], label=labels[i], color=colors[i])
    
    ax.xaxis.set_major_locator(md.MonthLocator())
    ax.xaxis.set_major_formatter(md.DateFormatter('%Y-%m-%d'))
    ax.set_ylabel('Stock price')
    ax.set_xlabel('Date')
    ax.tick_params(labelrotation=45)
    ax.legend()
    plt.show()

# Checking for stationarity ( Augmented Dickey Fuller test)
def dickey_fuller_test(timeseries):
    result = adfuller(timeseries)
    print(f'ADF Statistics: {result[0]}')
    print(f'p-value: {result[1]}')
    print(f'Critical Value: {result[4]}')
    if result[1] > 0.05:
        print(f'The time series is not stationary!!! Further action needed!')
    else:
        print(f'The time series is stationary! No further action needed!')
    return result[1]
 


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def plot_graph(predictions: pd.Series, train: pd.DataFrame, test: pd.DataFrame, model_name: str) -> pd.DataFrame:
    # Plot actual vs. predicted values
    predictions_df = pd.DataFrame({'Close': predictions, 'Date': test.index})

    plt.figure(figsize=(12, 6))
    plt.plot(train['Close'], label="TRAIN", marker='.', linestyle='--')
    plt.plot(test['Close'], label="TEST", marker='x', linestyle='--')
    plt.plot('Date', 'Close', data=predictions_df, label="Predicted")

    plt.title(model_name.title() + ' Model Prediction for Differenced Data')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.show()
    return predictions_df


def plot_losses(history):
    plt.plot(history.history["loss"], label="training loss")
    plt.plot(history.history["val_loss"], label="validation loss")
    plt.legend()
    plt.show()

def prediction_report(result: Result, n_time, duration_type: str)-> np.array:
    #Lets predict and check performance metrics
    train_predict = result.model.predict(result.dataframe_x)
    test_predict = result.model.predict(result.df_xtest)

    #Calculate RMSE performance metrics
    math.sqrt(mean_squared_error(result.df_y, train_predict))
    #Test Data RMSE
    math.sqrt(mean_squared_error(result.df_ytest, test_predict))

    # Plot actual vs. predicted values
    # Shift train prediction for plotting
    trainPredictPlot = np.empty_like(df2)
    trainPredictPlot[:,:] = np.nan
    trainPredictPlot[n_time:len(train_predict) + n_time, :] = train_predict

    # Shift test prediction for plotting
    testPredictPlot = np.empty_like(df2)
    testPredictPlot[:,:] = np.nan
    testPredictPlot[len(train_predict) + (n_time * 2):len(df2), :] = test_predict


    # Plot baseline and predictions

    lstm_predictions = pd.DataFrame({'Close': testPredictPlot.flatten(), 'Date': df2.index})

    plt.figure(figsize=(15, 6))
    plt.plot(train['Close'], label="TRAIN", marker='.', linestyle='--')
    plt.plot(test['Close'], label="TEST", marker='x', linestyle='--')
    plt.plot('Date', 'Close', data=lstm_predictions , label="Predicted")

    plt.title('LSTM Model Prediction on {} Data'.format(duration_type))
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.show()

    return testPredictPlot