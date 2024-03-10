import os
import sys

import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import seaborn as sns
import math
import pickle
from statsmodels.tsa.stattools import adfuller

from src.pipeline.result import Result
from src.exceptions import CustomException
from sklearn.metrics import mean_squared_error

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
    #plt.show()

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
    #plt.show()
    return predictions_df


def plot_losses(history):
    plt.plot(history.history["loss"], label="training loss")
    plt.plot(history.history["val_loss"], label="validation loss")
    plt.legend()
    #plt.show()

def prediction_report(lstm_model, train, test, X_train, y_train, X_test, y_test, n_time, duration_type: str):
    #Lets predict and check performance metrics
    train_predict = lstm_model.predict(X_train)
    test_predict = lstm_model.predict(X_test)

    #Calculate RMSE performance metrics
    math.sqrt(mean_squared_error(y_train, train_predict))
    #Test Data RMSE
    rmse = math.sqrt(mean_squared_error(y_test, test_predict))

    df1 = pd.DataFrame(train['Close'])
    df2 = pd.DataFrame(test['Close'])

    df = pd.concat([df1, df2])

    # Plot actual vs. predicted values
    # Shift train prediction for plotting
    
    trainPredictPlot = np.empty_like(df)
    trainPredictPlot[:,:] = np.nan
    trainPredictPlot[n_time:len(train_predict) + n_time, :] = train_predict

    # Shift test prediction for plotting
    testPredictPlot = np.empty_like(df)
    testPredictPlot[:,:] = np.nan
    testPredictPlot[len(train_predict) + (n_time * 2):len(df), :] = test_predict


    # Plot baseline and predictions

    lstm_predictions = pd.DataFrame({'Close': testPredictPlot.flatten(), 'Date': df.index})

    plt.figure(figsize=(15, 6))
    plt.plot(train['Close'], label="TRAIN", marker='.', linestyle='--')
    plt.plot(test['Close'], label="TEST", marker='x', linestyle='--')
    plt.plot('Date', 'Close', data=lstm_predictions , label="Predicted")

    plt.title('LSTM Model Prediction on {} Data'.format(duration_type))
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend()

    # fig_name = "lstm_" + str(n_time) + "_days"
    # file_path = "C:/Users/CHARU/Pictures/ML/MLProjects/static/images/"
    # final_path = os.path.join(file_path, fig_name) 
    # figure.savefig(final_path, format="png")
    #plt.show()
    plt.close()
    return testPredictPlot, rmse