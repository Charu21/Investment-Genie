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

from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import r2_score

from src.exceptions import CustomException
from src.logger import logging

from src.utils import plot_graph, save_object

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
                # LSTM_layer, dense, scaler, inputs = self.fit_LSTM(to_fit)
                
                # # forecast = self.predict_LSTM(inputs,
                # #                             scaler,
                # #                             LSTM_layer,
                # #                             dense,
                # #                             current_date,
                # #                             interval,
                # #                             periods)

                # prediction = forecast.iloc[-1] # Predicted future stock price
                # next_date = forecast.index[-1]
                print(f' LSTM has been called!')
            # elif self.model_name == 'prophet':
            #     predictor = Prophet()
            #     predictor.fit(to_fit)
            #     # Predictor is now fitted to data prior to current_date
            #     freq_dict = {1:"d", 7:"w", 30:"m", 90: 'Q'}
            #     future = predictor.make_future_dataframe(periods = periods, 
            #                                             freq = freq_dict[interval], 
            #                                             include_history = False)
            #     forecast = predictor.predict(future)
            #     print(forecast)
            #     forecast.rename(columns={'yhat': 'y'}, inplace = True)
            #     prediction = forecast['y'].iloc[-1] # Predicted future stock price
            #     next_date = forecast['ds'].iloc[-1]
            #     print(f'\ncurrent_date = {current_date}', f'\nfuture={future}', f'\nforecast={forecast}')
            #     print(f"last forecast = {forecast.iloc[-1]['ds']}")
            elif self.model_name == 'AR':
                ar_model = apply_autoreg(train, test)
                forecast = ar_model.predict(start=train_index, end=train_index + test_index -1)
                predictions_df = plot_graph(forecast, train, test, "AR")
                mse = mean_squared_error(test['Close'], predictions_df['Close'])
                print(f'Mean Squared Error (MSE): {mse}')
                return mse
            # elif self.model_name == 'arima':
            #     mask = history.index <= current_date
            #     to_fit = history[mask]
            #     to_fit = to_fit.set_index('ds')
            #     to_fit.index = pd.DatetimeIndex(to_fit.index).to_period('D')
            #     predictor = ARIMA(to_fit)
            #     # Predictor is now fitted to data prior to current_date
            #     forecast = predictor.predict()
            #     forecast = forecast[forecast.index>=start]
            #     prediction = forecast.iloc[-1] # Predicted future stock price
            #     next_date = forecast.index[-1]
            # elif self.model_name == 'sarimax':
            #     mask = history.index <= current_date
            #     to_fit = history[mask]
            #     to_fit = to_fit.set_index('ds')
            #     to_fit.index = pd.DatetimeIndex(to_fit.index).to_period('D')
            #     predictor = SARIMAX(to_fit)
            #     # Predictor is now fitted to data prior to current_date
            #     forecast = predictor.predict()
            #     forecast = forecast[forecast.index>=start]
            #     prediction = forecast.iloc[-1] # Predicted future stock price
            #     next_date = forecast.index[-1]
            # elif self.model_name == 'simpleexpsmoothing':
            #     mask = history.index <= current_date
            #     to_fit = history[mask]
            #     to_fit = to_fit.set_index('ds')
            #     to_fit.index = pd.DatetimeIndex(to_fit.index).to_period('D')
            #     predictor = SimpleExpSmoothing(to_fit)
            #     # Predictor is now fitted to data prior to current_date
            #     forecast = predictor.predict()
            #     forecast = forecast[forecast.index>=start]
            #     prediction = forecast.iloc[-1] # Predicted future stock price
            #     next_date = forecast.index[-1]

        # return forecast, prediction, next_date
            # model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                            #  models=models,param=params)
            
            # ## To get best model score from dict
            # best_model_score = max(sorted(model_report.values()))

            # ## To get best model name from dict

            # best_model_name = list(model_report.keys())[
            #     list(model_report.values()).index(best_model_score)
            # ]
            # best_model = models[best_model_name]

            # if best_model_score<0.6:
            #     raise CustomException("No best model found")
            # logging.info(f"Best found model on both training and testing dataset")

            # save_object(
            #     file_path=self.model_trainer_config.trained_model_file_path,
            #     obj=best_model
            # )

            # predicted=best_model.predict(X_test)

            # r2_square = r2_score(y_test, predicted)
            # return(
            #     self.model_trainer_config.
            #     self.feature_transformation_config.test_data_path

            # )

            



            
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
    