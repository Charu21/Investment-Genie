import sys
from dataclasses import dataclass

import pandas as pd

from src.utils import dickey_fuller_test
from src.data.make_dataset import DataIngestion
from src.exceptions import CustomException
from src.logger import logging
import os

@dataclass
class FeaturesConfig:
    train_data_path: str=os.path.join('data/processed', "train.csv")
    test_data_path: str=os.path.join('data/processed', "test.csv")

class FeatureTransformation:
    def __init__(self):
        self.feature_transformation_config = FeaturesConfig()

    def perform_eda(self, data_path):
        try:
            data_df = pd.read_csv(data_path) 
            logging.info("Read train and test data completed")
            
            data = data_df[['Close']].copy()

            dickey_fuller_test(data['Close'])
            data['Close_Diff'] = data['Close'].diff()
            data = data.dropna(how='any')

            # Again check for stationarity
            dickey_fuller_test(data['Close_Diff'])

             # Taking 80 % as training
            train_size = int(len(data) * 0.8)
            train, test = data.iloc[:train_size], data.iloc[train_size:]
            train.to_csv(self.feature_transformation_config.train_data_path, header=True)
            test.to_csv(self.feature_transformation_config.test_data_path, header=True)
            logging.info("Feature transformation step is completed!")

            return(
                self.feature_transformation_config.train_data_path,
                self.feature_transformation_config.test_data_path

            )

        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj=DataIngestion()
    raw_data = obj.fetch_data()

    data_eda=FeatureTransformation()
    train_arr,test_arr = data_eda.perform_eda(raw_data)