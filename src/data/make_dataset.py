import os
import sys
from src.exceptions import CustomException
from src.logger import logging
import pandas as pd
import yfinance as yf

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    raw_data_path: str=os.path.join('data/raw', "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def fetch_data(self):
        logging.info("Starting data fetch ...")

        try:
            data = yf.download("MSFT", start="2020-01-01")
            logging.info('Read the Microsoft data from Yahoo finance')
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path, header=True)
            
            logging.info("Make Dataset step is completed!")

            return(
                self.ingestion_config.raw_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)