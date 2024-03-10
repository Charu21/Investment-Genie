import sys
import pandas as pd
from src.data.make_dataset import DataIngestion
from src.exceptions import CustomException
from src.features.build_features import FeatureTransformation
from src.pipeline.train_model import ModelConfig
from src.utils import load_object

class PredictModel:
    def __init__(self):
        pass

    def evaluate_model(self, model_name: str, duration: str):
        try:
            model = ModelConfig(model_name, duration)
            obj=DataIngestion()
            raw_data = obj.fetch_data()

            data_eda=FeatureTransformation()
            train_arr,test_arr = data_eda.perform_eda(raw_data)
            result = model.evaluate(train_arr, test_arr)
            return result
        except Exception as e:
            raise CustomException(e, sys)

    def predict_from_model(self, model_name: str, duration: str):
        try:
            model = ModelConfig(model_name, duration)
            obj=DataIngestion()
            raw_data = obj.fetch_data()

            data_eda=FeatureTransformation()
            train_arr,test_arr = data_eda.perform_eda(raw_data)
            result, filepath = model.predict(train_arr, test_arr)
            return result, filepath
        except Exception as e:
            raise CustomException(e, sys)
class CustomData:
    def __init__(self,
                 model: str,
                 duration: str):
        self.model = model
        self.duration = duration

    def get_data_as_dataframe(self):
        try:
            user_data = {
                "model_name": [self.model],
                "duration": [self.duration]
            }
            return pd.DataFrame(user_data)
        
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    raw_data = obj.fetch_data()

    data_eda=FeatureTransformation()
    train_arr,test_arr = data_eda.perform_eda(raw_data)