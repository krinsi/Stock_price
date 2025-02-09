import sys
import os
import pandas as pd
from src.exception import CustomException
from src.util import load_object

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # Define paths for model and preprocessor
            
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            print("Before Loading")

            # Load model and preprocessor objects
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            print("After Loading")

            # Transform input data using preprocessor
            data_scaled = preprocessor.transform(features)

            # Predict using the model
            preds = model.predict(data_scaled)
            return preds

        except FileNotFoundError as fnfe:
            raise CustomException(f"File not found: {fnfe}", sys)
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
   def __init__(
        self,
        Ticker: str,
        Date: str,
        Open: float,
        High: float,
        Low: float,
        Close: float,
        Volume: int,
    ):
        self.Ticker = Ticker
        self.Date = Date
        self.Open = Open
        self.High = High
        self.Low = Low
        self.Close = Close
        self.Volume = Volume

def get_data_as_data_frame(self):
        try:
            # Create a dictionary of input data
            stock_data_input_dict = {
                "Ticker": [self.Ticker],
                "Date": [self.Date],
                "Open": [self.Open],
                "High": [self.High],
                "Low": [self.Low],
                "Close": [self.Close],
                "Volume": [self.Volume],
            }

            # Convert to DataFrame
            return pd.DataFrame(stock_data_input_dict)
        except Exception as e:
            raise Exception(f"Error in creating DataFrame: {e}")
