import logging
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

class Evaluation(ABC):
    """
    Abstract base class for evaluation metrics
    """

    @abstractmethod
    def calculate_scores(self, y_true:np.ndarray, y_pred:np.ndarray):
        """
        Abstract method that calclates scores for the model

        Args :
            y_true : True labels
            y_pred : predicted labels

        Returns:
            None
        """


class MSE(Evaluation):
    """
    Evaluation Strategy that uses Mean square error
    """

    def calculate_scores(self, y_true:np.ndarray, y_pred:np.ndarray):
        try:
            logging.info("calculating MSE")
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f"MSE : {mse}")
            return mse

        except Exception as e:
            logging.error(f"Error in calculating MSE : {e}")

class RMSE(Evaluation):
    """
    Evaluation Strategy that uses Root mean square error
    """

    def calculate_scores(self, y_true:np.ndarray, y_pred:np.ndarray):
        try:
            logging.info("calculating RMSE")
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            logging.info(f"RMSE : {rmse}")
            return rmse

        except Exception as e:
            logging.error(f"Error in calculating RMSE : {e}")  
            raise e

class R2(Evaluation):
    """
    Evaluation Strategy that uses R2 score
    """

    def calculate_scores(self, y_true:np.ndarray, y_pred:np.ndarray):
        try:
            logging.info("calculating R2")
            r2 =r2_score(y_true, y_pred)
            logging.info(f"R2 : {r2}")
            return r2

        except Exception as e:
            logging.error(f"Error in calculating R2 : {e}")