import logging
import pandas as pd 
import mlflow
from typing import Tuple
from typing_extensions import Annotated

from zenml import step
from sklearn.linear_model import LogisticRegression

from steps.src.model_evaluation import MSE, RMSE, R2

@step(enable_cache=True)
def evaluate_model(
    model:LogisticRegression, 
    X_test:pd.DataFrame, 
    y_test:pd.Series
    )->Tuple[
        Annotated[float,'r2'],
        Annotated[float,'rmse'],
    ]:

    """
    Evaluates the model

    Args :
    -------
        X_test: pd.DataFrame,
        y_test: pd.Series,

    Returns : 
    ----------
        r2_score : float
        rmse : float
    """
    try:
        prediction = model.predict(X_test)
        mse = MSE().calculate_scores(y_test, prediction)
        # mlflow.log_metric("mse", mse)
        prediction = model.predict(X_test)
        rmse = RMSE().calculate_scores(y_test, prediction)
        # mlflow.log_metric("rmse", rmse)
        prediction = model.predict(X_test)
        r2 = R2().calculate_scores(y_test, prediction)
        # mlflow.log_metric("r2", r2)
        return r2, rmse
        
    except Exception as e:
        logging.error(f"error evaluating the model : {e}")
        raise e






