from typing import List, Tuple
from typing_extensions import Annotated
from sklearn.tree import DecisionTreeClassifier
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.base import ClassifierMixin, RegressorMixin
from zenml import step
from zenml.client import Client
from zenml.logger import get_logger
from zenml.integrations.mlflow.experiment_trackers import MLFlowExperimentTracker

# from materializers.custom_materializer import ListMaterializer, SKLearnModelMaterializer
logger = get_logger(__name__)
experiment_tracker = Client().active_stack.experiment_tracker

@step(enable_cache=False, experiment_tracker=experiment_tracker.name)
def sklearn_train(
    X_train: Annotated[pd.DataFrame, "X_train"],
    y_train: Annotated[pd.Series, "y_train"]
)-> ClassifierMixin :
    """Trains a logistic regression model and outputs the summary."""
    try:
            mlflow.sklearn.autolog()
            # model = LogisticRegression()
            model = DecisionTreeClassifier()
            model.fit(X_train, y_train)
            return model
    except Exception as e:
        logger.error(e)
        raise e
