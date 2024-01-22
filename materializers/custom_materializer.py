import json
import os
import pickle
from typing import Any, List, Type, Union

import joblib
from sklearn.linear_model import LogisticRegression
from zenml.materializers.base_materializer import BaseMaterializer
from zenml.enums import ArtifactType
from zenml.io import fileio

DEFAULT_FILENAME = "RetailPriceOptimizationEnv"

class ListMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (list,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def load(self, data_type: Type[Any]) -> list:
        """Read from artifact store."""
        list_path = os.path.join(self.uri, 'list.json')
        with fileio.open(list_path, 'r') as f:
            data = json.load(f)
        return data

    def save(self, data: list) -> None:
        """Write to artifact store."""
        list_path = os.path.join(self.uri, 'list.json')
        with fileio.open(list_path, 'w') as f:
            json.dump(data, f)


class SKLearnModelMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (LogisticRegression, )
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.MODEL

    def load(self, data_type: Type[LogisticRegression]) -> LogisticRegression:
        """Read from artifact store."""
        model_path = os.path.join(self.uri, 'model.joblib')
        return joblib.load(model_path)

    def save(self, model: LogisticRegression) -> None:
        """Write to artifact store."""
        model_path = os.path.join(self.uri, 'model.joblib')
        joblib.dump(model, model_path)


# import statsmodels.api as sm

# class StatsModelMaterializer(BaseMaterializer):
#     ASSOCIATED_TYPES = (sm.regression.linear_model.RegressionResultsWrapper, )
#     ASSOCIATED_ARTIFACT_TYPE = ArtifactType.MODEL

#     def load(self, data_type: Type[sm.regression.linear_model.RegressionResultsWrapper]) -> sm.regression.linear_model.RegressionResultsWrapper:
#         """Read from artifact store."""
#         model_path = os.path.join(self.uri, 'model.pickle')
#         return sm.load(model_path)

#     def save(self, model: sm.regression.linear_model.RegressionResultsWrapper) -> None:
#         """Write to artifact store."""
#         model_path = os.path.join(self.uri, 'model.pickle')
#         model.save(model_path, remove_data=True)
        # joblib.dump(model, model_path)