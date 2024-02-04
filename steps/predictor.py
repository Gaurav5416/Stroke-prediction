import json
import numpy as np
import pandas as pd
from zenml import step
from rich import print as rich_print

from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.bentoml.services import BentoMLDeploymentService

@step(enable_cache=False)
def mlflow_predictor(
    service : MLFlowDeploymentService,
    data:pd.DataFrame,
    columns : list,
    )->np.ndarray :
    
    service.start(timeout=200)

    data = data.to_json(orient="split")
    data = json.loads(data)
    data.pop("columns")
    data.pop("index")
    df = pd.DataFrame(data["data"], columns=columns)
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data = np.array(json_list)

    prediction = service.predict(data)
    rich_print(prediction)
    return prediction

@step
def bentoml_predictor(
    service : BentoMLDeploymentService,
    data:pd.DataFrame,
    )->np.ndarray :
    
    service.start(timeout=60)
    data = data.to_numpy()
    prediction = service.predict("predict_ndarray", data)
    rich_print(prediction)
    return prediction