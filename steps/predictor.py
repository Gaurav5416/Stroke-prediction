import json
import numpy as np
import pandas as pd
from zenml import step
from rich import print as rich_print

from zenml.integrations.mlflow.services import MLFlowDeploymentService

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