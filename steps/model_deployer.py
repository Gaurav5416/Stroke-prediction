
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.integrations.bentoml.steps import bentoml_model_deployer_step
from constants import MODEL_NAME


mlflow_model_deployer = mlflow_model_deployer_step


bentoml_model_deployer = bentoml_model_deployer_step.with_options(
    parameters = dict(
        model_name = MODEL_NAME,
        port = 3001,
        production = False,
        timeout = 1000,
    )
)