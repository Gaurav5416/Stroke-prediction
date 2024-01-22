from zenml import pipeline
from zenml.config import DockerSettings

from zenml.integrations.constants import MLFLOW, BENTOML
from constants import MODEL_NAME
from steps.dynamic_importer import dynamic_importer
from steps.prediction_service_loader import mlflow_prediction_service_loader, bentoml_prediction_service_loader
from steps.predictor import mlflow_predictor, bentoml_predictor

docker_settings = DockerSettings(required_integrations=[MLFLOW])
 
@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline(
     pipeline_name:str,
     step_name:str,
     deployer : str,
     ):
     
     df_sample, predictors = dynamic_importer()

     if deployer == 'mlflow' or deployer == 'MLFLOW' :
        service = mlflow_prediction_service_loader(
            pipeline_name=pipeline_name,
            step_name='mlflow_model_deployer_step',
            running = False,
            )
        mlflow_predictor(service = service, data=df_sample, columns = predictors)
     
     if deployer == 'bentoml' or deployer == 'BENTOML' :
        service = bentoml_prediction_service_loader(
            model_name =  MODEL_NAME,
            pipeline_name=pipeline_name,
            step_name='bentoml_model_deployer_step',
            running = False,
            )
        bentoml_predictor(service = service, data=df_sample)