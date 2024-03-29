from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW
from zenml import pipeline


from constants import MODEL_NAME
from zenml import __version__ as zenml_version

from steps.ingest_data import ingest_data
from steps.process_data import categorical_encoding
from steps.data_splitter import split_data
from steps.train_model import sklearn_train
from steps.evaluate_model import evaluate_model
from steps.deployment_trigger import deployment_trigger
from steps.model_deployer import mlflow_model_deployer

docker_settings = DockerSettings(required_integrations=[MLFLOW])

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def continuous_deployment_pipeline(
    deployer : str,
):
    # Link all the steps artifacts together
    df = ingest_data("stroke") 
    df_processed = categorical_encoding(df)  
    X_train, X_test, y_train, y_test = split_data(df_processed)  
    model = sklearn_train(X_train, y_train)  
    r2, rmse = evaluate_model(model, X_test, y_test)
    decision = deployment_trigger(accuracy=r2) 
    mlflow_model_deployer(model = model, deploy_decision = decision)