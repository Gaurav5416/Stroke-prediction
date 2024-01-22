from pipelines.inference_pipeline import inference_pipeline
from pipelines.continuous_deployment_pipeline import continuous_deployment_pipeline
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.client import Client

import click

DEPLOY = "deploy"
PREDICT = "predict"
DEPLOY_AND_PREDICT = "deploy_and_predict"
 
MLFLOW = "mlflow"
BENTOML = "bentoml"

@click.command()
@click.option(
    "--config",
    type=click.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT]),
    default=DEPLOY_AND_PREDICT,

)
@click.option(
    "--deployer",
    type=click.Choice([MLFLOW, BENTOML]),
    default=MLFLOW,
    help="Optionally you can choose to deploy"
    "deploy the model using MLflow(local deployment only), or to "
    "deploy the model using BentoML, or to ",
)



def main(config:str, deployer : str):

    
    deploy = config == DEPLOY or config == DEPLOY_AND_PREDICT
    predict = config == PREDICT or config == DEPLOY_AND_PREDICT
    
    if deploy:
        continuous_deployment_pipeline(deployer = deployer)
    if predict :
        if deployer == 'mlflow':
            step_name = "mlflow_model_deployer_step"
        if deployer == 'bentoml':
            step_name = "bentoml_model_deployer_step"
        inference_pipeline(
            pipeline_name = "continuous_deployment_pipeline", 
            step_name = step_name,
            deployer = deployer)

    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri '{get_tracking_uri()}'\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the `mlflow_example_pipeline`"
        "experiment. Here you'll also be able to compare the two runs.)"
    )
    print(Client().active_stack.experiment_tracker.get_tracking_uri())


if __name__ == "__main__":
    main()