from zenml import step
from typing import cast

from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer

from zenml.integrations.bentoml.services import BentoMLDeploymentService
from zenml.integrations.bentoml.model_deployers.bentoml_model_deployer import BentoMLModelDeployer


@step
def mlflow_prediction_service_loader(
        pipeline_name : str,
        step_name : str,
        running : bool = True,
        model_name : str = "model",
    )-> MLFlowDeploymentService:
    """
    Get the prediction service started by the deployment pipeline
    
    Args :
        Pipeline_name : name of pipeline that deployed the mlflow prediction server
        Step_name : name of step that deployed the mlflow prediction server
        running : whether the service is running or not
        model_name : name of the mlflow model that is deployed
    
    """
    # Get the mlflow model deployer stack component
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()

    # fetch existing services with same pipeline name and step name and model name

    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=step_name,
        model_name=model_name,
        running = running,
    )
    service = cast(MLFlowDeploymentService, existing_services[0])
    if existing_services:
        service = cast(MLFlowDeploymentService, existing_services[0])
        if service.is_running:
            print(
                f"The MLflow prediction server is running locally as a daemon "
                f"process service and accepts inference requests at:\n"
                f"    {service.prediction_url}\n"
                f"To stop the service, run "
                f"[italic green]`zenml model-deployer models delete "
                f"{str(service.uuid)}`[/italic green]."
            )
        elif service.is_failed:
            print(
                f"The MLflow prediction server is in a failed state:\n"
                f" Last state: '{service.status.state.value}'\n"
                f" Last error: '{service.status.last_error}'"
            )
    else:
        print(
            "No MLflow prediction server is currently running. The deployment "
            "pipeline must run first to train a model and deploy it. Execute "
            "the same command with the `--deploy` argument to deploy a model."
        )
    return service

@step
def bentoml_prediction_service_loader(
        pipeline_name : str,
        step_name : str,
        running : bool = True,
        model_name : str = "model",
    )-> BentoMLDeploymentService:
    """Get the BentoML prediction service started by the deployment pipeline.

    Args:
        pipeline_name: name of the pipeline that deployed the model.
        step_name: the name of the step that deployed the model.
        model_name: the name of the model that was deployed.
    """
    model_deployer = BentoMLModelDeployer.get_active_model_deployer()

    services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=step_name,
        model_name=model_name,
        running = running,
    )
    if not services:
        raise RuntimeError(
            f"No BentoML prediction server deployed by the "
            f"'{step_name}' step in the '{pipeline_name}' "
            f"pipeline for the '{model_name}' model is currently "
            f"running."
        )

    if not services[0].is_running:
        raise RuntimeError(
            f"The BentoML prediction server last deployed by the "
            f"'{step_name}' step in the '{pipeline_name}' "
            f"pipeline for the '{model_name}' model is not currently "
            f"running."
        )

    return cast(BentoMLDeploymentService, services[0])