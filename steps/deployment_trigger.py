from zenml import step
from zenml.steps import BaseParameters

class DeploymentTriggerParameters(BaseParameters):
    """Parameters that are used to trigger the deployment"""

    min_accuracy: float = 0


@step(enable_cache=False)
def deployment_trigger(
    accuracy: float,
    params: DeploymentTriggerParameters,
) -> bool:
    """Implements a simple model deployment trigger that looks at the
    input model accuracy and decides if it is good enough to deploy"""
    decision = accuracy > params.min_accuracy
    decision = True
    return decision