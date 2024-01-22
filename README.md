Stroke prediction System

Model - Logistic Regression


## 🐍 Python Requirements

```Run these commands in the wsl terminal

pip install -r requirements.txt
zenml up
zenml integration install mlflow -y
zenml integration install bentoml -y
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer register mlflow_deployer --flavor=mlflow
zenml model-deployer register bentoml_deployer --flavor=bentoml
zenml stack register local_mlflow_stack -a default -o default -d mlflow_deployer -e mlflow_tracker
zenml stack register local_bentoml_stack -a default -o default -d bentoml_deployer -e mlflow_tracker
```
----------------------------------------------------------------------------------------

You can choose to either deploy model using bentoml or mlflow using "--deployer" option. Please make sure to set the relevant zenml stack as per the deployer.

run''
zenml stack set local_bentoml_stack
python run_pipeline.py --config deploy_and_predict --deployer bentoml

or 

zenml stack set local_mlflow_stack
python run_pipeline.py --config deploy_and_predict --deployer mlflow


---------------------------------------------------------------------------------------

Create a .env file specifying datapath of table and database path eg..
DB_URL = "postgresql://postgres:1234@localhost:5432/stroke"
datapath = "/mnt/c/name/stoke_system/data/stroke_capped.csv"