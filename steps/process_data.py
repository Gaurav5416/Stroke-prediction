import pandas as pd 
from zenml import step
from zenml.logger import get_logger

from steps.src.data_processor import CategoricalEncoder

logger = get_logger(__name__)

@step(enable_cache=True)
def categorical_encoding(df: pd.DataFrame)-> pd.DataFrame :
    try :
        encoder = CategoricalEncoder(method="onehot")
        df = encoder.fit_transform(df, columns = ["gender", "married", "work_type", "residence", "smoking"])
        logger.info("Successfully encoded categorical variables")
        df.drop("id", inplace=True, axis=1)
        print(df.head())
        return df
    except Exception as e :
        logger.error("Failed to encode categorical variables")
        raise e