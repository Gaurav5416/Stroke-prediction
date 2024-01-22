import logging
from typing_extensions import Annotated, Tuple
import numpy as np  
import pandas as pd 

from zenml import step
from steps.ingest_data import ingest_data
from steps.process_data import categorical_encoding

@step
def dynamic_importer()->Tuple[
    Annotated[pd.DataFrame, "df_sample"],
    Annotated[list, "predictors"],
    ]:
    try:
        df = ingest_data("stroke", for_predict=True) 
        df_processed = categorical_encoding(df)
        df_sample = df_processed.sample(n=10)
        print(df_processed.head())
        predictors = df_sample.columns.tolist()
        print(predictors)
        return df_sample, predictors
    
    except Exception as e:
        logging.exception(e)
        return e
        

