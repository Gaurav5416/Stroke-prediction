import os
import logging
import pandas as pd
from zenml import step
from dotenv import load_dotenv

from steps.src.data_loader import DataLoader

@step(enable_cache=False)
def ingest_data(
    table_name:str,
    for_predict : bool = False,
)-> pd.DataFrame:
    """Read data from SQL table and return a pandas dataframe
    
    Args :
        table_name : Name of the table to read from
    
    Returns : 
        Pandas.DataFrame"""
     
    try:
        load_dotenv()
        db_url = os.getenv('DB_URL')
        data_loader = DataLoader(db_url)
        data_loader.load_data(table_name)
        df = data_loader.get_data()
        if for_predict :
            df.drop(columns = ['stroke'], inplace = True)
        logging.info(f"Successfully ingest data from {table_name}")
        return df
    
    except Exception as e :
        logging.error(f"Error reading data from {table_name}")
        raise e

