from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from typing import List
import numpy as np 
import pandas as pd 


class CategoricalEncoder:
    """
    This class applies encoding to categorical columns
    
    Parameters :
    ---------------

    method : str, default = "onehot"
        The method to encode the categorical columns can be "onehot" or "ordinal"

    categories : 'auto' or a list of lists, default = 'auto'
        Categories for the encoders. Must match the number of columns. If 'auto', Categories are determined from data.
    """

    def __init__(self, method : str = "onehot", categories : str = "auto"):
        self.method = method
        self.categories = categories
        self.encoder = {}

    def fit(self, df:pd.DataFrame, columns)->None:
        """
        Looks at the categorical column and take the unique values in it.

        Args :
        ---------------
            df : Dataframe
            columns : list of categorical column names.

        Returns : 
        ---------------
            None
        """

        for col in columns :
            if self.method == "onehot":
                self.encoder[col] = OneHotEncoder(sparse_output=False, categories=self.categories) #initializing one hot encoder class as instance
            elif self.method == "ordinal":
                self.encoder[col] = OrdinalEncoder(categories=self.categories)#initializing ordinal encoder class as instance
            else:
                raise ValueError("Unknown method. Please use one of 'one hot' or 'ordinal'")
            self.encoder[col].fit(df[[col]])


    def transform(self, df, columns):
        """
        Transform the dataframe encoding the categorical columns

        Args :
        ---------------
            df : Dataframe
            columns : list of categorical column names.

        Returns :
        --------------- 
            None
        """
        df_encoded = df.copy()
        for col in columns :
            transformed = self.encoder[col].transform(df[[col]])
            if self.method == "onehot":
                transformed = pd.DataFrame(transformed, columns = self.encoder[col].get_feature_names_out([col]))
                df_encoded = pd.concat([df_encoded.drop(columns=[col]), transformed], axis = 1)
            else :
                df_encoded[col] = transformed
        return df_encoded
    

    def fit_transform(self, df, columns):
        self.fit(df, columns)
        return self.transform(df, columns)

