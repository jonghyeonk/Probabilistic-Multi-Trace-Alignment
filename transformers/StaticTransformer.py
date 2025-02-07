from sklearn.base import TransformerMixin
import pandas as pd
from time import time
from typing import Union, List, Tuple, Set
from pandas import DataFrame, Index


class StaticTransformer(TransformerMixin):
    
    def __init__(self, case_id_col: str , cat_cols: List[str], num_cols: List[str], fillna: bool=True):
        """
        Parameters
        -------------------
        case_id_col
            a column indicating the case identifier in an event log
        cat_cols
            columns indicating the categorical attributes in an event log
        num_cols
            columns indicating the numerical attributes in an event log       
        fillna
            TRUE: replace NA to 0 value in dataframe / FALSE: keep NA        
        """
        
        self.case_id_col = case_id_col
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.fillna = fillna
        self.columns = None
        self.fit_time = 0
        self.transform_time = 0
    
    
    def fit(self, X: DataFrame, y=None) -> None:
        return self
    
    
    def transform(self, X: DataFrame, y=None) -> DataFrame:
        """
        Tranforms the event log X into a static encoded matrix (calling the first state for each case id):

        Parameters
        -------------------
        X: DataFrame
            Event log / Pandas DataFrame to be transformed
            
        Returns
        ------------------
        :rtype: DataFrame
            Transformed event log
        """ 

        start = time()
        
        dt_first = X.groupby(self.case_id_col).first()
        # transform numeric cols
        dt_transformed = dt_first[self.num_cols]
        # transform cat cols
        if len(self.cat_cols) > 0:
            dt_cat = pd.get_dummies(dt_first[self.cat_cols])
            dt_transformed = pd.concat([dt_transformed, dt_cat], axis=1)
        # fill NA with 0 if requested
        if self.fillna:
            dt_transformed = dt_transformed.fillna(0)
            
        # add missing columns if necessary
        if self.columns is not None:
            missing_cols = [col for col in self.columns if col not in dt_transformed.columns]
            for col in missing_cols:
                dt_transformed[col] = 0
            dt_transformed = dt_transformed[self.columns]
        else:
            self.columns = dt_transformed.columns
        
        self.transform_time = time() - start
        return dt_transformed
    
    
    def get_feature_names(self) -> Index:
        """
        Print all attribute names in a Pandas DataFrame:

        Returns
        ------------------
        :rtype: Index
            column names of a Pandas DataFrame
        """
        return self.columns