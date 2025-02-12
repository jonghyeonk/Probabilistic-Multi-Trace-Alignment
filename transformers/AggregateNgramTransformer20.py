from sklearn.base import TransformerMixin
import pandas as pd
import numpy as np
from time import time
from itertools import chain
from typing import Union, List, Tuple, Set
from pandas import DataFrame, Index, array


class AggregateNgramTransformer(TransformerMixin):
    
    def __init__(self, case_id_col: str, act_col: str, n: int, v: float, cat_cols: List[str], num_cols: List[str], boolean: bool=False, max_events: int = None, fillna: bool = True, create_dummies: bool = True):
        """
        Parameters
        -------------------
        case_id_col
            a column indicating the case identifier in an event log
        act_col
            a column indicating the activities in an event log
        n
            an int value in [2,3,4] for the size of sub-sequence in n-gram
        v
            a decay factor parameter in n-gram, ranged in [0,1]    
        time_col
            a column indicating the completed timestamp in an event log
        cat_cols
            columns indicating the categorical attributes in an event log
        num_cols
            columns indicating the numerical attributes in an event log       
        max_events
            maximum prefix length to be transformed  / Default: maximum prefix length in traces
        fillna
            TRUE: replace NA to 0 value in dataframe / FALSE: keep NA
        create_dummies        
            TRUE: transform categorical attributes as dummy attributes         
        """
        
        self.case_id_col = case_id_col
        self.act_col = act_col
        self.columns = None
        self.fit_time = 0
        self.transform_time = 0
        self.n = n
        self.v = v
        self.cat_cols = cat_cols    
        self.num_cols = num_cols       
        self.max_events = max_events   
        self.fillna = fillna            
        self.create_dummies = create_dummies
        self.boolean = boolean  
    
    
    def fit(self, X: DataFrame, y=None) -> None:
        return self
    
    def transform(self, X: DataFrame, y=None) -> DataFrame:
        """
        Tranforms the event log into an integrated encoded matrix with (1) a n-gram based encoded matrix from activity attribute and
        (2) a complex index-based encoded matrix from other categorical and numerical attributes:

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
        

        # transform activity col into ngram matrix
        ngram = X.groupby([self.case_id_col]).apply(lambda x: (['|'.join( x[self.act_col][i:i+self.n]) for i in range(0,len(x[self.act_col])-self.n+1)]) )

        nested_list = list(ngram.values)
        my_unnested_list = list(chain(*nested_list))
        ngram_list = list(set(my_unnested_list))
        
        X2 = X.groupby([self.case_id_col]).apply(lambda x: np.array(x[self.act_col]) )
        X2 = X2.apply(lambda x: self.func_ngram(x, n=self.n, v=0.7, ngram_list = ngram_list ))
        dt_transformed = pd.DataFrame(columns=  [i for i in ngram_list])
        for i in range(len(X2)):
            dt_transformed.loc[i] = X2[i]

        dt_transformed[self.case_id_col] = X2.index
        

        # transform numeric cols
        if len(self.num_cols) > 0:
            dt_numeric = X.groupby([self.case_id_col])[self.num_cols].agg(['mean','max', 'min', 'sum', 'std']) # 
            
            dt_numeric.columns = ['_'.join(col).strip() for col in dt_numeric.columns.values]
        

        # transform cat cols
        if len(self.cat_cols) > 0:
            dt_transformed2 = pd.get_dummies(X[self.cat_cols])
            dt_transformed2[self.case_id_col] = X[self.case_id_col]
            del X
            if self.boolean:
                dt_transformed2 = dt_transformed2.groupby(self.case_id_col).max()
            else:
                dt_transformed2 = dt_transformed2.groupby(self.case_id_col).sum()
            
        dt_transformed.index = dt_transformed[self.case_id_col]
        dt_transformed  = dt_transformed.drop(self.case_id_col, axis=1)

        if len(self.num_cols) > 0:
            dt_transformed = pd.concat([dt_transformed, dt_numeric], axis=1)
            del dt_numeric
 
        if len(self.cat_cols) > 0:
            dt_transformed = pd.concat([dt_transformed, dt_transformed2], axis=1)
    

        # fill missing values with 0-s
        if self.fillna:
            dt_transformed = dt_transformed.fillna(0)

        # add missing columns if necessary
        if self.columns is None:
            self.columns = dt_transformed.columns
        else:
            missing_cols = [col for col in self.columns if col not in dt_transformed.columns]
            for col in missing_cols:
                dt_transformed[col] = 0
            dt_transformed = dt_transformed[self.columns]

        self.transform_time = time() - start
        return dt_transformed
        
     
    def func_ngram(self, y: array, n: int, v: float, ngram_list: list) -> array:
        """
        Parameterizes and implements a n-gram encoding 
        
        Parameters
        -------------------
        y
            an array from each row in PandasDataframe
        n
            an int value in [2,3,4] for the size of sub-sequence in n-gram
        v
            a decay factor parameter in n-gram, ranged in [0,1]     
        ngram_list
            a list of subsequences for ngram
            
        Returns
        ------------------
        :rtype: array
            Transformed array
        """
        
        if n == 2:
            return np.array([self.twogram(x, y, v) for x in ngram_list])
        elif n == 3:
            return np.array([self.threegram(x, y, v) for x in ngram_list])
        elif n == 4:
            return np.array([self.fourgram(x, y, v) for x in ngram_list])
    
    
    def twogram(self, x: str, y: array, v: float) -> float:
        """
        Tranforms the input array y into a 2-gram encoded array:
        
        Parameters
        -------------------
        x
            a subsequence from ngram_list
        y
            an array from each row in PandasDataframe
        v
            a decay factor parameter in n-gram, ranged in [0,1]     
        ngram_list
            a list of subsequences for ngram
            
        Returns
        ------------------
        :rtype: array
            a calculated score regarding a subsequence x
        """
        
        terms = x.split('|')
        loc1 = [i for i, e in enumerate(y) if e == terms[0]]
        layer1 = list()
        if len(loc1) >0:
            for i in loc1:      
                loc2 = [j for j in range(i, len(y)) if y[j] == terms[1]]
                layer1.append(sum(v**(np.array(loc2) - i)))
        else:
            layer1.append(0)
        
        return np.sum(layer1)
    
    def threegram(self, x: str, y: array, v: float) -> float:
        """
        Tranforms the input array y into a 3-gram encoded array:
        
        Parameters
        -------------------
        x
            a subsequence from ngram_list
        y
            an array from each row in PandasDataframe
        v
            a decay factor parameter in n-gram, ranged in [0,1]     
        ngram_list
            a list of subsequences for ngram
            
        Returns
        ------------------
        :rtype: array
            a calculated score regarding a subsequence x
        """
        
        terms = x.split('|')
        loc1 = [i for i, e in enumerate(y) if e == terms[0]]
        layer1 = list()
        if len(loc1) >0:
            for i in loc1:      
                loc2 = [j for j in range(i, len(y)) if y[j] == terms[1]]
                if len(loc2) > 0:
                    for x in loc2:
                        loc3 = [j for j in range(x, len(y)) if y[j] == terms[2]]
                        layer1.append( sum(v**(np.array(loc3) - i -1)))  
                else:
                    layer1.append(0)
        else:
            layer1.append(0)
        return np.sum(layer1)
    
    
    def fourgram(self, x: str, y: array, v: float) -> float:
        """
        Tranforms the input array y into a 4-gram encoded array:
        
        Parameters
        -------------------
        x
            a subsequence from ngram_list
        y
            an array from each row in PandasDataframe
        v
            a decay factor parameter in n-gram, ranged in [0,1]     
        ngram_list
            a list of subsequences for ngram
            
        Returns
        ------------------
        :rtype: array
            a calculated score regarding a subsequence x
        """
        
        terms = x.split('|')
        loc1 = [i for i, e in enumerate(y) if e == terms[0]]
        layer1 = list()
        if len(loc1) >0:
            for i in loc1:      
                loc2 = [j for j in range(i, len(y)) if y[j] == terms[1]]
                if len(loc2) > 0:
                    for x in loc2:
                        loc3 = [j for j in range(x, len(y)) if y[j] == terms[2]]
                        if len(loc3) > 0:
                            for w in loc3:
                                loc4 = [j for j in range(w, len(y)) if y[j] == terms[3]]
                                layer1.append( sum(v**(np.array(loc4) - i-2)))
                        else:
                            layer1.append(0)       
                else:
                    layer1.append(0)
        else:
            layer1.append(0)
        return np.sum(layer1)
    
    
    def get_feature_names(self) -> Index:
        """
        Print all attribute names in a Pandas DataFrame:

        Returns
        ------------------
        :rtype: Index
            column names of a Pandas DataFrame
        """
        return self.columns