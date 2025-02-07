from sklearn.base import TransformerMixin
import pandas as pd
import numpy as np
from time import time
from itertools import chain
from typing import Union, List, Tuple, Set
from pandas import DataFrame, Index, array

class NgramTransformer(TransformerMixin):
    
    def __init__(self, case_id_col: str, act_col: str, n: int, v: float):
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
        """
        
        self.case_id_col = case_id_col
        self.act_col = act_col
        self.columns = None
        self.fit_time = 0
        self.transform_time = 0
        self.n = n
        self.v = v
    
    
    def fit(self, X: DataFrame, y=None) -> None:
        return self
    
    def transform(self, X: DataFrame, y=None) -> DataFrame:
        """
        Tranforms the event log into a n-gram based encoded matrix, considering control-flow perspective of the event log:

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
        ngram = X.groupby([self.case_id_col], sort=False).apply(lambda x: (['|'.join( x[self.act_col][i:i+self.n]) for i in range(0,len(x[self.act_col])-self.n+1)]) )

        nested_list = list(ngram.values)
        my_unnested_list = list(chain(*nested_list))
        ngram_list = list(set(my_unnested_list))
        
        
        X = X.groupby([self.case_id_col], sort=False).apply(lambda x: np.array(x[self.act_col]) )
        X = X.apply(lambda x: self.func_ngram(x, n=self.n, v=0.7, ngram_list = ngram_list ))
        
        dt_transformed = pd.DataFrame(columns=  [i for i in ngram_list])
        for i in range(len(X)):
            dt_transformed.loc[i] = X[i]
        
        dt_transformed.index = X.index

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

            loc2 = [j for j in range(loc1[0], len(y)) if y[j] == terms[1]]
            layer1.append(sum(v**(np.array(loc2) - loc1[0])))
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