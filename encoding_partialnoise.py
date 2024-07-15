

import os
from sys import maxsize
import numpy as np
import pandas as pd
import random
from random import choices
import warnings
warnings.filterwarnings(action='ignore')
dir_home = os.getcwd()

from transformers.LastStateTransformer import LastStateTransformer
from transformers.AggregateTransformer import AggregateTransformer
from transformers.IndexBasedTransformer import IndexBasedTransformer
from transformers.ComplexIndexBasedTransformer import ComplexIndexBasedTransformer
from transformers.ComplexIndexNgramTransformer import ComplexIndexNgramTransformer 
from transformers.IndexNgramTransformer import IndexNgramTransformer
from transformers.AggregateNgramTransformer20 import AggregateNgramTransformer

from sklearn.pipeline import Pipeline

import time
from sklearn.cluster import DBSCAN
dir_ref = dir_home +"/ref"
dir_data = dir_home +"/org_log"


name = 'credit-card_new' # 'mccloud_new' , 'road_weibull4', 'helpdesk',  'hospital billing'


data = pd.read_csv(dir_data + "/" + name + ".csv")

p_list = [3]
pt_list = ['skip', 'repeat', 'replace', 'replace_attr']


def repeat(trace):
    trace = trace.reset_index(drop=True)
    loc = random.sample(range(len(trace)),1)[0]
    line = trace.iloc[loc:(loc+1)]
    trace = pd.concat([trace.iloc[0:(loc+1)], line, trace.iloc[(loc+1):] ]).reset_index(drop= True)
    return trace

def skip(trace):
    trace = trace.reset_index(drop=True)
    loc = random.sample(range(len(trace)),1)[0]
    trace.drop(index = trace.iloc[loc:(loc+1), :].index.tolist(),  inplace=True)
    trace = trace.reset_index(drop=True)
    return trace

def replace(trace, actlist):
    trace = trace.reset_index(drop=True)
    loc = random.sample(range(len(trace)),1)[0]
    actlist_out = [i for i in actlist if i != trace.Activity[loc]]
    trace.Activity[loc] = random.sample(actlist_out, 1)[0]
    return trace

def replace_attr(trace, A1list, A2list):
    trace = trace.reset_index(drop=True)
    attr = random.sample(['A1', 'A2'],1)[0]
    if attr == 'A1':
        attrlist =A1list
    else:
        attrlist =A2list

    loc = random.sample(range(len(trace)),1)[0]
    attrlist_out = [i for i in attrlist if i != trace[attr][loc]]
    if type(trace[attr][loc]) == str:
        trace[attr][loc] = random.sample(attrlist_out, 1)[0] 
    else:
        trace[attr][loc] = trace[attr][loc]*10
    return trace

def apply(trace, actlist, A1list, A2list, power = 1):
    pattern = ['repeat', 'skip', 'replace', 'replace_attr']
    sel_pattern = choices(pattern, k= power, weights=[0.3, 0.1, 0.3, 0.3])
    while sel_pattern.count('skip') >= len(set(trace.index)):
        sel_pattern[sel_pattern.index('skip')] = choices(['repeat', 'replace', 'replace_attr'], k= 1)[0]
        
    for i in sel_pattern:
        if i == 'repeat':
            trace = repeat(trace)
        elif i == 'skip':
            trace = skip(trace)
        elif i =='replace':
            trace = replace(trace,actlist)
        else:
            trace = replace_attr(trace, A1list, A2list)
            
    return trace


def apply2(trace, actlist, A1list, A2list, power = 1, pattern = 'skip'):
    sel_pattern = [pattern]*power
    
    if sel_pattern.count('skip') >= len(set(trace.index)):
        power = len(set(trace.index))-1
        sel_pattern = [pattern]*power

    for i in sel_pattern:
        if i == 'repeat':
            trace = repeat(trace)
        elif i == 'skip':
            trace = skip(trace)
        elif i =='replace':
            trace = replace(trace,actlist)
        else:
            trace = replace_attr(trace, A1list, A2list)
            
    return trace

print(name)
for p in p_list:
    for pt in pt_list:
        print('p =', p,  ', noise =', pt)

        if name == 'road_weibull4':

            df_train = data
            col_name = ['Case', 'Activity']
            i=0
            for col in df_train.columns[:2]:
                    df_train.rename(columns= {col: col_name[i]} , inplace = True)
                    i = i+1
            i=1
            for col in df_train.columns[2:]:
                    df_train.rename(columns= {col: 'A'+str(i)} , inplace = True)
                    i = i+1
            
            df_train.Case = df_train.Case.astype(str)

        else:

            if name == 'sepsis_weibull4':
                df_train = data[['Case.ID','Complete.Timestamp', 'Activity',  'CRP', 'Diagnose']] 
            elif name == 'helpdesk':
                df_train = data[['Case ID','Complete Timestamp', 'Activity',  'product', 'responsible_section']]
            elif name == 'BPIC12':
                df_train = data[['Case ID','Complete Timestamp', 'Activity',  'Resource', 'AMOUNT_REQ']]
                df_train.Resource = df_train.Resource.astype(str)
            elif name == 'hospital billing':
                df_train = data[['case_id','timestamp', 'activity',  'speciality', 'closecode']]
            elif name in ['credit', 'mccloud']:
                df_train = data[['Case ID','Complete Timestamp', 'Activity',  'Resource', 'resourceCost']]

            col_name = ['Case','Timestamp', 'Activity']
            i=0
            for col in df_train.columns[:3]:
                df_train.rename(columns= {col: col_name[i]} , inplace = True)
                i = i+1

            i=1
            for col in df_train.columns[3:]:
                df_train.rename(columns= {col: 'A'+str(i)} , inplace = True)
                i = i+1
                
            df_train.Case = df_train.Case.astype(str)
            df_train = df_train.sort_values(["Case", "Timestamp", "Activity"],ascending=[True, True, True]) # Reorder rows
        
        
        ###
        if (p ==1) & (pt == pt_list[0]):
            df_class = df_train.groupby(['Case'], as_index = False).apply(lambda x: str(x[["Activity", 'A1', 'A2']].values))
            df_class.columns = ['Case', 'Class']
            df_class.Class = df_class.Class.astype('category')
            cat_columns = df_class.select_dtypes(['category']).columns
            df_class[cat_columns] = df_class[cat_columns].apply(lambda x: x.cat.codes)

            df_class.to_csv("label_" + name + ".csv", index= False)


        c100 = list(df_train.Case.unique())
        c30 = random.sample(c100, round(len(c100)*0.3))

        df_test = df_train.loc[df_train.Case.isin(c30)]
        df_test.Case = df_test.Case.apply(lambda x: 'test_' + str(x))

        actlist = list(df_train.Activity.unique())
        A1list = list(df_train.A1.unique())
        A2list =  list(df_train.A2.unique())

        test = df_test.groupby('Case').apply(lambda x: apply2(x, actlist, A1list, A2list, power = p, pattern= pt)).reset_index(drop=True)
        
        df = pd.concat([df_train, test]).reset_index(drop=True)

        cols = df.columns[2:].tolist()
        for j in cols:
            df[j] = df[j].replace('', np.NaN)
            
        dir_home = os.getcwd()

        
        if name in ['helpdesk', 'hospital billing']:
            cc = ['A1', 'A2']
            nc = []
        elif name == 'sepsis_weibull4':
            cc = ['A2']
            nc = ['A1']
        elif name == 'road_weibull4':
            cc = cols
            nc = []
        elif name in ['BPIC12', 'credit', 'mccloud']:
            cc = ['A1']
            nc = ['A2']

        pipe = Pipeline(steps=[
                ("AggregateTransformer", AggregateTransformer(case_id_col = 'Case',
                                cat_cols = ['Activity'] + cc, 
                                num_cols = nc 
                                ))
                ])
        encoded_df = pipe.fit_transform(df)
        encoded_df.to_csv(dir_home+"/data_trans/" + name +"_aggregate_" + str(pt)+str(p) + ".csv", index= True)

        pipe = Pipeline(steps=[
                ("AggregateTransformer", AggregateTransformer(case_id_col = 'Case',
                                cat_cols = ['Activity'] + cc, 
                                num_cols = nc,
                                boolean= True))
                ])
        encoded_df = pipe.fit_transform(df)
        encoded_df.to_csv(dir_home+"/data_trans/" + name +"_bool_" + str(pt)+str(p) + ".csv", index= True)

        pipe = Pipeline(steps=[
                ("IndexBasedTransformer", IndexBasedTransformer(case_id_col = 'Case',
                                cat_cols = ['Activity'] + cc, 
                                num_cols = nc
                                ))
                ])
        encoded_df = pipe.fit_transform(df)
        encoded_df.to_csv(dir_home+"/data_trans/" + name +"_index_" + str(pt)+str(p) + ".csv", index= True)

        # pipe = Pipeline(steps=[
        #         ("AggregateNgramTransformer", AggregateNgramTransformer(case_id_col = 'Case',
        #                         act_col = 'Activity', n=2 , v= 0.7,
        #                         cat_cols = cc,
        #                         num_cols = nc
        #                         ))
        #         ])
        # encoded_df = pipe.fit_transform(df)

        # encoded_df.to_csv(dir_home+"/data_trans/" + name +"_aggngram_" + str(pt)+str(p) + ".csv", index= True)

        pipe = Pipeline(steps=[
                ("LastStateTransformer", LastStateTransformer(case_id_col = 'Case',
                                cat_cols = ['Activity'] + cc, 
                                num_cols = nc
                                ))
                ])
        encoded_df = pipe.fit_transform(df)
        encoded_df.to_csv(dir_home+"/data_trans/" + name +"_laststate_" + str(pt)+str(p) + ".csv", index= True)