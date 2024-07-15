

import os
from sys import maxsize

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd

dir_home = os.getcwd()
dir_ref = dir_home +"/ref"
import time 
import warnings
warnings.filterwarnings(action='ignore')


def mydist0(x, y):
    l = len(x)
    d_x = x[:(l-1)]
    d_y = y[:(l-1)]
    d = np.linalg.norm(d_x-d_y)
    
    return d   # for minimizing problem

def mydist2(x, y):
    l = len(x)
    d_x = x[:(l-1)]
    d_y = y[:(l-1)]
    p_y = y[(l-1):][0]

    d = np.linalg.norm(d_x-d_y)
    
    return 0.8*d + 0.2*p_y  # for minimizing problem

def mydist4(x, y):
    l = len(x)
    d_x = x[:(l-1)]
    d_y = y[:(l-1)]
    p_y = y[(l-1):][0]

    d = np.linalg.norm(d_x-d_y)
    
    return 0.6*d + 0.4*p_y  # for minimizing problem


def mydist6(x, y):
    l = len(x)
    d_x = x[:(l-1)]
    d_y = y[:(l-1)]
    p_y = y[(l-1):][0]

    d = np.linalg.norm(d_x-d_y)
    
    return 0.4*d + 0.6*p_y  # for minimizing problem

def mydist8(x, y):
    l = len(x)
    d_x = x[:(l-1)]
    d_y = y[:(l-1)]
    p_y = y[(l-1):][0]

    d = np.linalg.norm(d_x-d_y)
    
    return 0.2*d + 0.8*p_y  # for minimizing problem

def mydist10(x, y):
    l = len(x)
    p_y = y[(l-1):][0]
    
    return p_y   # for minimizing problem



def async_move(event):
  return 1

def sync_move_standard(event1, event2):
  if event1["label"] != event2["label"]:
    return maxsize # infinity in definition
  else:
    data1 = event1["data"]
    data2 = event2["data"]
    keys = set(list(data1.keys()) + list(data2.keys()))
    penalty = 0
    for k in keys:
      if (k in data1 and not k in data2) or (k in data2 and not k in data1):
        penalty += 1
      else:
        penalty += (0 if data1[k] == data2[k] else 1)
    return penalty

def distance_faster(trace1, trace2, sync_move):
  # initialize to 0
  delta = [ [0 for j in range(0,len(trace2)+1)] for i in range(0,len(trace1)+1)]

  for i in range(0,len(trace1)+1):
    for j in range(0,len(trace2)+1):
      if i == 0 and j == 0:
        continue # delta[i][j] = 0
      elif i == 0:
        delta[i][j] = async_move(trace2[j-1]) +  delta[i][j-1]
      elif j == 0:
        delta[i][j] = async_move(trace1[j-1])  +  delta[i-1][j]
      else:
        delta[i][j] = min(
                delta[i-1][j-1] + sync_move(trace1[i-1], trace2[j-1]),
                delta[i-1][j] + async_move(trace1[i-1]),
                delta[i][j-1] + async_move(trace2[j-1]))
  return delta[len(trace1)][len(trace2)]

def distance_standard(trace1, trace2):
  return distance_faster(trace1, trace2, sync_move_standard)


name_list = ['credit','mccloud'  , 'road_weibull4', 'helpdesk', 'hospital billing']
split = 0.5
encoding = ['aggregate', 'bool',  'index',  'laststate', 'aggngram']
metric =  [ 'cosine', 'euclidean', 'manhattan']
klist = [1, 15, 30]
plist=[1,3,5]





for name in name_list:
    result = pd.DataFrame(columns=['data', 'encoding_method' , 'distance_metric', 'precision', 'd_mean', 'd_sd','k', 'p',
                               'time'])
    ref =  pd.read_csv("label_" + name + ".csv")
    ref.Case  = ref.Case.astype(str)

    #### Ground start

    if name == 'helpdesk':
        ground = pd.read_csv(dir_home + '/org_log/' + name + ".csv")
        ground_train = ground[['Case ID','Complete Timestamp', 'Activity',  'product', 'responsible_section']]  # helpdesk

    if name in ['credit', 'mccloud']:
        if name == 'credit':
            name2= 'credit-card_new'
        else:
            name2 = 'mccloud_new'
        ground = pd.read_csv(dir_home + '/org_log/' + name2 + ".csv")
        ground_train = ground[['Case','Timestamp', 'Activity',  'Resource', 'System']] #credit mccloud
        ground_train['Case'] = ground_train['Case'].astype(str) #credit mccloud

    if name == 'hospital billing':
        ground = pd.read_csv(dir_home + '/org_log/' + name + ".csv")
        ground_train = ground[['case_id','timestamp', 'activity',  'speciality', 'closecode']]  # hospital billing

    if name == "helpdesk":
        ground = pd.read_csv(dir_home + '/org_log/' + name + ".csv")
        ground_train = ground

    ### 
    if name == 'road_weibull4':
        f = lambda x: [{"label": y['Activity'], 
            "data": { 'A'+str(v) : y['A'+str(v)] for v in range(1,133) }  } for index, y in x.iterrows()]
        col_name = ['Case', 'Activity']
        i=0
        for col in ground_train.columns[:2]:
            ground_train.rename(columns= {col: col_name[i]} , inplace = True)
            i = i+1
        i=1
        for col in ground_train.columns[2:]:
            ground_train.rename(columns= {col: 'A'+str(i)} , inplace = True)
            i = i+1
            
        ground_train.Case = ground_train.Case.astype(str)
        ground_trace = ground_train.groupby('Case')[ground_train.columns[1:].tolist()].apply(f).reset_index()

    else:
        f = lambda x: [{"label": y['Activity'], 
            "data": { 'A'+str(v) : y['A'+str(v)] for v in range(1,3) }  } for index, y in x.iterrows()]
        col_name = ['Case','Timestamp', 'Activity']
        i=0
        for col in ground_train.columns[:3]:
            ground_train.rename(columns= {col: col_name[i]} , inplace = True)
            i = i+1

        i=1
        for col in ground_train.columns[3:]:
            ground_train.rename(columns= {col: 'A'+str(i)} , inplace = True)
            i = i+1
            
        ground_train.Case = ground_train.Case.astype(str)
        ground_train = ground_train.sort_values(["Case", "Timestamp", "Activity"],ascending=[True, True, True]) # Reorder rows
        ground_trace = ground_train.groupby('Case')[['Activity', 'A1', 'A2']].apply(f).reset_index()

    ground_trace.index = ground_trace['Case']
    ground_trace = ground_trace.drop('Case', axis=1)



    #### Ground end


    unique_case = ref.groupby('Class').apply(lambda x: x['Case'].tolist()[0]).tolist()
    print(name)
    for p in plist:
        print("p:",p)
        for k in klist:
            print("k:", k)
            for e in encoding:
                print(e)
                data = pd.read_csv(dir_home + '/data_trans/' + name + "_" + e + "_" + str(p)  +".csv") #for multinoise
                # data = pd.read_csv(dir_home + '/data_trans/' + name + "_" + e + "_" + str(pt)+str(p)  +".csv") #for partialnoise

                data = data.fillna(0)
                data.Case =  data.Case.astype(str)
                
                if name == 'BPIC12':
                    A2cols=  [dc for dc in data.columns if 'A2' in dc]
                    data[A2cols] = (data[A2cols]-data[A2cols].mean())/data[A2cols].std()
                    data = data.fillna(0)

                if name in ['credit-card_new','mccloud_new' ]:
                    scaler = MinMaxScaler()
                    if encoding in ['aggregate']:
                        Actcols=  [dc for dc in data.columns if 'Activity' in dc]
                        data[Actcols] = scaler.fit_transform( data[Actcols])
                        data = data.fillna(0)             
                    A2cols=  [dc for dc in data.columns if 'A2' in dc]
                    if len(A2cols)>0:
                        data[A2cols] = scaler.fit_transform( data[A2cols])
                    data = data.fillna(0)

                train = data[~data['Case'].astype(str).str.contains('test\_', na=False)].reset_index(drop=True)
                test = data[data['Case'].astype(str).str.contains('test\_', na=False)].reset_index(drop=True)

                train_unique = train[train.Case.isin(unique_case)]
                train_unique.index = train_unique['Case']
                train_unique = train_unique.drop( 'Case', axis=1)
                
                
                train.index = train['Case']
                train = train.drop( 'Case', axis=1)
                test.index = test['Case']
                test = test.drop( 'Case', axis=1)
                
                if e in ["ngram", "aggngram"]:
                    train_trace = train.filter(regex='\|')
                    test_trace = test.filter(regex='\|')
                    
                    train_unique_trace = train_unique.filter(regex='\|')
                else:
                    train_trace = train.filter(regex='Activity_')
                    test_trace = test.filter(regex='Activity_') 
                    
                    train_unique_trace = train_unique.filter(regex='Activity_')   
                    
                train_attr = train.drop(train_trace.columns , axis =1)
                test_attr = test.drop(test_trace.columns, axis = 1)

                train_unique_attr = train_unique.drop(train_unique_trace.columns , axis =1)

                w_a = split*np.repeat(1, len(train_trace.columns))/len(train_trace.columns)
                w_a = w_a.tolist()

                w_b = pd.DataFrame([str.split(c, "_")[0] for c in train_attr.columns], columns=['key'] )
                w_b['weight'] = 1
                w_b['weight2'] = w_b.groupby('key')['weight'].cumsum()
                w_b['max'] = w_b.groupby('key')['weight2'].transform(max)
                w_b['weight3'] = w_b['weight']/w_b['max']

                w_b = w_b['weight3'].tolist()
                w_b = [(1-split)*w/sum(w_b) for w in w_b]
                
                loc_a = [ train.columns.tolist().index(ttt)  for ttt in train_trace.columns]
                loc_b = [ train.columns.tolist().index(ttt)  for ttt in train_attr.columns]
                customized_weights = np.repeat(0.0, len(train.columns))
                customized_weights[loc_a] = w_a
                customized_weights[loc_b] = w_b
                
                train = train.apply(lambda x: x*customized_weights, axis= 1)
                test = test.apply(lambda x: x*customized_weights, axis= 1)
                
                train_unique = train_unique.apply(lambda x: x*customized_weights, axis= 1)
                dict_rate = {0:mydist0, 2:mydist2, 4:mydist4, 6:mydist6, 8:mydist8, 10:mydist10}


                for d in metric:                    
                    trigger = 1 
                    acc_all = []
                    time_all =[]
                    for rate in [0, 2, 4, 6, 8, 10]:
                        encoded_train = train.values.tolist()
                        encoded_test = test.values.tolist()
                        time_start_align = time.time()

                        if (rate !=0) and (trigger == 1):
                            
                            nn = NearestNeighbors(n_neighbors = k, algorithm='auto', metric = d, metric_params = None).fit(np.array(encoded_train) )  # np.array(encoded_train) 
                            temp = np.mean(nn.kneighbors()[0])
                            if temp == 0:
                                temp = 0.00001
                            clustering = DBSCAN(eps= temp, min_samples=1, metric= d ).fit(np.array(encoded_train))
                            # clustering = KMeans(n_clusters= 1000 ).fit(np.array(encoded_train) ) # int(np.floor(len(train)*rate))
                            cluster, freq = np.unique(clustering.labels_, return_counts=True)
                            trace_cluster = pd.DataFrame({'cluster': cluster, 'prob': freq/sum(freq)})

                            train_prob = pd.DataFrame({'case': train.index , 'cluster': clustering.labels_})
                            # print(train_prob.describe())
                            train_prob = train_prob.merge(trace_cluster, how='left', on= 'cluster')
                            train_prob['sim'] = train_prob['prob'].apply(lambda x: 1/(1+x))
                            train_prob.sim = train_prob.sim*np.max(nn.kneighbors()[0])/(train_prob.sim.max())

                            train['prob'] = train_prob.sim.tolist()
                            test['prob'] = 0
                            
                            train_prob_unique = train_prob[train_prob.case.isin(unique_case)]
                            train_unique['prob'] = train_prob_unique.sim.tolist()

                            encoded_test = test.values.tolist()
                            trigger = 0


                        encoded_train = train_unique.values.tolist()
                        nn = NearestNeighbors(n_neighbors = k, algorithm='ball_tree', metric = 'pyfunc', metric_params={"func":dict_rate[rate]}).fit(np.array(encoded_train) ) 
                        dists, idxs = nn.kneighbors(np.array(encoded_test) )

                        time_finish_align = time.time()

                        predict = list()

                        for i in range(0, len(test_trace)):
                            predict.append(  train_unique.index[idxs[i]] )
                            

                        acc_sum = 0 
                        dist_mean = 0
                        dist_sd = 0

                        for l in range(len(test)):  # change
                            ground = test.index[l].split('test_')[1] 
                            ground_class = ref[ref.Case == str(ground)].Class.values[0]
                            ground_trace_opt = ground_trace.loc[str(ground)][0]
                            
                            acc = 0
                            dist = []
                            for pr in predict[l]:
                                predict_class = ref[ref.Case == str(pr)].Class.values[0]
                                ground_trace_pred = ground_trace.loc[str(pr)][0]

                                dist = dist+ [distance_standard(ground_trace_opt, ground_trace_pred)]
                                if ground_class == predict_class:
                                    acc = 1

                            dist_mean = dist_mean + np.mean(dist)
                            dist_sd = dist_sd + np.std(dist)

                            acc_sum = acc_sum + acc
                            
                        acc = acc_sum/len(test)
                        acc_all = acc_all + [acc]
                        d_mean = dist_mean/len(test)
                        d_sd = dist_sd/len(test)
                        dur = (time_finish_align - time_start_align)
                        time_all = time_all + [dur]

                    result.loc[len(result)+1] = [name, e, d, str(acc_all), d_mean, d_sd, k, p,
                                            str(time_all)]
                
                
    result.to_csv(dir_home+"/result/result_prob_" + name +"_" + str(split)  + ".csv", index= False) 
