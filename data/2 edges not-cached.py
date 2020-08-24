# -*- coding: utf-8 -*-
"""
Created on Thu May 28 09:01:27 2020

@author: arian92
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#%%

with open('DRL_e1.pkl', 'rb') as f:
    e1_hits, e1_utilization, e1_avg_fresh, e1_rewards = pickle.load(f)
    

with open('DRL_e2.pkl', 'rb') as f:
    e2_hits, e2_utilization, e2_avg_fresh, e2_rewards = pickle.load(f)
#%%
e1_req = pd.read_csv('./data/edge1.csv')
e2_req = pd.read_csv('./data/edge2.csv')
not_cached=[]
for i in range(99998):
    if e1_hits[i] == 0:
        not_cached.append([i, e1_req.loc[i].values[0], e1_req.loc[i].values[1]])
    if e2_hits[i]==0 :
        not_cached.append([i, e2_req.loc[i].values[0], e2_req.loc[i].values[1]])

not_cached = pd.DataFrame(not_cached, columns=['time_stamp', 'Requested File', 'Lifetime'])
not_cached.to_csv('./data/parent.csv', index=False)