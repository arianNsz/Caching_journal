# -*- coding: utf-8 -*-
"""
Created on Wed May 13 20:41:47 2020

@author: arian92
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import gc
import matplotlib.pyplot as plt

ones = np.ones(shape=(136000,))
others= np.random.randint(1,41, size=39000) 
bs1= np.concatenate((ones, others))
del ones, others
gc.collect()

#%%
pop1 =[1, 2, 3, 4, 5, 6, 7, 8]
commons= [10, 11, 12, 13, 14, 15, 16, 17]
pop2= [40, 39, 38, 37, 36, 35, 34, 33]

#%%
bs1[:10000]= pop1[7]
bs1[10000:20000]= pop1[6]
bs1[20000:30000]= pop1[5]
bs1[30000:40000]= pop1[4]
bs1[40000:47000]= pop1[3]
bs1[470000:54000]= pop1[2]
bs1[54000:61000]= pop1[1]
bs1[61000:68000]= pop1[0]
bs1[68000:78000]= commons[7]
bs1[78000:88000]= commons[6]
bs1[88000:98000]= commons[5]
bs1[98000:108000]= commons[4]
bs1[108000:115000]= commons[3]
bs1[115000:122000]= commons[2]
bs1[122000:129000]= commons[1]
bs1[129000:136000]= commons[0]
bs1= bs1.astype(int)
#%%
ones = np.ones(shape=(136000,))
others= np.random.randint(1,41, size=39000) 
bs2= np.concatenate((ones, others))
del ones, others
gc.collect()

#%%
bs2[:10000]= pop2[7]
bs2[10000:20000]= pop2[6]
bs2[20000:30000]= pop2[5]
bs2[30000:40000]= pop2[4]
bs2[40000:47000]= pop2[3]
bs2[470000:54000]= pop2[2]
bs2[54000:61000]= pop2[1]
bs2[61000:68000]= pop2[0]
bs2[68000:78000]= commons[7]
bs2[78000:88000]= commons[6]
bs2[88000:98000]= commons[5]
bs2[98000:108000]= commons[4]
bs2[108000:115000]= commons[3]
bs2[115000:122000]= commons[2]
bs2[122000:129000]= commons[1]
bs2[129000:136000]= commons[0]
bs2= bs2.astype(int)

#%%
lt = np.ones(shape=(40,), dtype=int)
lt[0], lt[1], lt[2], lt[3], lt[9], lt[10], lt[11], lt[12], lt[39], lt[38], lt[37], lt[36] = 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30
lt[4], lt[5], lt[6], lt[7], lt[13], lt[14], lt[15], lt[16], lt[35], lt[34], lt[32], lt[31] = 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2

#%%
for x in range(len(lt)):
    if lt[x]==1:
        lt[x]= np.random.randint(1, 14)        
        
#%%
bs1 = np.array(bs1).astype(int)
lifetimes1 = lt[bs1-1]

#%%
bs2 = np.array(bs2).astype(int)
lifetimes2 = lt[bs2-1]
#%%

requests1= np.concatenate((bs1.reshape(-1,1), lifetimes1.reshape(-1,1)), axis=1)
#%%
requests2= np.concatenate((bs2.reshape(-1,1), lifetimes2.reshape(-1,1)), axis=1)
#%%
del bs1, bs2, lifetimes1, lifetimes2, lt, pop1, pop2, commons
gc.collect()

#%%    
final_requests= np.concatenate((requests1, requests2), axis=0)


#%%
final_requests=[]
np.random.shuffle(requests1)
np.random.shuffle(requests2)
#%%
for i in range(len(requests1)):
    final_requests.append(requests1[i,:])
    final_requests.append(requests2[i,:])
    
#%%
final_requests = pd.DataFrame(final_requests)
final_requests.columns= ['Requested File', 'Lifetime']
final_requests.to_csv('may27.csv', index= False)




