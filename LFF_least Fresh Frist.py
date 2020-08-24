# -*- coding: utf-8 -*-
"""
Created on Fri May  1 11:16:14 2020

@author: Arian Nsz
"""

import numpy as np
import pandas as pd
import gc

# data = pd.read_csv('./data/may27.csv')
data = pd.read_csv('./data/challenging_popularity.csv')

""" take a look at the shape"""
CAP= 6
mem_status= np.zeros(shape=(6, CAP))
freshness= np.zeros(shape=(6, CAP))
total_mem = 3 * CAP
avg_fresh= 0
current_step = 0
done = False
MAX_STEPS = 1000
MIN_COMMUN= 5
MID_COMMUN= 2
MAX_COMMUN= -1
fresh_cost_weight= 1
reward=0.0
greedy_punishment = 0
mem_slots= [0, 2, 4]
fresh_slots= [1, 3, 5]
hits=[]
rewards=[]
episodes_avg_fresh=[]
utilization = []
line1= np.zeros(shape=(1,CAP))



while current_step<= MAX_STEPS:
    print(f'Step: {current_step}')
    request = data.loc[current_step].values    
    # update the table (considering life-time):
    # life = current_step - mem_status[:, [1,4]]
    # expire = life > mem_status[:, [2,5]]
    
    """" Removing the expired content from cache """
    expired = np.where(mem_status[fresh_slots, :] >= 1)
    mem_status[expired[0]*2, expired[1]] = 0 
    mem_status[expired[0]*2 + 1, expired[1]] = 0
    freshness[expired[0]*2, expired[1]] = 0 
    freshness[expired[0]*2 + 1, expired[1]] = 0
    
    edge1_has = np.where(mem_status[0, :] == request[0])
    edge2_has = np.where(mem_status[2, :] == request[0])
    parent_has = np.where(mem_status[4:, :] == request[0])
    e1=0
    e2=0
    p_has=0
    if (edge1_has[0].shape != (0,)):
        e1 = 1
    if (edge2_has[0].shape != (0,)):
        e2 = 1
    if (parent_has[0].shape != (0,)):
        p_has = 1
        
    do_cache = True
    
    # under_utilized = np.where(mem_status[mem_slots, :] == 0)
    # under_utilized = max(under_utilized[0].shape[0], under_utilized[1].shape[0])
    # under_utilized /= total_mem
    # under_utilized = np.clip(under_utilized, 0, 3)
    # self.reward -= under_utilized
    # self.reward = np.clip(self.reward, -21, 100)
    not_utilized = (mem_status[mem_slots, :]==0).sum()
    reward -= not_utilized
    utilization.append((total_mem - not_utilized)/ total_mem)
    
    for i in range(3):
        a12 = current_step - freshness[2*i,:]
        b12 = freshness[2*i +1,:]
        c12= np.zeros_like(b12)
        for j in range(len(b12)):
            if b12[j]>=1:
                c12[j]= a12[j]/ b12[j]
                # c12 = c12.reshape(CAP,1)
        mem_status[2*i+1, :] = c12
        # self.avg_fresh += sum(c12)/CAP
        

    # reward = 0.0    
    which_BS = current_step % 2
    if which_BS==0:
         if e1==1:
            do_cache= False
            reward += e1 * MIN_COMMUN
            mem_status[0:2, :edge1_has[0][0]] = np.roll(mem_status[0:2, :edge1_has[0][0]], 1, axis=1)
            freshness[0:2, :edge1_has[0][0]] = np.roll(freshness[0:2, :edge1_has[0][0]], 1, axis=1)
            reward -= mem_status[1, edge1_has[0][0]] #substract freshness cost
         elif p_has == 1:
            do_cache= False
            reward += p_has * MID_COMMUN
            mem_status[4:, :parent_has[0][0]] = np.roll(mem_status[2*i: 2*i+1, :parent_has[0][0]], 1, axis=1)
            freshness[4:, :parent_has[0][0]] = np.roll(freshness[4: 2*i+1, :parent_has[0][0]], 1, axis=1)
            reward -= mem_status[5, parent_has[0][0]] #substract freshness cost
         else:
            reward += MAX_COMMUN
    elif which_BS==1:
         if e2==1:
            do_cache= False
            reward += e2 * MIN_COMMUN
            mem_status[2:4, :edge2_has[0][0]] = np.roll(mem_status[2:4, :edge2_has[0][0]], 1, axis=1)
            freshness[2:4, :edge2_has[0][0]] = np.roll(freshness[2:4, :edge2_has[0][0]], 1, axis=1)
            reward -= mem_status[3, edge2_has[0][0]] #substract freshness cost
         elif p_has == 1:
            do_cache= False
            reward += p_has * MID_COMMUN
            mem_status[4:, :parent_has[0][0]] = np.roll(mem_status[2*i: 2*i+1, :parent_has[0][0]], 1, axis=1)
            freshness[4:, :parent_has[0][0]] = np.roll(freshness[4: 2*i+1, :parent_has[0][0]], 1, axis=1)
            reward -= mem_status[5, parent_has[0][0]] #substract freshness cost
         else:
            reward += MAX_COMMUN
            
    if do_cache:
        hits.append(0)
        # first look for empty spaces in the memory
        empty_e = np.where(mem_status[0, :]== 0)
        empty_e2 = np.where(mem_status[2, :]== 0)
        empty_p = np.where(mem_status[4, :]== 0)
        if empty_e[0].shape != (0,) and which_BS==0:
            mem_status[0, empty_e[0][-1]] = request[0]
            mem_status[1, empty_e[0][-1]] = 0
            freshness[0, empty_e[0][-1]] = current_step
            freshness[1, empty_e[0][-1]] = request[1]
        elif which_BS==0:
            e1_least_fresh= np.argsort(mem_status[1,:])[-1]
            mem_status[0, e1_least_fresh] = request[0]
            mem_status[1, e1_least_fresh] = 0
            freshness[0, e1_least_fresh] = current_step
            freshness[1, e1_least_fresh] = request[1]
            
        if empty_e2[0].shape != (0,) and which_BS==1:
            mem_status[2, empty_e2[0][-1]] = request[0]
            mem_status[3, empty_e2[0][-1]] = 0
            freshness[2, empty_e2[0][-1]] = current_step
            freshness[3, empty_e2[0][-1]] = request[1]
        elif which_BS==1:
            e2_least_fresh= np.argsort(mem_status[3,:])[-1]
            mem_status[2, e2_least_fresh] = request[0]
            mem_status[3, e2_least_fresh] = 0
            freshness[2, e2_least_fresh] = current_step
            freshness[3, e2_least_fresh] = request[1]
            
        if empty_p[0].shape != (0,) :
            mem_status[4, empty_p[0][-1]] = request[0]
            mem_status[5, empty_p[0][-1]] = 0
            freshness[4, empty_p[0][-1]] = current_step
            freshness[5, empty_p[0][-1]] = request[1]
        else:
            p_least_fresh = np.argsort(mem_status[5,:])[-1]
            mem_status[4, p_least_fresh] = request[0]
            mem_status[5, p_least_fresh] = 0
            freshness[4, p_least_fresh] = request[0]
            freshness[5, p_least_fresh] = 0
    else:
        hits.append(1)
    
    rewards.append(reward)
    print(f'Next Requests: {data[current_step: current_step + 1].values}')
    print(f'Memory Status: {mem_status}')
    print(f'Reward: {reward}')
    current_step += 1
#%%
import matplotlib.pyplot as plt
plt.title('accumulated rewards LFF')
plt.plot(rewards)
plt.show()

# plt.title('average freshness in each step')
# avg_freshness = np.array(avg_freshness)
# avg_freshness /= 3
# plt.plot(avg_freshness)
# plt.show()
# plt.title('average utilization in each step (higher is better)')
# utilization = np.array(utilization)
# print(f'Avg Utilization: {np.mean(utilization)}')
# plt.plot(utilization)
# plt.show()
    
# #%%
# import pickle
# with open('LFF.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
#     pickle.dump([sum(hits), utilization, avg_freshness, rewards], f)