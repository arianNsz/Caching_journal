# -*- coding: utf-8 -*-
"""
Created on Thu May 28 09:01:27 2020

@author: arian92
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
#%%

with open('DRL_e1_1000.pkl', 'rb') as f:
    e1_hits, e1_utilization, e1_avg_fresh, e1_rewards = pickle.load(f)
    
with open('DRL_e2_1000.pkl', 'rb') as f:
    e2_hits, e2_utilization, e2_avg_fresh, e2_rewards = pickle.load(f)
    
with open('DRL_paren_1000.pkl', 'rb') as f:
    p_hits, p_utilization, p_avg_fresh, p_rewards = pickle.load(f)

#%%

p_counter=0
reward=0
for i in range(999):
    if e1_hits[i] == 1:
        reward += 5
    else:
        if p_hits[p_counter] == 0:
            reward += 2
        else:
            reward -= 1
        p_counter += 1

    if e2_hits[i] == 1:
        reward += 5
    else:
        if p_hits[p_counter] == 0:
            reward += 2
        else:
            reward -= 1
        p_counter += 1
        
print(p_counter)    
#%%        
DRL_accu_reward=[]
for i in range(len(DRL_rewards)):
    DRL_accu_reward.append(sum(DRL_rewards[:i]))    
#%%
plt.title('Accumulated Reward')
plt.xlabel('time step')
plt.ylabel("accumulated reward")
plt.plot(DRL_accu_reward, label='DRL')
plt.plot(LRU_rewards, label='LRU')
plt.legend()
plt.show()
#%%
plt.title('Average Freshness')
DRL_mean = np.mean(DRL_avg_fresh)
LRU_mean= np.mean(LRU_avg_fresh)
plt.xlabel('time step')
plt.ylabel("avg freshness")
plt.plot(DRL_avg_fresh, zorder=1, label='DRL')
plt.plot(LRU_avg_fresh, zorder=1, label='LRU')
plt.hlines(DRL_mean, 0, 1000, colors='k', zorder=5, label='DRL_avg')
plt.hlines(LRU_mean, 0, 1000, colors='r', zorder=5, label='LRU_avg')
# plt.plot(np.mean(DRL_avg_fresh), label='DRL_mean')
# plt.plot(np.mean(LRU_avg_fresh), label='LRU_mean')
plt.legend()
plt.show()

#%%%
plt.title('Average Utilization')
DRL_util_mean = np.mean(DRL_utilization)
LRU_util_mean= np.mean(LRU_utilization)
plt.xlabel('time step')
plt.ylabel("avg Utilization")
plt.ylim(((0, 1.1)))
plt.plot(DRL_utilization, zorder=1, label='DRL')
plt.plot(LRU_utilization, zorder=1, label='LRU')
# plt.hlines(DRL_util_mean, 0, 1000, colors='k', zorder=5, label='DRL_avg')
# plt.hlines(LRU_util_mean, 0, 1000, colors='r', zorder=5, label='LRU_avg')
# plt.plot(np.mean(DRL_avg_fresh), label='DRL_mean')
# plt.plot(np.mean(LRU_avg_fresh), label='LRU_mean')
plt.legend()
plt.show()

#%%

labels = ['Cache Hit Rates (%)', 'Energy Consumption (%)']
DRL_hits = np.floor(DRL_hits)
LRU_energy = (1000 - LRU_hits)*0.0005
DRL_energy = (1000 - DRL_hits)*0.0005

DRL_energy /= LRU_energy
DRL_energy = np.around(DRL_energy, decimals=3)
LRU_energy /= LRU_energy

DRL_bar = [DRL_hits/1000, DRL_energy]
LRU_bar = [LRU_hits/1000, LRU_energy]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, DRL_bar, width, label='DRL')
rects2 = ax.bar(x + width/2, LRU_bar, width, label='LRU')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
# ax.set_title('Scores by group and gender')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()