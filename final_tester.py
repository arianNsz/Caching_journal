# -*- coding: utf-8 -*-
"""
Created on Fri May 22 09:50:54 2020

@author: arian92
"""


import gym

from stable_baselines.common.policies import MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
# from stable_baselines.common.schedules import LinearSchedule as lr
from stable_baselines import A2C
from stable_baselines import PPO2
from pandas.plotting import register_matplotlib_converters
from stable_baselines.common.vec_env import DummyVecEnv
register_matplotlib_converters()

import pandas as pd
import matplotlib.pyplot as plt


from env.test_env import cache_env

# df = pd.read_csv('./data/edge1.csv')
# df = pd.read_csv('./data/edge2.csv')
df = pd.read_csv('./data/parent.csv')
df= df.drop(columns=['time_stamp'])
#%%

# df = pd.read_csv('./data/may27.csv')


env = DummyVecEnv([lambda: cache_env(df)])
# model = A2C('MlpPolicy', env, gamma= 0.99 , n_steps= 20, learning_rate=0.003, alpha=0.98, epsilon=2e-05, lr_schedule='linear')


# model = PPO2.load('e1.zip')
# model = PPO2.load('e2.zip')
model = PPO2.load('parent.zip')


# model = A2C.load('A2C_RNN.zip')
# model = A2C.load('final_version.zip')
# model = A2C.load('A2C_cache_hit.zip')
#model = PPO2.load('PPO2_real_final.zip')

r=[]
rewards=[]

obs = env.reset()
for i in range(1368):
    print('')
    print('')
    print('----------------------------------------------')
    # print(obs)
    print("")
    action, _states = model.predict(obs, deterministic= True)
    # print(f'Action : {action}')
    # print("")
    obs, rewards, done, info = env.step(action)
    # print(f'Rewards : {rewards}')
    # print("")
    r.append(rewards)
    env.render()
    
    print('sum of reward is ' + str(sum(r)))
    #%%
r2=[]
for i in range(len(r)):
    r2.append(sum(r[:i]))
plt.title('accumulated rewards')
plt.plot(r2)
plt.show()