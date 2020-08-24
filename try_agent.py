import gym
import time

from stable_baselines.common.policies import MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
# from stable_baselines.common.schedules import LinearSchedule as lr
# from stable_baselines import A2C
from stable_baselines import PPO2
from pandas.plotting import register_matplotlib_converters
from stable_baselines.common.vec_env import DummyVecEnv
register_matplotlib_converters()

from env.try_env import cache_env

#%%
import pandas as pd
# df = pd.read_csv('./data/may27.csv')
# df = pd.read_csv('./data/edge1.csv')

# df = pd.read_csv('./data/edge2.csv')

df = pd.read_csv('./data/parent.csv')
df= df.drop(columns=['time_stamp'])
#%%

#env = cache_env(df)

env = DummyVecEnv([lambda: cache_env(df)])

#%%

# model = A2C('MlpPolicy', env, gamma= 0.99 , n_steps= 14, learning_rate=0.0008,  tensorboard_log="./logs/")

model= PPO2('MlpPolicy', env, n_steps=12, learning_rate=0.00075, nminibatches=4)#, tensorboard_log="./logs/")
# model = A2C('MlpPolicy', env, gamma= 0.99 , n_steps= 14, learning_rate=0.001, lr_schedule='linear', tensorboard_log="./logs/")
# A2C('MlpLstmPolicy', env, gamma= 0.99 , n_steps= 20, learning_rate=0.001, lr_schedule='linear', tensorboard_log="./logs/")
#A2C('MlpPolicy', env, gamma= 0.99 , n_steps= 14, learning_rate=0.001, lr_schedule='linear')
#A2C('MlpPolicy', env, gamma= 0.9, n_steps= 18, learning_rate=0.00095)
#A2C('MlpLstmPolicy', env, gamma= 0.9, n_steps= 18, learning_rate=0.01, alpha=0.9, epsilon=1e-05, lr_schedule='linear')
#model.load('A2C_cache.zip')
#model.load(
# model.load('A2C_new_env')
#%%
start_time = time.time()

model.learn(total_timesteps=80000)

print("--- %s seconds ---" % (time.time() - start_time))

#%%
# model.save('e1')
# model.save('e2')
model.save('parent')
#%%
#model = A2C('MlpPolicy', env)
#model.load('A2C_cache.zip')

#%%
# r=[]
# rewards=[]

# obs = env.reset()
# for i in range(500):
#     print('')
#     print('')
#     print('----------------------------------------------')
#     print(obs)
#     print("")
#     action, _states = model.predict(obs, deterministic= True)
#     print(f'Action : {action}')
#     print("")
#     obs, rewards, done, info = env.step(action)
#     print(f'Rewards : {rewards}')
#     # print("")
#     r.append(rewards)
#     env.render()
    
    # print(sum(r))

#import matplotlib.pyplot as plt
#plt.plot(r)