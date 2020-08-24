import numpy as np
import pandas as pd
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import gc
import matplotlib.pyplot as plt
class cache_env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, requests_df):
        super(cache_env, self).__init__()
        """   """
        self.df = requests_df
        self.mem_status= np.zeros(shape=(6,6))
        self.freshness= np.zeros(shape=(6,6))
        self.avg_fresh= 0
        self.current_step = 0
        self.done = False
        self.MEM_SIZE = 6
        self.MAX_STEPS = 200000
        self.MIN_COMMUN= 5
        self.MID_COMMUN= 2
        self.MAX_COMMUN= -1
        self.fresh_cost_weight= 1
        self.reward=0.0
        self.greedy_punishment = 0
        self.mem_slots= [0, 2, 4]
        self.fresh_slots= [1, 3, 5]
        self.episodes=[]
        self.episodes_avg_fresh=[]
        self.line1= np.zeros(shape=(1,6))


        """
        Defining Observation and Action Spaces
        """
        self.action_space = spaces.Discrete(8)
        self.observation_space= spaces.Box(
                                           low=0,
                                           high=40,
                                           shape= (7,6),
                                           dtype= np.float16
                                          )
        print(self.observation_space.sample())
        
    def _next_observation(self):
        temp = self.df.loc[self.current_step].values
        edge1_has = np.where(self.mem_status[0, :] == temp[0])
        edge2_has = np.where(self.mem_status[2, :] == temp[0])
        p_has = np.where(self.mem_status[4, :] ==  temp[0])

        if (edge1_has[0].shape != (0,)):
            edge1_has = 1
        else:
            edge1_has = 0
        if (edge2_has[0].shape != (0,)):
            edge2_has = 1
        else:
            edge2_has = 0
        if (p_has[0].shape != (0,)):
            p_has = 1 
        else:
            p_has = 0
            
        self.which_BS = self.current_step % 2
        self.line1 = np.array([temp[0], temp[1], edge1_has, 
                          edge2_has, p_has, self.which_BS]).reshape(1,6)
        obs = np.concatenate((self.line1, self.mem_status), axis=0)
        obs = np.array(obs).reshape(7,6)
        return obs
    
    def step(self, action):
        under_utilized = 0
        """ Calculating the reward"""
        self.reward = 0
        self.avg_fresh=0
        if self.current_step==0:
            self.episodes=[]
        
        request = self.df.loc[self.current_step].values
        edge1_has = np.where(self.mem_status[0, :] == request[0])
        edge2_has = np.where(self.mem_status[2, :] == request[0])
        parent_has = np.where(self.mem_status[4, :] == request[0])
        e1=0
        e2=0
        p_has=0
        if (edge1_has[0].shape != (0,)):
            e1 = 1
        if (edge2_has[0].shape != (0,)):
            e2 = 1
        if (parent_has[0].shape != (0,)):
            p_has = 1
            
        for i in range(3):
            a12 = self.current_step - self.freshness[2*i,:]
            b12 = self.freshness[2*i +1,:]
            c12= np.zeros_like(b12)
            for j in range(len(b12)):
                if b12[j]>=1:
                    c12[j]= a12[j]/ b12[j]
                    # c12 = c12.reshape(6,1)
            self.mem_status[2*i+1, :] = c12
            # self.avg_fresh += sum(c12)/6
        
      #  calculate the reward for each case       
        if self.which_BS == 0:
            # self.reward = e1 * self.MIN_COMMUN + p_has * self.MID_COMMUN
            if e1==1:
                self.reward = e1 * self.MIN_COMMUN
                self.mem_status[0:2, :edge1_has[0][0]] = np.roll(self.mem_status[0:2, :edge1_has[0][0]], 1, axis=1)
                self.freshness[0:2, :edge1_has[0][0]] = np.roll(self.freshness[0:2, :edge1_has[0][0]], 1, axis=1)
                self.reward -= self.mem_status[1, edge1_has[0][0]] #substract freshness cost
                if action!=0:
                    self.reward += self.greedy_punishment * (e1 + p_has)
            elif p_has == 1:
                self.reward = p_has * self.MID_COMMUN
                self.mem_status[4:, :parent_has[0][0]] = np.roll(self.mem_status[2*i: 2*i+1, :parent_has[0][0]], 1, axis=1)
                self.freshness[4:, :parent_has[0][0]] = np.roll(self.freshness[4: 2*i+1, :parent_has[0][0]], 1, axis=1)
                self.reward -= self.mem_status[5, parent_has[0][0]] #substract freshness cost
                if action!=0:
                    self.reward += self.greedy_punishment
            else:
                self.reward += self.MAX_COMMUN
                
            # self.reward -= self.avg_fresh
            
        elif self.which_BS ==1:
            # self.reward = e2 * self.MIN_COMMUN + p_has * self.MID_COMMUN
            if e2==1:
                self.reward = e2 * self.MIN_COMMUN
                self.mem_status[2:4, :edge2_has[0][0]] = np.roll(self.mem_status[2:4, :edge2_has[0][0]], 1, axis=1)
                self.freshness[2:4, :edge2_has[0][0]] = np.roll(self.freshness[2:4, :edge2_has[0][0]], 1, axis=1)
                self.reward -= self.mem_status[3, edge2_has[0][0]] #substract freshness cost
                if action!=0:
                    self.reward += self.greedy_punishment * (e2 + p_has)
            elif p_has == 1:
                self.reward = p_has * self.MID_COMMUN
                self.mem_status[4:, :parent_has[0][0]] = np.roll(self.mem_status[4: , :parent_has[0][0]], 1, axis=1)
                self.freshness[4:, :parent_has[0][0]] = np.roll(self.freshness[4: , :parent_has[0][0]], 1, axis=1)
                self.reward -= self.mem_status[5, parent_has[0][0]] #substract freshness cost
                if action!=0:
                    self.reward += self.greedy_punishment
            else:
                self.reward += self.MAX_COMMUN
    
            # self.reward -= self.avg_fresh   
            
            
        """" Removing the expired content from cache """
        expired = np.where(self.mem_status[self.fresh_slots, :] >= 1)
        self.mem_status[expired[0]*2, expired[1]] = 0 
        self.mem_status[expired[0]*2 + 1, expired[1]] = 0
        self.freshness[expired[0]*2, expired[1]] = 0 
        self.freshness[expired[0]*2 + 1, expired[1]] = 0

        
        
        """ Calling the required functions and returning the obs & reward"""
        self._take_action(action)
        
        # under_utilized = np.where(self.mem_status[self.mem_slots, :] == 0)
        # under_utilized = max(under_utilized[0].shape[0], under_utilized[1].shape[0])
        # under_utilized /= 18
        # # under_utilized = np.clip(under_utilized, 0, 3)
        # self.reward -= under_utilized
        # self.reward = np.clip(self.reward, -21, 100)
        if action>0:
            self.reward -= self.MAX_COMMUN
        
        obs= self._next_observation()
        self.current_step += 1
        self.done = self.current_step >= self.MAX_STEPS
        self.episodes.append(self.reward)
        self.episodes_avg_fresh.append(self.avg_fresh)
        return obs, self.reward, self.done, {}


    def reset(self):
        # print('    !!!!!!    RESTARTED    !!!!')
        # print('    !!!!!!    RESTARTED    !!!!')
        self.current_step = np.random.randint(0, 10000)
        self.done = False
        self.reward= 0
        self.avg_fresh=0
        self.mem_status= np.zeros(shape=(6,6))
        self.freshness= np.zeros(shape=(6,6))
        self.freshness[self.mem_slots, :] = self.current_step
        self.line1= np.zeros(shape=(1,6))
        self.episodes=[]
        return self._next_observation()


    def _take_action(self, action):
        # action = np.round(action)
        action_final = np.array(action).astype(np.uint8)
        action_final = np.unpackbits(action_final, bitorder='little')
        self.action = action_final
        request = self.df.loc[self.current_step].values
        for i in range(3):
            if action_final[i]== 1:
                empty = np.where(self.mem_status[2*i, :]== 0)
                if empty[0].shape != (0,) :
                    self.mem_status[2*i, empty[0][-1]] = request[0]
                    self.mem_status[2*i+1, empty[0][-1]] = 0
                    self.freshness[2*i, empty[0][-1]] = self.current_step
                    self.freshness[2*i+1, empty[0][-1]] = request[1]
                    self.mem_status[2*i: 2*i+2, :empty[0][-1]+1] = np.roll(self.mem_status[2*i: 2*i+2, :empty[0][-1]+1], 1, axis=1)
                    self.freshness[2*i: 2*i+2, :empty[0][-1]+1] = np.roll(self.freshness[2*i: 2*i+2, :empty[0][-1]+1], 1, axis=1)
                else:
                    self.mem_status[2*i, -1] = request[0]
                    self.mem_status[2*i+1, -1] = 0
                    self.freshness[2*i, -1] = self.current_step
                    self.freshness[2*i+1, -1] = request[1]
                    self.mem_status[2*i: 2*i+2, :] = np.roll(self.mem_status[2*i: 2*i+2, :], 1, axis=1)
                    self.freshness[2*i: 2*i+2, :] = np.roll(self.freshness[2*i: 2*i+2, :], 1, axis=1)
                    
                    
                
                
                
                
            
            
    def render(self, mode='human', close= False):
        """   """
        print(f'Step: {self.current_step}')
        print(f'action: {self.action}')
        print(f'Next Requests: {self.df.loc[self.current_step].values}')
        ops = np.concatenate((self.line1, self.mem_status), axis=0)
        ops = np.around(ops, decimals=3)
        print(f'Observation: {ops}')
        fresh2 = np.around(self.freshness, decimals=3)
        print(f'Freshnes: {fresh2}')
        print(f'Reward: {self.reward}')
        if self.current_step == 499 :
            plt.title('rewards in each step')
            plt.plot(self.episodes)
            plt.show()
            plt.title('average freshness in each step (lower is better)')
            plt.plot(self.episodes_avg_fresh)
            plt.show()
            print(f'first reward: {self.episodes[0]}')
            print(f'lastreward: {self.episodes[-1]}')
            