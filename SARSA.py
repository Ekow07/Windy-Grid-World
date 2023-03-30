#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2022
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import softmax, argmax

class SarsaAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        
        #Q_value table Initialized to 0
        self.Q_sa = np.zeros((n_states,n_actions))
        
        
    # Action Selection Policy
    def select_action(self, s, policy='egreedy', epsilon=None, temp=None):
        
        if policy == 'egreedy':
            if np.random.uniform(0,1) < epsilon:
                a = np.random.choice(self.n_actions)
                 
            else :
                a =  argmax(self.Q_sa[s])
                
        elif policy == 'softmax':
           
            Prob= softmax(self.Q_sa[s],temp)
            a = np.random.choice(self.n_actions,p= Prob)
            
        return a
    
    #update function
    def update(self,s,a,r,s_next,a_next,done):
        # TO DO: Add own code
        G_target = r + self.gamma * self.Q_sa[s_next][a_next]
        
         
        self.Q_sa[s][a] +=   self.learning_rate * (G_target - self.Q_sa[s][a])
            
       
        
def sarsa(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True):
    ''' runs a single repetition of SARSA
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    pi = SarsaAgent(env.n_states, env.n_actions, learning_rate, gamma)
    rewards = []

    # TO DO: Write your SARSA algorithm here!
    counter = 0 
    s = env.reset()
    a = pi.select_action(s,policy,epsilon,temp)
    for n in range(n_timesteps ):
        
        s_next, r, done = env.step(a)
        a_next = pi.select_action(s_next,policy,epsilon,temp)
        # s_next, r, done = env.step(a)
        # pi.update( s,a,r, s_next,  done)
        pi.update(s,a, r,s_next, a_next,  done)
        
        rewards.append(r)
        if done is True:
            s = env.reset()
            a = pi.select_action(s,policy,epsilon,temp)
            counter+= 1
        else:
            s = s_next
            a = a_next
    
    # if plot:
    #    env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during SARSA execution
    print("Number of successful runs",counter)
    return rewards 


def test():
    n_timesteps = 10001
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True
    
    rewards = sarsa(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot)
    print(len(rewards))
    # print("Obtained rewards: {}".format(rewards))        
    
if __name__ == '__main__':
    test()
