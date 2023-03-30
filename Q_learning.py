#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2021
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import softmax, argmax

class QLearningAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        
        # Q_Value table Initialized to 0
        self.Q_sa = np.zeros((n_states,n_actions))
        
    # Action  selection Policy  
    
    def select_action(self, s, policy='egreedy', epsilon=None, temp=None):
        
        if policy == 'egreedy':
             
            if np.random.uniform(0,1) < epsilon:
                a= np.random.choice(self.n_actions)
                 
            else :
                a =  argmax(self.Q_sa[s]) #argmax(self.n_actions)
               
            # a = np.random.randint(0,self.n_actions) # Replace this with correct action selection
        
            
        elif policy == 'softmax':
             
            Prob= softmax(self.Q_sa[s],temp)
            a = np.random.choice(self.n_actions,p= Prob)
             
        return a
        
    # Q_Learning Update Function     
    def update(self,s,a,r,s_next,done):
         
        predict = self.Q_sa[s][a]
        G_target = r + self.gamma * max(self.Q_sa[s_next])
        
        self.Q_sa[s][a] += self.learning_rate * (G_target - predict)
         
         

def q_learning(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True):
    ''' runs a single repetition of q_learning
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    env.render()
    pi = QLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    rewards = []

    # TO DO:   Q-learning algorithm implementation!
    counter = 0
    s = env.reset()
    for n in range(n_timesteps):
        a = pi.select_action(s,policy,epsilon , temp )
        
        s_next, r, done = env.step(a)
        rewards.append(r)
        #   
        # self,s,a,r,s_next,done
        pi.update( s,a,r, s_next,  done)
        
        if done is True:
            s = env.reset()
            counter+=1
        else:
            s = s_next
    
    # if plot:
    #    env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during Q-learning execution
    # print("Number of successful runs",counter)
    return rewards

def test():
    
    n_timesteps = 10000
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True

    rewards = q_learning(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot)
    # print("Obtained rewards: {}".format(rewards))

if __name__ == '__main__':
    test()
