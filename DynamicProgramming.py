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
from Helper import argmax
import matplotlib as plt

class QValueIterationAgent:
    ''' Class to store the Q-value iteration solution, perform updates, and select the greedy action '''

    def __init__(self, n_states, n_actions, gamma, threshold=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states,n_actions))
        
    def select_action(self,s):
        ''' Returns the greedy best action in state s ''' 
        # TO DO: Add own code
        #a = np.random.randint(0,self.n_actions) # Replace this with correct action selection
        a = argmax(self.Q_sa[s])
    
        return a
        
    def update(self,s,a,p_sas,r_sas):
        ''' Function updates Q(s,a) using p_sas and r_sas '''
        # TO DO: Add own code\
        #print(self.Q_sa)
        Q_old = self.Q_sa.copy()
        self.Q_sa[s,a] = 0
        for s_next in range(self.n_states):
            self.Q_sa[s,a] += p_sas[s, a, s_next] * (r_sas[s,a,s_next] + self.gamma * max(Q_old[s_next]))
            
        
        
        pass  
    
#print (QValueIterationAgent.Q_sa )
    
    
    
def Q_value_iteration(env, gamma=1.0, threshold=0.001):
    ''' Runs Q-value iteration. Returns a converged QValueIterationAgent object '''
    
    QIagent = QValueIterationAgent(env.n_states, env.n_actions, gamma)    
        
    # TO DO: IMPLEMENT Q-VALUE ITERATION HERE
    while True:
        delta = 0 
        for s in range(env.n_states):
            for a in range(env.n_actions):
                x = QIagent.Q_sa[s,a]
                QIagent.update(s, a, env.p_sas, env.r_sas)
                                    
                delta = max(delta, abs(x-QIagent.Q_sa[s,a]))
                
                #print(delta)
        if delta < threshold:
            break
       
        
    # Plot current Q-value estimates & print max error
    env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.9)
    #print("Q-value iteration, iteration {}, max error {}".format(i,max_error))
    
    return QIagent

def experiment():
    gamma = 1.0
    threshold = 0.001
    env = StochasticWindyGridworld(initialize_model=True)
    # env.render()
    QIagent = Q_value_iteration(env,gamma,threshold)
    rewards = []
    
    # View optimal policy
    done = False
    s = env.reset()
    while not done:
        a = QIagent.select_action(s)
        s_next, r, done = env.step(a)
        #env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.9)
        s = s_next  
        rewards.append(r)
        
        
         
    
    
    # TO DO: Compute mean reward per timestep under the optimal policy
    
    print('Mean reward per timestep under optimal policy: ', np.mean(rewards))
    
if __name__ == '__main__':
    experiment()



 