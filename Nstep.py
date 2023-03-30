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

class NstepQLearningAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma, n):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.n = n
        
        #Q_value table Initialized to 0
        self.Q_sa = np.zeros((n_states,n_actions))
        
    # action selection Policy
    def select_action(self, s, policy='egreedy', epsilon=None, temp=None):
        
        if policy == 'egreedy':
             
            epsilon = 0.18
            if np.random.uniform(0,1) < epsilon:
                a = np.random.choice(self.n_actions)
                 
            else :
                a =  argmax(self.Q_sa[s])
            # TO DO: Add own code
            # a = np.random.randint(0,self.n_actions) # Replace this with correct action selection
            
                
        elif policy == 'softmax':
             
            Prob= softmax(self.Q_sa[s],temp)
            a = np.random.choice(self.n_actions,p= Prob)
             
        return a
        
    def update(self, states, actions, rewards, done):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''
        # TO DO: Add own code
        # T_ep = self.n
        # G_target = 0 
        # for n in range(T_ep):
        #     G_target += ((self.gamma)**n )* rewards[n] 
        # (self.gamma)**T_ep * max(self.Q_sa[states][actions]) 
        
        # self.Q_sa[states][actions] += self.learning_rate * (G_target - self.Q_sa[states][actions])
             
def n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy='egreedy', epsilon=None, temp=None, plot=True, n=5):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    pi = NstepQLearningAgent(env.n_states, env.n_actions, learning_rate, gamma, n)
    rewards = []

    # TO DO: Write your n-step Q-learning algorithm here!
     
    
     
    step = 0 
     
    while  step < n_timesteps:
        states, actions, cum_rewards= [], [], [] 
        terminal =[]
        state = env.reset()
        states.append(state)
        
        
        for episode_step in range(max_episode_length):
            a = pi.select_action(state,policy,epsilon, temp)
            
            actions.append(a)
            
            next_state, reward, done = env.step(a)
            states.append(next_state)
            cum_rewards.append(reward)
            # rewards.append(reward)
            terminal.append(done)
            state = next_state
          
            rewards.append(reward)
            step += 1    
            
            
            if done is True or step == n_timesteps:
               
                break
            
       
       
        episode_step += 1
        for  t in range( episode_step):
             
            m = min(n,episode_step - t )
            G_target = 0 
            
            if terminal[t+m-1] is True: 
                for i in range(m):
                     
                    G_target += ((gamma)**i )* rewards[t+i] 
          
            else:
                for i in range(m):
                    G_target += ((gamma)**i )* rewards[t+i] 
                G_target += (gamma)**m * np.max(pi.Q_sa[states[t+m]] )
            pi.Q_sa[states[t]][actions[t]] += learning_rate * (G_target - pi.Q_sa[states[t]][actions[t]]) 
            
            
       
    if plot:
        env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.2) # Plot the Q-value estimates during n-step Q-learning executio
        
      
        
    return rewards 

def test():
    n_timesteps = 50000
    max_episode_length = 1
    gamma = 1.0
    learning_rate = 0.25
    n = 5
    
    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = False

    rewards = n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot, n=n)
    
  
    
if __name__ == '__main__':
    test()
