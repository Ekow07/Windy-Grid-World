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

class MonteCarloAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        
        #Q_value table Initialized to 0
        self.Q_sa = np.zeros((n_states,n_actions))
        
    #actiono Selection Policy
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
        
    def update(self, states, actions, rewards):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''
        # TO DO: Add own code
         
        pass
         
       
         

def monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy='egreedy', epsilon=None, temp=None, plot=True):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    pi = MonteCarloAgent(env.n_states, env.n_actions, learning_rate, gamma)
    rewards = []
    
    # TO DO: Write your Monte Carlo RL algorithm here!
    step = 0 
    counter =0
    while step < n_timesteps:
        states, actions, cum_rewards = [], [], [] 
        state = env.reset()
        states.append(state)
        
        for episode_step in range(max_episode_length-1):
            a = pi.select_action(state,policy,epsilon,temp)
            
            
            
            next_state, reward, done = env.step(a)
            
            actions.append(a)
            states.append(next_state)
            # cum_rewards.append(reward)
            
            rewards.append(reward)
            
            state = next_state
            step += 1 
            if done is True or  step == n_timesteps:
                counter += 1
                break
        # rewards.append(reward)
        
        G_target_next = 0
        G_target=0 
        for i in range(episode_step,0,-1):
            # pi.update(state,actions,rewards,done) 
              
            G_target +=   rewards[i] + (gamma* G_target_next) 
            pi.Q_sa[states[i]][actions[i]] += learning_rate * (G_target - pi.Q_sa[states[i]][actions[i]] )
            G_target_next = G_target
            # pi.Q_sa[i] += learning_rate * (G_target - pi.Q_sa[i]) 
    print(counter) 
    if plot:
       env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during Monte Carlo RL execution

    return rewards 
    
def test():
    n_timesteps = 10000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True

    rewards = monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot)
    # print("Obtained rewards: {}".format(rewards))
    
if __name__ == '__main__':
    test()
