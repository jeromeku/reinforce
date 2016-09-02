import pdb
import numpy as np
from collections import defaultdict

import tensorflow as tf
import config as config
from agent import PGAgent


def run_trajectory(agent, env):
    states, action_probs, actions, rewards = [], [], [], []

    #Start the simulation, get initial state
    state = env.reset()
    states.append(state)

    #Agent's response to initial state
    aprob, a = agent.act(state)
    actions.append(a)

    done = False

    while not done:
        #Advance simulation wrt agent action and record reward
        next_state, reward, done, info = env.step(a)
        rewards.append(reward)

        #Update and record state
        state = next_state
        states.append(state)        
        
        #Agent response to new state
        aprob, a = agent.act(state)
        actions.append(a)

    return [np.array(l) for l in [states, actions, rewards]]

def rollout(N, agent, env):
    '''Run N rollouts (trajectories)'''
    init_state = env.reset()
    done = False
    trajectories = defaultdict([])

    for i in range(N):
        states, actions, rewards = run_trajectory(agent, env)
        trajectories['states'].append(states)
        trajectories['actions'].append(actions)
        trajectories['rewards'].append(rewards)

    for k,v in trajectories.iteritem():
        trajectories[k] = np.vstack(v)
        
    return trajectories

def agent_cor(agent):
   
    state = yield
    
    try:
        while True:
            aprob, action = agent.act(state)
            next_state = yield action            
            state = next_state
    except GeneratorExit, StopIteration:
        print "Simulation Finished"
    else:
        raise
        
def simulate(agent, env):
    states, actions, rewards = [], [], []
    
    acor = agent_cor(agent)
    next(acor) #prime 
     
    state = init_state = env.reset()
    action = init_action = acor.send(init_state)
    states.append(state)
    actions.append(action)
    done = False

    while not done:       
        next_state, reward, done, info = env.step(action)          
        rewards.append(reward)
        
        state = next_state
        action = acor.send(state)    
        states.append(state)
        actions.append(action)
        
    acor.close()
    
    return map(np.array, [states, actions, rewards])