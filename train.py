import tensorflow as tf
from .config import config
from agent import PGAgent
from collections import defaultdict


def run_trajectory(agent, env):
    states, actions, rewards = [], [], []
    state = env.reset()
    done = False

    while not done:
        a = agent.act(state)
        next_state, reward, done, info = env.step(a)

        states.append(state)
        actions.append(a)
        rewards.append(reward)

        state = next_state

    return states, actions, rewards

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


