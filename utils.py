import numpy as np
import matplotlib.pyplot as plt
from functools import wraps
from toolz import accumulate, compose

import gym

def display(state, preprocessor=None):
    if preprocessor:
        state = preprocessor(state).reshape(80,80)

    plt.imshow(state, cmap=plt.cm.Greys_r)

def trajectory_dx(states, actions, rewards):
    print "Total actions: {}".format(len(actions))
    print "Total reward: {}".format(np.sum(rewards))
    
    print "# of Up moves: {}".format(np.sum(actions == 2))
    print "# of Down moves: {}".format(np.sum(actions == 3))
    print "# of positive rewards {}".format(np.sum(rewards > 0))
    print "# of negative rewards {}".format(np.sum(rewards < 0))

def play(env, agent):
    state = env.reset()
    display(state, agent._preprocess)

    done = False
    action = agent.act(state)

    while not done:
        next_state, reward, done, info = env.step(action)
        state = next_state
        print action

        yield display(state, preprocessor=agent._preprocess)
        action = agent.act(state)

def discount_rewards(rewards, discount_rate):
    pass

def discount(rs, discount_rate):
    discounted = accumulate(lambda prev, curr: discount_rate * prev + curr, reversed(rs))
    return np.fromiter(discounted,'float')[::-1]

def partition_rewards(rewards):
    '''Partition rewards by reward sequence where each sequence ends when reward != 0
    
    Returns:
        list of lists: each list is a reward sequence
    '''
    rewards = rewards.ravel()
    bounds = np.zeros(np.count_nonzero(rewards) + 1, dtype=int)
    bounds[1:] = np.argwhere(rewards).ravel() + 1
    return [rewards[bounds[i]:bounds[i+1]] for i in range(len(bounds) - 1)]

def discount_rewards(rewards, discount_rate):
    discounted_seqs = [discount(rs, discount_rate) for rs in rewards]
    return np.concatenate(discounted_seqs).reshape((-1,1)).ravel()

def discount_check(rewards, discount_rate):
    def Kp_rewards(r, discount_rate):
      """ take 1D float array of rewards and compute discounted reward """
      discounted_r = np.zeros_like(r)
      running_add = 0
      for t in reversed(xrange(0, r.size)):
        if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * discount_rate + r[t]
        discounted_r[t] = running_add
      return discounted_r

    my_r = discount_rewards(partition_rewards(rewards), discount_rate)
    Kp_r = Kp_rewards(rewards, discount_rate)

    print np.allclose(my_r, Kp_r)
    return my_r, Kp_r


def normalize(r):
    r -= np.mean(r)
    r /= np.std(r)
    return r
    
def wrap_graph(g):
    '''Decorator for adding ops to a graph'''
    def wrapper(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            with g.as_default():
                return f(*args, **kwargs)
        return wrapped
    return wrapper

def wrap_graph_c(f):
    '''Decorator for adding ops to a graph within a class'''
    @wraps(f)
    def wrapped(self, *args, **kwargs):
        with self.g.as_default():
            return f(self, *args, **kwargs)
    return wrapped

