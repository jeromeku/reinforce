import matplotlib.pyplot as plt
from functools import wraps

import gym

def display(state, preprocessor=None):
    if preprocessor:
        state = preprocessor(state).reshape(80,80)

    plt.imshow(state, cmap=plt.cm.Greys_r)

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

