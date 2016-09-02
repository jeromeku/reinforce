import itertools
import numpy as np
from collections import deque

import tensorflow as tf
from utils import wrap_graph_c as wrap_graph

class Policy(object):
    def __init__(self):
        pass

class RandomPolicy(Policy):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, state):
        return np.random.choice(self.action_space, size=1)[0]

class PongAgent(object):

    def __init__(self):
        self.VALID_ACTIONS = [2,3]
        self.MAX_LEN = 100 # Max capacity of experience buffer

        self._state_buffer = deque(maxlen=self.MAX_LEN)
        self._action_buffer = deque(maxlen=self.MAX_LEN)
        self._reward_buffer = deque(maxlen=self.MAX_LEN)
        #self._experience_buffer = deque(maxlen=self.MAX_LEN)

    def _preprocess(self, I):
        I = I[35:195] # crop
        I = I[::2,::2,0] # downsample by factor of 2
        I[I == 144] = 0 # erase background (background type 1)
        I[I == 109] = 0 # erase background (background type 2)
        I[I != 0] = 1 # everything else (paddles, ball) just set to 1

        return I.astype(np.float).ravel()

    def random_policy(self, state):
        a = np.random.choice(self.VALID_ACTIONS, size=1)[0]
        if a == 2:
            aprob = [1., 0.]
        else:
            aprob = [0., 1.]

        return aprob, a

    def act(self, state):
        raise NotImplementedError

    @property
    def num_episodes(self):
        assert len(self._state_buffer) == len(self._action_buffer) == len(self._reward_buffer)
        return len(self._state_buffer)

class RandomAgent(PongAgent):
    def act(self, state):
        a = np.random.choice(self.VALID_ACTIONS, size=1)[0]

        if a == 2:
            aprob = [.5] * 2
        else:
            aprob = [.5] * 2

        return aprob, a
        
class PGAgent(PongAgent):
    
    def __init__(self, g, sess, state_dim, action_net_ctor, action_net_params):
        super(PGAgent, self).__init__()
        
        self.state_dim = state_dim
        
        assert sess.graph == g
        self.g = g
        self.sess = sess
        
        #Create input ops, action network
        self._create_variables()
        self._build_action_network(action_net_ctor, action_net_params)
        
        #Housekeeping: initialize variables, set up tensorflow summaries
        self._initialize()
    
    @wrap_graph
    def _initialize(self):
        init_op = tf.initialize_all_variables()
        self.sess.run(init_op)
    
    @wrap_graph
    def _create_variables(self):
        with tf.name_scope("inputs"):
            self.states = tf.placeholder(tf.float32, (None, self.state_dim), name="states")
            self.test_var = tf.Variable(initial_value=1.0, name="test_var")

    @wrap_graph
    def _build_action_network(self, action_net_ctor, ctor_params):
        with tf.variable_scope("action_network"):
            self.action_net = action_net_ctor(self.states, **ctor_params)
            self.action_logits, self.action_probs = self.action_net

    def act(self, state):
        
        #Preprocess raw observation
        state = self._preprocess(state)
        if len(state.shape) == 1:
            state = state.reshape((1, self.state_dim))
        
        #Run policy to get action
        action_probs = self.action_probs.eval(session=self.sess, feed_dict={self.states: state})
        action = np.argmax(action_probs) + 2 #map to discrete state 2 (up) or 3 (down)

        return action_probs, action
    
    def run_trajectory(self, env):
        states, action_probs, actions, rewards = [], [], [], []

        #Start the simulation by getting initial state
        state = env.reset()
        done = False

        while not done:
            #Agent action
            aprob, a = self.act(state)
            #Advance simulation wrt agent action and record
            next_state, reward, done, info = env.step(a)
            states.append(state)        
            actions.append((aprob, a))
            rewards.append(reward)

            #Update state
            state = next_state

        #Write to experience buffer
        self._state_buffer.append(states)
        self._action_buffer.append(actions)
        self._reward_buffer.append(rewards)

        return states, actions, rewards

    def rollout(self, N, env):
        '''Run N rollouts (trajectories)
        
        Returns list of N (states, actions, rewards) tuples where N is the number of rollouts
        '''
        
        trajectories = []
        done = False

        for i in range(N):
            states, actions, rewards = self.run_trajectory(env)
            trajectories.append((states, actions, rewards))
        
        return trajectories

    @property
    def _episodes(self):
        return zip(self._state_buffer, self._action_buffer, self._reward_buffer)
    
    def fetch_episodes(self, N, shuffle=True):
        '''Retrieve min(N, num_episodes) random episodes from experience buffer
        Returns list of (states, actions, rewards) episode tuples
        '''
        episodes = self._episodes
        indices = np.random.choice(self.num_episodes, size=min(N, self.num_episodes), replace=False)
        shuffled = [episodes[i] for i in indices]
        
        return shuffled
