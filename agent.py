import itertools
import numpy as np
from collections import deque
import pdb

import tensorflow as tf
from toolz import accumulate

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

        #Experience buffers: state_buffer stores raw observations, internal_state_buffer stores preprocessed states that are fed to the policy
        self._state_buffer = deque(maxlen=self.MAX_LEN)
        self._internal_state_buffer = deque(maxlen=self.MAX_LEN)
        self._action_logits_buffer = deque(maxlen=self.MAX_LEN)
        self._action_buffer = deque(maxlen=self.MAX_LEN)
        self._reward_buffer = deque(maxlen=self.MAX_LEN)

    def _preprocess(self, I):
        I = I[35:195] # crop
        I = I[::2,::2,0] # downsample by factor of 2
        I[I == 144] = 0 # erase background (background type 1)
        I[I == 109] = 0 # erase background (background type 2)
        I[I != 0] = 1 # everything else (paddles, ball) just set to 1

        return I.astype(np.float).ravel()

    def act(self, state):
        raise NotImplementedError

    @property
    def num_episodes(self):
        assert len(self._state_buffer) == len(self._action_buffer) == len(self._reward_buffer)
        return len(self._state_buffer)

    def discount(self, rs, discount_rate):
        discounted = accumulate(lambda prev, curr: discount_rate * prev + curr, reversed(rs))
        return np.fromiter(discounted,'float')[::-1]

    def partition_rewards(self, rewards):
        '''Partition episode of rewards into list of iterables where each sequence ends when reward != 0
        
        Returns:
            list of lists: each list is a reward sequence within the episode
        '''
        rewards = np.array(rewards)
        bounds = np.zeros(np.count_nonzero(rewards) + 1, dtype=int)
        bounds[1:] = np.argwhere(rewards).ravel() + 1
        return [rewards[bounds[i]:bounds[i+1]] for i in range(len(bounds) - 1)]

    def discount_rewards(self, rewards, discount_rate):
        rewards = self.partition_rewards(rewards)
        discounted_seqs = [self.discount(rs, discount_rate) for rs in rewards]
        return np.concatenate(discounted_seqs).reshape((-1,1)).ravel()

class RandomAgent(PongAgent):
    def act(self, state):
        a = np.random.choice(self.VALID_ACTIONS, size=1)[0]

        if a == 2:
            aprob = [.5] * 2
        else:
            aprob = [.5] * 2

        return aprob, a
        
class PGAgent(PongAgent):
    
    def __init__(self, g, sess, state_dim, action_net_ctor, action_net_params, gamma=.99):
        super(PGAgent, self).__init__()
        
        self.state_dim = state_dim
        self.gamma = gamma #rate at which to discount rewards

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
            self.labels = tf.placeholder(tf.int32, shape=[None], name="labels")
            self.rewards = tf.placeholder(tf.float32, shape=[None], name="rewards")
    @wrap_graph
    def _build_action_network(self, action_net_ctor, ctor_params):
        with tf.variable_scope("action_network"):
            self.action_net = action_net_ctor(self.states, **ctor_params)
            self.action_logits, self.action_probs = self.action_net

    @wrap_graph
    def _calculate_loss(self):
        '''Policy Gradient loss
        TODO:
        regularization
        gradient clipping
        advantage estimation
        actor-critic
        '''
        with tf.name_scope("pg_gradient"):
            self.actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="action_network")

            #Logits
            with tf.variable_scope("action_network", reuse=True):
                action_logits = self.action_logits
    
            #Cross Entropy Loss, Logits as predictions, Actions taken as "targets"
            with tf.name_scope("loss"):    
                self.x_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(action_logits, self.labels, name="cross_entropy_loss")
                self.pg_loss = tf.reduce_mean(self.x_entropy_loss, name="pg_loss")

            #Discounted rewards for policy gradient
            self.discounted_r = self.discount_rewards(self.rewards, self.gamma)
            #normalize discounted rewards
            self.discounted_r -= np.mean(self.discounted_r)
            self.discounted_r /= np.std(self.discounted_r)

            #Combine cross entropy loss and discounted rewards to arrive at policy gradients - per episode per step per variable.  I.e., if batch size is an episode, then have batch_size rewards per episode and batch_size grads per variable per episode, so grad * discounted rewards will be component-wise multiplication of dimension batch_size x grad_size
            with tf.name_scope("gradient_calc"):
                self.gradients = self.optimizer.compute_gradients(self.pg_loss, self.actor_vars)
                for i, (grad, var) in enumerate(self.gradients):
                    if grad is not None:
                        self.gradients[i] = (grad * self.discounted_r)

            with tf.name_scope("summaries"):
                tf.scalar_summary("actor_loss", self.pg_loss)

                #Gradient summaries
                for grad, var in self.gradients:
                    tf.histogram_summary(var.name, var)
                    if grad is not None:
                        tf.histogram_summary(var.name + '/gradients', grad)
                
                self.summarize = tf.merge_all_summaries()
            
            with tf.name_scope("train"):
                self.train_op = self.optimizer.apply_gradients(self.gradients)

    def act(self, state):
        """Policy implementation: given state, returns action logits and actions

        Args:
            state: should be in preprocessed form
        Returns:
            2-tuple consisting of logits and action
        """
        #Preprocess raw observation
        #state = self._preprocess(state)
        if len(state.shape) == 1:
            state = state.reshape((1, self.state_dim))
        
        #Run policy to get action
        action_logits = self.action_logits.eval(session=self.sess, feed_dict={self.states: state})
        action = np.argmax(action_logits) + 2 #map to discrete state 2 (up) or 3 (down)

        return action_logits, action
    
    def run_trajectory(self, env):
        states, prep_states = [], []
        action_logits, actions = [], []
        rewards = []

        #Start the simulation by getting initial state
        state = env.reset()
        done = False

        while not done:
            #Agent action
            prep_state = self._preprocess(state)
            alog, a = self.act(prep_state)

            #Advance simulation wrt agent action and record
            next_state, reward, done, info = env.step(a)
            states.append(state)        
            prep_states.append(prep_state)
            action_logits.append(alog)
            actions.append(a)
            rewards.append(reward)

            #Update state
            state = next_state

        #Write to experience buffer
        self._state_buffer.append(np.array(states))
        self._internal_state_buffer.append(np.array(prep_states))
        self._action_logits_buffer.append(np.array(action_logits))
        self._action_buffer.append(np.array(actions))
        self._reward_buffer.append(np.array(rewards))

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
        return zip(self._state_buffer, self._internal_state_buffer, self._action_logits_buffer, self._action_buffer, self._reward_buffer)
    
    def shuffle_episodes(self):
        '''Retrieve min(N, num_episodes) random episodes from experience buffer
        Returns list of (states, actions, rewards) episode tuples
        '''
        episodes = self._episodes
        indices = np.random.permutation(len(episodes))
        shuffled = [episodes[i] for i in indices]

        return shuffled

    def batch_iter(self):
        shuffled_eps = self.shuffle_episodes()

        for eps in shuffled_eps:
            pdb.set_trace()
            _, prep_state, _, actions, rewards = eps
            yield prep_state, actions, rewards

