import numpy as np
import tensorflow as tf
from utils import wrap_graph_c as wrap_graph

class PongAgent(object):

    VALID_ACTIONS = [2,3]

    def __init__(self):
        pass

    def _preprocess(self, I):
        I = I[35:195] # crop
        I = I[::2,::2,0] # downsample by factor of 2
        I[I == 144] = 0 # erase background (background type 1)
        I[I == 109] = 0 # erase background (background type 2)
        I[I != 0] = 1 # everything else (paddles, ball) just set to 1

        return I.astype(np.float).ravel()

    def act(self, state):
        raise NotImplementedError


class RandomAgent(PongAgent):
    def act(self, state):
        action = np.random.choice(PongAgent.VALID_ACTIONS, size=1)[0]
        return action

class PGAgent(PongAgent):
    
    def __init__(self, g, sess, state_dim, action_net_ctor, action_net_params):
        #super(PGAgent, self).__init__()
        #self._action_network = action_network
        
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
        #with self.g.as_default():
        with tf.name_scope("inputs"):
            self.states = tf.placeholder(tf.float32, (None, self.state_dim), name="states")
            self.test_var = tf.Variable(initial_value=1.0, name="test_var")

    @wrap_graph
    def _build_action_network(self, action_net_ctor, ctor_params):
    #with self.g.as_default():
        with tf.variable_scope("action_network"):
            self.action_net = action_net_ctor(self.states, **ctor_params)
            self.action_logits, self.action_probs = self.action_net

    def act(self, state):
        state = self._preprocess(state)
        if len(state.shape) == 1:
            state = state.reshape((1, self.state_dim))
        action_probs = self.action_probs.eval(session=self.sess, feed_dict={self.states: state})
        action = np.argmax(action_probs) + 2 #map to discrete state 2 (up) or 3 (down)
        return action_probs, actio