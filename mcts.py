import numpy as np
import sys
import random
import logging
import itertools
import numpy as np
from collections import namedtuple
import argparse

import gym
import tensorflow

from utils import take, partition_points

LOG_FILE = "./MCTS.log"
LOG_FMT = "%(name)s %(asctime)s %(levelname)s %(message)s"
DATE_FMT = '%m/%d/%Y %I:%M:%S'
LOG_LEVEL = logging.INFO
logging_params = { 'format': LOG_FMT,
                   'datefmt': DATE_FMT,
                   'filename': LOG_FILE, 
                   'filemode': 'w',
                   'level': LOG_LEVEL
                 }

formatter = logging.Formatter(LOG_FMT, DATE_FMT)

fhdlr = logging.FileHandler(LOG_FILE, mode='w')
fhdlr.setLevel(logging.DEBUG)
fhdlr.setFormatter(formatter)

chdlr = logging.StreamHandler(sys.stdout)
chdlr.setLevel(logging.CRITICAL)
chdlr.setFormatter(formatter)

class TestEnv(object):
    def __init__(self, name):
        self.env = gym.make(name)
        self.state = self.env.reset()
        self.trajectory = []
        self.history = []
        self.is_terminal = False
        
    def reset(self):
        self.state = self.env.reset()
        self.history = []
    
    def step(self, a):
        next_state, reward, terminal, info = self.env.step(a)
        self.state = next_state
        self.is_terminal
        return next_state, reward, terminal, info
        
    # def simulate(self):
    #     '''Simulate path starting from current state
    #     '''
    #     path = []
    #     while not self.is_terminal:
    #         action = self.sample_action()
    #         next_state, reward, terminal, info = self.env.step(action)
    #         path.append((self.state, action, reward))
    #         self.state = next_state
    #         self.is_terminal = terminal
            
    #     return self._unpack(path)
    
    def rollout(self, num_plays=1):
        '''Simulate num_play trajectories
        '''
        if self.is_terminal:
            self.clear()
            
        for i in range(num_plays):
            trajectory = self.simulate()
            self.history.append(trajectory)
            self.clear()

    def _unpack(self, trajectory):
        s, a, r = map(list, zip(*trajectory))
        return s, a, r
    
    def get_trajectories(self):
        '''Returns list of trajectories, where each trajectory is a list of states, actions, and rewards
        '''
        if self.history:
            return [self._unpack(trajectory) for trajectory in self.history]
        else:
            print "No trajectories"
            return None
        
    def sample_action(self, state=None, policy=None, n=1):
        if not policy:
            return np.random.choice(self.valid_actions, size=n)
            
    def clear(self):
        self.trajectory = []
        self.state = self.env.reset()
        self.is_terminal = False
        
    def clear_all(self):
        self.clear()
        self.history = []
    
    @property
    def valid_actions(self):
        raise NotImplementedError

        
class PongEnv(TestEnv):
    def __init__(self):
        super(PongEnv, self).__init__("Pong-v0")
        self.is_point = False #state for when a point is scored by either player
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(chdlr)
        self.logger.addHandler(fhdlr)

    @property
    def games_played(self):
        return len(self.history)
    
    @property
    def points_played(self):
        '''Points played = points from history + current trajectory
        '''
        history_pts = sum(len(p) for p in self.history)
        current_pts = len(self.trajectory)

        return history_pts + current_pts

    @property
    def valid_actions(self):
        '''Valid actions limited to 2 (up) and 3 (down)
        '''
        return [2,3]
    
    def simulate(self, max_points=None):
        '''Simulate path starting from current state
        '''
        path = []

        while not self.is_terminal and not self.is_point:
        
            action = self.sample_action()
            next_state, reward, terminal, info = self.env.step(action)
            path.append((self.state, action, reward))
            self.state = next_state
            self.is_terminal = terminal
        
            if abs(reward) > 0:
                self.logger.info("point scored")
                self.is_point = True
        
        #Store point path to current game trajectory
        self.trajectory.append(self._unpack(path))  
        self.is_point = False

        #Append entire game trajectory to environment history
        if self.is_terminal:
            self.logger.info("End of Game")
            self.history.append(self.trajectory)
            self.clear()

        return self._unpack(path)


class Node(object):
    def __init__(self, parent=None, action=None, state=None, root=False, terminal=False):
        self.parent = parent
        self.action = action
        self.state = state
        self.is_root = root
        self.is_terminal = terminal
        self.children, self.explored_children = [],[]
        self.value, self.visits = 0., 0.
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(chdlr)
        self.logger.addHandler(fhdlr)

        if root:
            self.name = "root"
        else:
            child_num = len(self.parent.children) + 1
            self.name = self.parent.name + "/" + "child" + str(child_num) 
        
    def expand(self, env):
        for action in env.valid_actions:
            self.children.append(Node(parent=self, action=action))
        self.child_iter = iter(self.children)
        
    @property
    def num_children(self):
        return len(self.children)
    
    @property
    def has_children(self):
        return self.num_children > 0
    
    @property
    def num_explored(self):
        return len(self.explored_children)
    
    def get_unvisited(self):
        child = next(self.child_iter, None)
        self.explored_children.append(child)
        return child
        
    @property
    def has_unvisited(self):
        return (self.num_explored < self.num_children) and self.num_children > 0
    
fields = ["state", "action", "parent", "children", "explored_children", "visits", "value"]

class MCTS(object):

    def __init__(self, gamma=1):
        self.gamma = gamma
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(chdlr)
        self.logger.addHandler(fhdlr)
    
    def ucb(self, node):
        exploit_score = float(node.value) / node.visits
        explore_score = np.sqrt((2 * np.log(node.parent.visits)) / float(node.visits))
        
        return exploit_score + explore_score

    def expand(self, node, env):
        for action in env.valid_actions:
            node.children.append(Node(parent=node, action=action))
        
        node.child_iter = iter(node.children)

    def select(self, node, env):
        #Expand if new state (and not terminal)
        if node.num_children == 0:
            self.logger.info("expanding {}".format(node.name))
            self.expand(node, env)
            
        if node.has_unvisited:
            child = node.get_unvisited()
            self.logger.info("exploring child {}".format(child.name))
            
            #Simulate
            action = child.action
            results = self.simulate(child, env)
            self.printDx(results)
            states, actions, rewards = results
            
            #Backprop
            self.backprop(rewards[-1], child)
            
            #Point scored, no need to reset
            if env.is_point:
                self.logger.info("Point scored")

            #End of game reached, reset game
            if env.is_terminal:
                self.logger.info("Gameover")
                env.clear()
        else:
            #Run bandit selection algorithm if expanded and all children visited at least once
            ucb_scores = map(self.ucb, node.children)
            best_child = np.argmax(ucb_scores)
            selected = node.children[best_child]
            self.logger.info("all children visited, running bandit")
            self.logger.info("selected child {}".format(best_child + 1))
            self.select(selected, env)

    def simulate(self, node, env, max_points=None):
        self.logger.info("simulating...")

        #Initial step following expansion
        a = node.action
        s, r, terminal, info = env.step(a)
        node.state = s
        
        #Rollout from expanded node
        states, actions, rewards = env.simulate(max_points=max_points)
        states = [s] + states
        actions = [a] + actions
        rewards = [r] + rewards
        
        return states, actions, rewards

    def bandit(self, node):
        pass

    def printDx(self, result):
        s, a, r = map(np.array, result)
        print
        print "{0} Point Dx {0}".format("".join(["*"] * 6))
        print "Num steps: ", len(s)
        print "Final Score: {} to {}".format(np.sum(r > 0), np.sum(r < 0))
        print "Num Up moves: {}".format(np.sum(a == 2))
        print "Num Down moves: {}".format(np.sum(a == 3))
        print
        
    def backprop(self, reward, node):
                
        #Update stats starting until root
        self.logger.info("backpropagating...")
        node.visits += 1
        node.value += reward
        if not node.is_root:
            self.backprop(self.gamma*reward, node.parent)
        else:
            self.logger.info("reached root")
            self.logger.info("{}".format(" ".join(["*"] * 13)))

if __name__ == "__main__":
    
    global DEBUG
    global NUM_ROLLOUTS

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action="store_true", help="Output debugging messages")
    parser.add_argument('--num_rollouts', default=1, help="Number of rollouts to simulate", type=int)
    args = parser.parse_args()
    
    DEBUG = args.debug
    NUM_ROLLOUTS = args.num_rollouts

    if DEBUG:
        logging.getLogger().handlers[0].setLevel(logging.DEBUG)
    else:
        logging.getLogger().handlers[0].setLevel(logging.ERROR)

    pong = PongEnv()

    #Initial State
    init_state = pong.state
    root = Node(state=init_state, root=True)

    #Run MCTS
    mcts = MCTS(gamma=1)
    #logging.info("Running {} rollouts".format(NUM_ROLLOUTS))
    for i in range(NUM_ROLLOUTS):
        mcts.select(root, pong)

    