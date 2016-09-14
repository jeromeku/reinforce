import numpy as np
import random
import itertools
import numpy as np
from collections import namedtuple

import gym
import tensorflow

from utils import take, partition_points

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

    @property
    def valid_actions(self):
        '''Valid actions limited to 2 (up) and 3 (down)
        '''
        return [2,3]
    
    def simulate(self):
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
                print "point scored"
                self.is_point = True
        
        #Store path to point to current game trajectory
        self.trajectory.append(self._unpack(path))  
        self.is_point = False

        #Append entire game trajectory to environment history
        if self.is_terminal:
            print "End of Game"
            self.history.append(self.trajectory)

        return self._unpack(path)


class Node(object):
    def __init__(self, parent=None, action=None, state=None, terminal=False):
        self.parent = parent
        self.action = action
        self.state = state
        self.is_terminal = terminal
        self.children = []
        self.explored_children = []
        self.value = 0.
        self.visits = 0.
        
        
    def expand(self, env):
        for action in env.valid_actions:
            self.children.append(Node(parent=self, action=action))
        self.child_iter = iter(self.children)
        
    @property
    def num_children(self):
        return len(self.children)
    
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
    def select(self, node, env):
        #Expand if new state (and not terminal)
        if node.num_children == 0:
            print "expanding"
            node.expand(env)
            print 
            
        if node.has_unvisited:
            child = node.get_unvisited()
            print "exploring child {}".format(node.num_explored)
            
            #Simulate
            action = child.action
            result = self.simulate(child, env)
            self.printDx(result)
            
            #Backprop
            self.backprop(result, child)
            
            #Point scored, no need to reset
            if env.is_point:
                print "Point scored"

            #End of game reached, reset game
            if env.is_terminal:
                print "Gameover"
                env.clear()
        else:
            #Run bandit selection algorithm if expanded and all children visited at least once
            self.bandit(node)
            print "all children visited, running bandit"
            return
            #self.select(node, env)

    def simulate(self, node, env):
        print "simulating"

        #Initial step following expansion
        a = node.action
        s, r, terminal, info = env.step(a)
        node.state = s
        
        #Rollout from expanded node
        states, actions, rewards = env.simulate()
        states = [s] + states
        actions = [a] + actions
        rewards = [r] + rewards
        
        return states, actions, rewards

    def bandit(self, node):
        pass

    def printDx(self, result):
        s, a, r = map(np.array, result)
        print "Num steps: ", len(s)
        print "Final Score: {} to {}".format(np.sum(r > 0), np.sum(r < 0))
        print "Num Up moves: {}".format(np.sum(a == 2))
        print "Num Down moves: {}".format(np.sum(a == 3))
        print
        
    def backprop(self, result, child):
        print "backprop'ing"
        s, a, r = result

        print 
        
    