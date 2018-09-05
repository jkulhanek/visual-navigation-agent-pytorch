# -*- coding: utf-8 -*-
import sys
import h5py
import json
import numpy as np
import random
import skimage.io
from skimage.transform import resize
from . import Environment
from torchvision import transforms

class THORDiscreteEnvironment(Environment):
    def __init__(self, 
            scene_name = 'bedroom_04',
            random_start = True,
            n_feat_per_location = 1,
            history_length : int = 4,
            screen_width = 224,
            screen_height = 224,
            terminal_state_id = 0,
            h5_file_path = None,
            ):
        super(THORDiscreteEnvironment, self).__init__()

        if h5_file_path is None:
            h5_file_path = f"/app/data/{scene_name}.h5"

        self.h5_file = h5py.File(h5_file_path, 'r')
        self.n_feat_per_location = n_feat_per_location
        self.locations = self.h5_file['location'][()]
        self.rotations = self.h5_file['rotation'][()]
        self.history_length = history_length
        self.n_locations = self.locations.shape[0]
        self.terminals = np.zeros(self.n_locations)
        self.terminals[terminal_state_id] = 1
        self.terminal_states, = np.where(self.terminals)
        self.transition_graph = self.h5_file['graph'][()]
        self.shortest_path_distances = self.h5_file['shortest_path_distance'][()]
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


    def reset(self):
        # randomize initial state
        while True:
            k = random.randrange(self.n_locations)
            min_d = np.inf

            # check if target is reachable
            for t_state in self.terminal_states:
                dist = self.shortest_path_distances[k][t_state]
                min_d = min(min_d, dist)

            # min_d = 0  if k is a terminal state
            # min_d = -1 if no terminal state is reachable from k
            if min_d > 0: break
        
        # reset parameters
        self.current_state_id = k
        self.s_t = self._tiled_state(self.current_state_id)

        self.reward   = 0
        self.collided = False
        self.terminal = False

    def step(self, action):
        assert not self.terminal, 'step() called in terminal state'
        k = self.current_state_id
        if self.transition_graph[k][action] != -1:
            self.current_state_id = self.transition_graph[k][action]
            if self.terminals[self.current_state_id]:
                self.terminal = True
                self.collided = False
            else:
                self.terminal = False
                self.collided = False
        else:
            self.terminal = False
            self.collided = True

        self.reward = self._reward(self.terminal, self.collided)
        self.s_t = np.append(self.s_t[:,1:], self.state, axis=1)


    def _tiled_state(self, state_id):
        k = random.randrange(self.n_feat_per_location)
        f = self.h5_file['resnet_feature'][state_id][k][:,np.newaxis]
        f = self.normalize(f)
        return np.tile(f, (1, self.history_length))

    def _reward(self, terminal, collided):
        # positive reward upon task completion
        if terminal: return 10.0
        # time penalty or collision penalty
        return -0.1 if collided else -0.01

    @property
    def state(self):
        return {
            "reward" : self.reward,
            "terminal" : self.terminal
        }

    @property
    def is_terminal(self):
        return self.state['terminal']

    def render(self, mode):
        return self.s_t

    def render_goal(self, mode):
        return self.