# -*- coding: utf-8 -*-
import sys
import h5py
import json
import numpy as np
import random
import skimage.io
from skimage.transform import resize
from agent.environment.environment import Environment
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

        self.terminal_state_id = terminal_state_id
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
        self.s_target = self._tiled_state(self.terminal_state_id)
        self.time = 0

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

        self.collided = False
        self.terminal = False
        self.time = 0

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

        self.s_t = np.append(self.s_t[:,1:], self._get_state(self.current_state_id), axis=1)
        self.time = self.time + 1

    def _get_state(self, state_id):
        # read from hdf5 cache
        k = random.randrange(self.n_feat_per_location)
        return self.h5_file['resnet_feature'][state_id][k][:,np.newaxis]

    def _tiled_state(self, state_id):
        f = self._get_state(state_id)
        return np.tile(f, (1, self.history_length))

    def _calculate_reward(self, terminal, collided):
        # positive reward upon task completion
        if terminal: return 10.0
        # time penalty or collision penalty
        return -0.1 if collided else -0.01

    @property
    def reward(self):
        return self._calculate_reward(self.is_terminal, self.collided)

    @property
    def is_terminal(self):
        return self.terminal or self.time >= 5e3

    def render(self, mode):
        assert mode == 'resnet_features'
        return self.s_t

    def render_target(self, mode):
        assert mode == 'resnet_features'
        return self.s_target

    @property
    def actions(self):
        return ["MoveForward", "RotateRight", "RotateLeft", "MoveBackward"]