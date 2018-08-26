from .network import SceneSpecificNetwork, SharedNetwork
from .environment import AI2ThorEnvironment
import torch.nn as nn
from typing import Dict
import random
import torch
from input import get_screen

class TrainingThread:
    def __init__(self, device : torch.device, env : AI2ThorEnvironment, shared_network : SharedNetwork, scene : str, scene_networks : Dict[str, SceneSpecificNetwork]):
        self.env = env
        self.device = device
        self.shared_network = shared_network
        self.scene_networks = scene_networks
        self.action_space_size = self.get_action_space_size()
        if not scene in scene_networks:
            self.scene_networks[scene] = SceneSpecificNetwork(self.action_space_size)
        
        self.scene_network : SceneSpecificNetwork = self.scene_networks[scene]

    def get_action_space_size(self):
        return len(self.env.actions)


    def choose_action(self, policy):
        # TODO: a place for modification
        # Computes weighed randomized value over policy's preferences
        values = []
        sum = 0.0
        for rate in policy:
            sum = sum + rate
            value = sum
            values.append(value)
        r = random.random() * sum
        for i in range(len(values)):
            if values[i] >= r:
                return i
        # fail safe
        return len(values) - 1

    def start(self):
        self.env.start()

    def run_iteration(self):
        x : torch.Tensor = get_screen(self.env.render(), self.device)
        goal : torch.Tensor = get_screen(self.env.render_target(), self.device)

        (policy, value) = self.scene_network.forward(self.shared_network.forward())

        pass
