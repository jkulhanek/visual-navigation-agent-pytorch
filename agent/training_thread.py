from network import SceneSpecificNetwork, SharedNetwork, ActorCriticLoss
from environment import AI2ThorEnvironment
import torch.nn as nn
from typing import Dict, Collection
import random
import torch
from input import get_screen
from resnet import ResNet
from replay import ReplayMemory, Sample
from collections import namedtuple
import torch.multiprocessing as mp
import numpy as np

TrainingSample = namedtuple('TrainingSample', ('state', 'policy', 'value', 'action_taken', 'goal', 'R', 'temporary_difference'))

class TrainingThread(mp.Process):
    def __init__(self,
            id : int,
            device : torch.device,
            shared_network : SharedNetwork, 
            scene : str,
            entropy_beta : float,
            max_t: int):

        super(TrainingThread, self).__init__(name = "process_{id}")

        self.policy_network = None
        self.device = device
        self.shared_network = shared_network
        self.scene_networks = scene_networks
        self.resnet = resnet
        self.master = master
        self.id = id
        self.max_t = max_t
        self.local_t = 0
        self.action_space_size = self.get_action_space_size()

        if not scene in scene_networks:
            self.scene_networks[scene] = SceneSpecificNetwork(self.action_space_size)
        
        self.scene_network : SceneSpecificNetwork = self.scene_networks[scene]
        self.criterion = ActorCriticLoss(entropy_beta)
        self.policy_network = nn.Sequential(self.shared_network, self.scene_network)
        self.memory = ReplayMemory()

        # Initialize the environment
        self.env = AI2ThorEnvironment(scene)

        # Initialize the episode
        self._reset_episode()

        self._sync_network()
    
    def _sync_network(self):
        self.shared_network.load_state_dict(self.master.shared_network.state_dict())

    def _ensure_shared_grads(self, model, shared_model):
        for param, shared_param in zip(model.parameters(), shared_model.parameters()):
            if shared_param.grad is not None:
                return 
            shared_param._grad = param.grad 
    
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

    def _reset_episode():
        self.episode_reward = 0
        self.episode_length = 0
        self.episode_max_q = -np.inf
        self.env.reset()


    def _simulate(self, memory_buffer : Collection[Sample]):
        # Does the evaluation end naturally?
        is_terminal = False

        # Plays out one game to end or max_t
        for t in range(self.max_t):
            x : torch.Tensor = get_screen(self.env.render(), self.device)
            goal : torch.Tensor = get_screen(self.env.render_target(), self.device)
            x_processed = self.resnet.forward(x)
            goal_processed = self.resnet.forward(goal)

            (policy, value) = self.policy_network(x_processed, goal_processed)
            action = self.choose_action(policy)

            # Makes the step in the environment
            self.env.step(action)

            # Receives the game reward
            is_terminal = self.env.is_terminal()

            # ad-hoc reward for navigation
            reward = 10.0 if terminal else -0.01

            # Max episode length
            if self.episode_length > 5e3: is_terminal = True

            # Update episode stats
            self.episode_length += 1
            self.episode_reward += reward
            self.episode_max_q = max(self.episode_max_q, np.max(value))

            # clip reward
            reward = np.clip(reward, -1, 1)

            memory_buffer.push(Sample(x_processed, action, value, goal_processed, reward))

            # Increase local time
            self.local_t += 1

            if is_terminal:
                # TODO: add logging
                terminal_end = True
                self._reset_episode()
                break

        if terminal_end:
            return 0
        else:
            x : torch.Tensor = get_screen(self.env.render(), self.device)
            goal : torch.Tensor = get_screen(self.env.render_target(), self.device)
            x_processed = self.resnet.forward(x)
            goal_processed = self.resnet.forward(goal)

            (_, value) = self.policy_network(x_processed, goal_processed)
            return value

    def _optimize(self, batch : TrainingSample):
        (state, policy, value, action_taken, target, playout_reward, temporary_difference) = batch
        _, _ = self.policy_network.forward(state, target)
        loss = self.criterion.forward(policy, value, action_taken, temporary_difference, playout_reward)
       
        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pass

    def _prepare_batches(self, memory_buffer : Collection[Sample], playout_reward: float):
        memory_buffer.reverse()
        batches : Collection[TrainingSample] = []
        for (state, action, value, target, reward) in memory_buffer:
            playout_reward = reward + self.gamma * playout_reward
            temporary_difference = playout_reward - value

            action_taken = np.zeros([ACTION_SIZE])
            action_taken[action] = 1

            batches.append(TrainingSample(state, policy, value, action_taken, target, playout_reward, temporary_difference))


        return TrainingSample(*zip(*batches))

    def run(self):
        # Initialize samples memory
        memory_buffer : Collection[Sample] = []
        
        # Plays some samples
        with torch.no_grad():
            playout_reward : float = self._simulate(memory_buffer)

            training_batches = self._prepare_batches(memory_buffer, playout_reward)

        # Train on collected samples
        self._optimize(training_batches)
        pass

if __name__ == '__main__':
    from network import SharedNetwork, SceneSpecificNetwork

    thread = TrainingThread(
        id = 1,
        device = torch.device('cpu'),
        shared_network = SharedNetwork(),
        scene = 'bedroom_04',
        entropy_beta = 0.2,
        max_t = 5
    )