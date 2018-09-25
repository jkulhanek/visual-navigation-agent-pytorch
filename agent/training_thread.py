from agent.network import SceneSpecificNetwork, SharedNetwork, ActorCriticLoss
from agent.environment import Environment, THORDiscreteEnvironment
import torch.nn as nn
from typing import Dict, Collection
import random
import torch
from agent.replay import ReplayMemory, Sample
from collections import namedtuple
import torch.multiprocessing as mp
import torch.nn.functional as F
import numpy as np
import logging

TrainingSample = namedtuple('TrainingSample', ('state', 'policy', 'value', 'action_taken', 'goal', 'R', 'temporary_difference'))





class TrainingThread(mp.Process):
    def __init__(self,
            id : int,
            device : torch.device, 
            network : torch.nn.Module,
            master,
            logger,
            scene : str,
            **kwargs):

        super(TrainingThread, self).__init__(name = "process_{id}")

        # Initialize the environment
        self.env = THORDiscreteEnvironment(scene, **kwargs)
        self.device = device
        self.local_backbone_network = SharedNetwork()
        self.master = master

        self.gamma : float= kwargs.get('gamma', 0.99)
        self.grad_norm: float = kwargs.get('grad_norm', 40.0)
        entropy_beta : float = kwargs.get('entropy_beta', 0.01)
        self.max_t : int = kwargs.get('max_t', 1) # TODO: 5)

        self.logger : logging.Logger = logger
        self.local_t = 0
        self.action_space_size = self.get_action_space_size()

        self.master_network = network
        self.scene_network : SceneSpecificNetwork = SceneSpecificNetwork(self.get_action_space_size())
        self.criterion = ActorCriticLoss(entropy_beta)
        self.policy_network = nn.Sequential(self.local_backbone_network, self.scene_network)

        self.master.optimizer = self.master.createOptimizer(self.policy_network.parameters())

        import torch.optim as optim
        optimizer = optim.RMSprop(self.policy_network.parameters(), eps=0.1, alpha=0.99, lr=0.0007001643593729748)
        
        # Initialize the episode
        self._reset_episode()
        self._sync_network()
    
    def _sync_network(self):
        self.policy_network.load_state_dict(self.master_network.state_dict())

    def _ensure_shared_grads(self):
        for param, shared_param in zip(self.policy_network.parameters(), self.master_network.parameters()):
            if shared_param.grad is not None:
                return 
            shared_param._grad = param.grad 
    
    def get_action_space_size(self):
        return len(self.env.actions)

    def start(self):
        self.env.start()

    def _reset_episode(self):
        self.episode_reward = 0
        self.episode_length = 0
        self.episode_max_q = -np.inf
        self.env.reset()

    def _forward_explore(self):
        # Does the evaluation end naturally?
        is_terminal = False
        terminal_end = False

        results = { "policy":[], "value": []}
        rollout_path = {"state": [], "action": [], "rewards": [], "done": []}

        # Plays out one game to end or max_t
        for t in range(self.max_t):
            state = { 
                "current": self.env.render('resnet_features'),
                "goal": self.env.render_target('resnet_features'),
            }

            x_processed = torch.from_numpy(state["current"])
            goal_processed = torch.from_numpy(state["goal"])

            (policy, value) = self.policy_network((x_processed, goal_processed,))

            # Store raw network output to use in backprop
            results["policy"].append(policy)
            results["value"].append(value)

            with torch.no_grad():
                (_, action,) = policy.max(0)
                action = action[0]
                # action = F.softmax(policy, dim=0).multinomial(1).data.numpy()[0]
            
            policy = policy.data.numpy()
            value = value.data.numpy()
            
            

            # Makes the step in the environment
            self.env.step(action)

            # Receives the game reward
            is_terminal = self.env.is_terminal

            # ad-hoc reward for navigation
            reward = 10.0 if is_terminal else -0.01

            # Max episode length
            if self.episode_length > 5e3: is_terminal = True

            # Update episode stats
            self.episode_length += 1
            self.episode_reward += reward
            self.episode_max_q = max(self.episode_max_q, np.max(value))

            # clip reward
            reward = np.clip(reward, -1, 1)

            # Increase local time
            self.local_t += 1

            rollout_path["state"].append(state)
            rollout_path["action"].append(action)
            rollout_path["rewards"].append(reward)
            rollout_path["done"].append(is_terminal) 

            if is_terminal:
                # TODO: add logging
                self.logger.info('playout finished')
                self.logger.info(f'episode length: {self.episode_length}')
                self.logger.info(f'episode reward: {self.episode_reward}')
                self.logger.info(f'episode max_q: {self.episode_max_q}')

                terminal_end = True
                self._reset_episode()
                break

        if terminal_end:
            return np.array([0], dtype = np.float32), results, rollout_path
        else:
            x_processed = torch.from_numpy(self.env.render('resnet_features'))
            goal_processed = torch.from_numpy(self.env.render_target('resnet_features'))

            (_, value) = self.policy_network((x_processed, goal_processed,))
            return value.data.numpy(), results, rollout_path
    
    def _optimize_path(self, playout_reward: float, results, rollout_path):
        policy_batch = []
        value_batch = []
        action_batch = []
        temporary_difference_batch = []
        playout_reward_batch = []


        for i in reversed(range(len(results["value"]))):
            reward = rollout_path["rewards"][i]
            value = results["value"][i]
            action = rollout_path["action"][i]

            playout_reward = reward + self.gamma * playout_reward
            temporary_difference = playout_reward - value.data.numpy()

            policy_batch.append(results['policy'][i])
            value_batch.append(results['value'][i])
            action_batch.append(action)
            temporary_difference_batch.append(temporary_difference)
            playout_reward_batch.append(playout_reward)
        
        policy_batch = torch.stack(policy_batch, 0)
        value_batch = torch.stack(value_batch, 0)
        action_batch = torch.from_numpy(np.array(action_batch))
        temporary_difference_batch = torch.from_numpy(np.array(temporary_difference_batch))
        playout_reward_batch = torch.from_numpy(np.array(playout_reward_batch))
        
        # Compute loss
        loss = self.criterion.forward(policy_batch, value_batch, action_batch, temporary_difference_batch, playout_reward_batch)
        loss = loss.sum()

        loss_value = loss.detach().numpy()
        self.master.optimizer.optimize(loss, 
            self.policy_network.parameters(), 
            self.master_network.parameters())

    def run(self):
        self.env.reset()
        self._sync_network()
        while True:
            self._sync_network()
            # Plays some samples
            playout_reward, results, rollout_path = self._forward_explore()
            # Train on collected samples
            self._optimize_path(playout_reward, results, rollout_path)
            pass

if __name__ == '__main__':
    from agent.network import SharedNetwork, SceneSpecificNetwork
    import sys
    import pickle

    model_data = pickle.load(open('D:\\models\\visual-navigation\\weights.p', 'rb'))


    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    thread = TrainingThread(
        id = 1,
        device = torch.device('cpu'),
        shared_network = SharedNetwork(),
        scene = 'bedroom_04',
        entropy_beta = 0.2,
        logger = logger,
        max_t = 5,
        terminal_state_id = 26,
        h5_file_path = 'D:\\datasets\\visual_navigation_precomputed\\bathroom_02.h5'
    )

    print('Loaded')
    thread.run()