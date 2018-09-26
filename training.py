from agent.network import SharedNetwork, SceneSpecificNetwork
from agent.training_thread import TrainingThread
from agent.optim import SharedRMSprop
from typing import Collection, List
import torch.nn as nn
import torch.multiprocessing as mp 
import logging
import sys
import torch
import os
import threading

TASK_LIST = {
  'bathroom_02': ['26', '37', '43', '53', '69'],
  #'bedroom_04': ['134', '264', '320', '384', '387'],
  #'kitchen_02': ['90', '136', '157', '207', '329'],
  #'living_room_08': ['92', '135', '193', '228', '254']
}

class TrainingOptimizer:
    def __init__(self, grad_norm, optimizer, named_parameters):
        self.optimizer : torch.optim.Optimizer = optimizer
        self.named_parameters = named_parameters
        self.grad_norm = grad_norm
        self.lock = threading.Lock()

    def optimize(self, loss, local_params, shared_params):
        local_params = list(local_params)
        shared_params = list(shared_params)
        
        with self.lock:
            self.optimizer.zero_grad()

            # We will set the local gradient to 0
            for param in local_params:
                if not param.grad is None:
                    param.grad.data.zero_()

            # Calculate the new gradient with the respect to the local network
            loss.backward()

            # Clip gradient
            torch.nn.utils.clip_grad_norm_(local_params, self.grad_norm)
            
            # Ensure optimization occurs on the local gradients
            for (shared_param, local_param) in zip(shared_params, local_params):
                if shared_param.grad is None:
                    shared_param.grad = local_param.grad.clone()
                else:
                    shared_param.grad.copy_(local_param.grad)            
            
            self.optimizer.step()

class Training:
    def __init__(self, device, config):
        self.device = device
        self.config = config
        self.logger : logging.Logger = self._init_logger()
        self.learning_rate = config.get('learning_rate')
        self.rmsp_alpha = config.get('rmsp_alpha')
        self.rmsp_epsilon = config.get('rmsp_epsilon')
        self.grad_norm = config.get('grad_norm', 40.0)

        # Shared network
        self.shared_network = SharedNetwork()
        self.scene_networks = { key:SceneSpecificNetwork(4) for key in TASK_LIST.keys() }

        # Share memory
        self.shared_network.share_memory()
        for net in self.scene_networks.values():
            net.share_memory()

    def run(self):
        print("Training started")
        self.print_parameters()

        # Callect all parameters from all networks
        parameters = list(self.shared_network.parameters())
        for net in self.scene_networks.values():
            parameters.extend(net.parameters())

        # Create optimizer
        optimizer = SharedRMSprop(parameters,  eps=0.1, alpha=0.99, lr=0.0007001643593729748) # lr = self.learning_rate, alpha = self.rmsp_alpha, eps = self.rmsp_epsilon)
        optimizer.share_memory()
        optimizer_wrapper = TrainingOptimizer(self.grad_norm, optimizer, parameters)
        self.optimizer = optimizer_wrapper

        # Prepare threads
        branches = [(scene, int(target)) for scene in TASK_LIST.keys() for target in TASK_LIST.get(scene)]
        def _createThread(id, task):
            (scene, target) = task
            net = nn.Sequential(self.shared_network, self.scene_networks[scene])
            net.share_memory()
            return TrainingThread(
                id = id,
                device = self.device,
                master = self,
                network = net,
                scene = scene,
                logger = self.logger,
                terminal_state_id = target,
                **self.config)

        self.threads = [_createThread(i, task) for i, task in enumerate(branches)]
        self.threads[0].start()
        self.threads[0].join()

        for thread in self.threads:
            thread.start()

        for thread in self.threads:
            thread.join()
        

    def _init_logger(self):
        logger = logging.getLogger('agent')
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.StreamHandler(sys.stdout))
        return logger

    def print_parameters(self):
        self.logger.info(f"- batch size: {self.config.get('batch_size')}")
        self.logger.info(f"- gamma: {self.config.get('gamma')}")
        self.logger.info(f"- learning rate: {self.config.get('learning_rate')}")

if __name__ == "__main__":
    mp.set_start_method('spawn')
    training = Training(torch.device('cpu'), {
        'learning_rate': 7 * 10e4,
        'rmsp_alpha': 0.99,
        'rmsp_epsilon': 0.1,
        'h5_file_path': (lambda scene: f"/mnt/d/datasets/visual_navigation_precomputed/{scene}.h5")
    })

    import pickle
    shared_net = SharedNetwork()
    scene_nets = { key:SceneSpecificNetwork(4) for key in TASK_LIST.keys() }

    # Load weights trained on tensorflow
    data = pickle.load(open(os.path.normpath(os.path.join(__file__, '../weights.p')), 'rb'), encoding='latin1')
    def convertToStateDict(data):
        return {key:torch.Tensor(v) for (key, v) in data.items()}

    training.shared_network.load_state_dict(convertToStateDict(data['navigation']))
    for key in TASK_LIST.keys():
        training.scene_networks[key].load_state_dict(convertToStateDict(data[f'navigation/{key}']))

    training.run()