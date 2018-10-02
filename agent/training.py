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
from contextlib import suppress
import re

TOTAL_PROCESSED_FRAMES = 20 * 10**6 # 10 million frames
TASK_LIST = {
  'bathroom_02': ['26', '37', '43', '53', '69'],
  'bedroom_04': ['134', '264', '320', '384', '387'],
  'kitchen_02': ['90', '136', '157', '207', '329'],
  'living_room_08': ['92', '135', '193', '228', '254']
}

class TrainingSaver:
    def __init__(self, shared_network, scene_networks, optimizer, config):
        self.checkpoint_path = config.get('checkpoint_path', 'model/checkpoint-{checkpoint}.pth')
        self.saving_period = config.get('saving_period', 10 ** 6)
        self.shared_network = shared_network
        self.scene_networks = scene_networks
        self.optimizer = optimizer
        self.config = config

    def after_optimization(self):
        iteration = self.optimizer.get_global_step()
        if iteration % self.saving_period == 0:
            self.save()

    def save(self):
        iteration = self.optimizer.get_global_step()
        filename = self.checkpoint_path.replace('{checkpoint}', str(iteration))
        model = dict()
        model['navigation'] = self.shared_network.state_dict()
        for key, val in self.scene_networks.items():
            model[f'navigation/{key}'] = val.state_dict()
        model['optimizer'] = self.optimizer.state_dict()
        model['config'] = self.config
        
        with suppress(FileExistsError):
            os.makedirs(os.path.dirname(filename))
        torch.save(model, open(filename, 'wb'))

    def restore(self, state):
        if 'optimizer' in state: self.optimizer.load_state_dict(state['optimizer'])
        if 'config' in state: self.config = state['config']
        self.shared_network.load_state_dict(state['navigation'])

        tasks = state['config'].get('tasks', TASK_LIST)
        for scene in tasks.keys():
            self.scene_networks[scene].load_state_dict(state[f'navigation/{scene}'])

class TrainingOptimizer:
    def __init__(self, grad_norm, optimizer, scheduler):
        self.optimizer : torch.optim.Optimizer = optimizer
        self.scheduler = scheduler
        self.grad_norm = grad_norm
        self.global_step = torch.tensor(0)
        self.lock = mp.Lock()

    def state_dict(self):
        state_dict = dict()
        state_dict['optimizer'] = self.optimizer.state_dict()
        state_dict['scheduler'] = self.scheduler.state_dict()
        state_dict["global_step"] = self.global_step
        return state_dict

    def share_memory(self):
        self.global_step.share_memory_()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])
        self.global_step.copy_(state_dict['global_step'])
    
    def get_global_step(self):
        return self.global_step.item()

    def optimize(self, loss, local_params, shared_params):
        local_params = list(local_params)
        shared_params = list(shared_params)

        # Fix the optimizer property after unpickling
        self.scheduler.optimizer = self.optimizer

        with self.lock:
            self.scheduler.step(self.global_step.item())

            # Increment step
            self.global_step.copy_(torch.tensor(self.global_step.item() + 1))
            
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

class AnnealingLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, total_epochs, last_epoch=-1):
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.total_epochs = total_epochs
        super(AnnealingLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * (1.0 - self.last_epoch / self.total_epochs)
                for base_lr in self.base_lrs]

class Training:
    def __init__(self, device, config):
        self.device = device
        self.config = config
        self.logger : logging.Logger = self._init_logger()
        self.learning_rate = config.get('learning_rate')
        self.rmsp_alpha = config.get('rmsp_alpha')
        self.rmsp_epsilon = config.get('rmsp_epsilon')
        self.grad_norm = config.get('grad_norm', 40.0)
        self.tasks = config.get('tasks', TASK_LIST)
        self.checkpoint_path = config.get('checkpoint_path', 'model/checkpoint-{checkpoint}.pth')
        self.max_t = config.get('max_t', 5)
        self.total_epochs = TOTAL_PROCESSED_FRAMES // self.max_t
        self.initialize()

    @staticmethod
    def load_checkpoint(config, fail = True):
        device = torch.device('cpu')
        checkpoint_path = config.get('checkpoint_path', 'model/checkpoint-{checkpoint}.pth')
        max_t = config.get('max_t', 5)
        total_epochs = TOTAL_PROCESSED_FRAMES // max_t
        files = os.listdir(os.path.dirname(checkpoint_path))
        base_name = os.path.basename(checkpoint_path)
        
        # Find latest checkpoint
        # TODO: improve speed
        restore_point = None
        if base_name.find('{checkpoint}') != -1:
            regex = re.escape(base_name).replace(re.escape('{checkpoint}'), '(\d+)')
            points = [(fname, int(match.group(1))) for (fname, match) in ((fname, re.match(regex, fname),) for fname in files) if not match is None]
            if len(points) == 0:
                if fail:
                    raise Exception('Restore point not found')
                else: return None
            
            (base_name, restore_point) = max(points, key = lambda x: x[1])

            
        print(f'Restoring from checkpoint {restore_point}')
        state = torch.load(open(os.path.join(os.path.dirname(checkpoint_path), base_name), 'rb'))
        training = Training(device, state['config'] if 'config' in state else config)
        training.saver.restore(state)        
        return training

    def initialize(self):
        # Shared network
        self.shared_network = SharedNetwork()
        self.scene_networks = { key:SceneSpecificNetwork(4) for key in TASK_LIST.keys() }

        # Share memory
        self.shared_network.share_memory()
        for net in self.scene_networks.values():
            net.share_memory()

        # Callect all parameters from all networks
        parameters = list(self.shared_network.parameters())
        for net in self.scene_networks.values():
            parameters.extend(net.parameters())

        # Create optimizer
        optimizer = SharedRMSprop(parameters, eps=self.rmsp_epsilon, alpha=self.rmsp_alpha, lr=self.learning_rate)
        optimizer.share_memory()

        # Create scheduler
        scheduler = AnnealingLRScheduler(optimizer, self.total_epochs)

        # Create optimizer wrapper
        optimizer_wrapper = TrainingOptimizer(self.grad_norm, optimizer, scheduler)
        self.optimizer = optimizer_wrapper
        optimizer_wrapper.share_memory()

        # Initialize saver
        self.saver = TrainingSaver(self.shared_network, self.scene_networks, self.optimizer, self.config)
    
    def run(self):
        self.logger.info("Training started")
        self.print_parameters()

        # Prepare threads
        branches = [(scene, int(target)) for scene in TASK_LIST.keys() for target in TASK_LIST.get(scene)]
        def _createThread(id, task):
            (scene, target) = task
            net = nn.Sequential(self.shared_network, self.scene_networks[scene])
            net.share_memory()
            return TrainingThread(
                id = id,
                optimizer = self.optimizer,
                network = net,
                scene = scene,
                saver = self.saver,
                max_t = self.max_t,
                terminal_state_id = target,
                **self.config)

        self.threads = [_createThread(i, task) for i, task in enumerate(branches)]
        
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
        self.logger.info(f"- gamma: {self.config.get('gamma')}")
        self.logger.info(f"- learning rate: {self.config.get('learning_rate')}")

if __name__ == "__main__":
    mp.set_start_method('spawn')

    #training = Training.load_checkpoint(dict())
    training = Training(torch.device('cpu'), {
        'learning_rate': 0.0007001643593729748,
        'rmsp_alpha': 0.99,
        'rmsp_epsilon': 0.1,
        'h5_file_path': "D:\\datasets\\visual_navigation_precomputed\\{scene}.h5"
    })

    import pickle
    shared_net = SharedNetwork()
    scene_nets = { key:SceneSpecificNetwork(4) for key in TASK_LIST.keys() }

    # Load weights trained on tensorflow
    data = pickle.load(open(os.path.normpath(os.path.join(__file__, '..\\weights.p')), 'rb'), encoding='latin1')
    def convertToStateDict(data):
        return {key:torch.Tensor(v) for (key, v) in data.items()}

    training.shared_network.load_state_dict(convertToStateDict(data['navigation']))
    for key in TASK_LIST.keys():
        training.scene_networks[key].load_state_dict(convertToStateDict(data[f'navigation/{key}']))

    training.run()