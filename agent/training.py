from agent.network import SharedNetwork, SceneSpecificNetwork
from agent.training_thread import TrainingThread
from agent.optim import SharedRMSprop
from typing import Collection, List
import torch.multiprocessing as mp 
import logging
import sys
import torch

TASK_LIST = {
  'bathroom_02': ['26', '37', '43', '53', '69'],
  'bedroom_04': ['134', '264', '320', '384', '387'],
  'kitchen_02': ['90', '136', '157', '207', '329'],
  'living_room_08': ['92', '135', '193', '228', '254']

}

class Training:
    def __init__(self, device, config):
        self.device = device
        self.config = config
        self.logger : logging.Logger = self._init_logger()
        self.learning_rate = config.get('learning_rate')
        self.rmsp_alpha = config.get('rmsp_alpha')
        self.rmsp_epsilon = config.get('rmsp_epsilon')

    def run(self):
        print("Training started")
        self.print_parameters()

        # Shared network
        self.shared_network = SharedNetwork()
        self.shared_network.share_memory()

        branches = [(scene, int(target)) for scene in TASK_LIST.keys() for target in TASK_LIST.get(scene)]

        def _createThread(id, task):
            (scene, target) = task
            return TrainingThread(
                id = id,
                device = self.device,
                master = self,
                scene = scene,
                logger = self.logger,
                terminal_state_id = target,
                **self.config)

        self.createOptimizer = lambda params: torch.optim.RMSprop(params, lr = self.learning_rate, alpha = self.rmsp_alpha, eps = self.rmsp_epsilon)
        threads = [_createThread(i, task) for i, task in enumerate(branches)]
        
        # Callect all parameters from all networks
        parameters = list(self.shared_network.parameters())
        for thread in threads:
            parameters.extend(thread.get_local_parameters())

        # Create optimizer
        self.optimizer = SharedRMSprop(parameters, lr = self.learning_rate, alpha = self.rmsp_alpha, eps = self.rmsp_epsilon)
        self.optimizer.share_memory()

        threads[0].run()
        # threads[0].join()

        return

        for thread in threads:
            thread.start()

        for thread in threads:
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
    training = Training(torch.device('cpu'), {
        'learning_rate': 7 * 10e4,
        'rmsp_alpha': 0.99,
        'rmsp_epsilon': 0.1,
        'h5_file_path': (lambda scene: f"D:\\datasets\\visual_navigation_precomputed\\{scene}.h5")
    })

    training.run()