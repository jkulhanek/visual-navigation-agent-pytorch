from resnet import ResNet, resnet50
from .network import SharedNetwork, SceneSpecificNetwork
from training_thread import TrainingThread
from .replay import ReplayMemory
from agent.optim import SharedRMSprop

class Training:
    def __init__(self, device, batch_size, entropy_beta : float):
        self.batch_size = batch_size
        self.device = device
        self.entropy_beta : float = entropy_beta
        self.optimizer = SharedRMSprop()

    def run(self):
        print("Training started")
        self.print_parameters()

        # Shared network
        self.shared_net = SharedNetwork()

        # Scene specific networks
        


        thread = TrainingThread(
            env = 
            device = self.device,
            scene_networks = scene_specific_nets,
            shared_network = shared_net,
            resnet = resnet)

    def print_parameters(self):
        print(f"- batch size: {self.batch_size}")
    

