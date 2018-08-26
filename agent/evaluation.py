

from .network import SharedNetwork, SceneSpecificNetwork
import torch.nn.functional as F

def evaluate_network(x, goal, action_space_size):
    shared_net = SharedNetwork()
    scene_net = SceneSpecificNetwork(action_space_size)

    (policy, action) = scene_net.forward(shared_net.forward(x, goal))
    policy = F.softmax(policy)

    