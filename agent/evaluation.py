

from agent.network import SharedNetwork, SceneSpecificNetwork
from agent.environment import THORDiscreteEnvironment
import torch.nn.functional as F
import torch
import pickle
import os
import numpy as np

TASK_LIST = {
  'bathroom_02': ['26', '37', '43', '53', '69'],
  'bedroom_04': ['134', '264', '320', '384', '387'],
  'kitchen_02': ['90', '136', '157', '207', '329'],
  'living_room_08': ['92', '135', '193', '228', '254']
}

ACTION_SPACE_SIZE = 4
NUM_EVAL_EPISODES = 100
VERBOSE = False

shared_net = SharedNetwork()
scene_nets = { key:SceneSpecificNetwork(ACTION_SPACE_SIZE) for key in TASK_LIST.keys() }

# Load weights trained on tensorflow
data = pickle.load(open(os.path.join(__file__, '..\\..\\weights.p'), 'rb'), encoding='latin1')
def convertToStateDict(data):
    return {key:torch.Tensor(v) for (key, v) in data.items()}

shared_net.load_state_dict(convertToStateDict(data['navigation']))
for key in TASK_LIST.keys():
    scene_nets[key].load_state_dict(convertToStateDict(data[f'navigation/{key}']))

scene_stats = dict()
for scene_scope, items in TASK_LIST.items():
    scene_net = scene_nets[scene_scope]
    scene_stats[scene_scope] = list()
    for task_scope in items:
        env = THORDiscreteEnvironment(
            scene_name=scene_scope,
            h5_file_path=(lambda scene: f"D:\\datasets\\visual_navigation_precomputed\\{scene}.h5"),
            terminal_state_id=int(task_scope)
        )

        ep_rewards = []
        ep_lengths = []
        ep_collisions = []
        for i_episode in range(NUM_EVAL_EPISODES):
            env.reset()
            terminal = False
            ep_reward = 0
            ep_collision = 0
            ep_t = 0
            while not terminal:
                state = torch.Tensor(env.render(mode='resnet_features'))
                target = torch.Tensor(env.render_target(mode='resnet_features'))
                (policy, value,) = scene_net.forward(shared_net.forward((state, target,)))

                with torch.no_grad():
                    action = F.softmax(policy, dim=0).multinomial(1).data.numpy()[0]
                env.step(action)
                terminal = env.is_terminal

                if ep_t == 10000: break
                if env.collided: ep_collision += 1
                ep_reward += env.reward
                ep_t += 1

            ep_lengths.append(ep_t)
            ep_rewards.append(ep_reward)
            ep_collisions.append(ep_collision)
            if VERBOSE: print("episode #{} ends after {} steps".format(i_episode, ep_t))

        print('evaluation: %s %s' % (scene_scope, task_scope))
        print('mean episode reward: %.2f' % np.mean(ep_rewards))
        print('mean episode length: %.2f' % np.mean(ep_lengths))
        print('mean episode collision: %.2f' % np.mean(ep_collisions))
        scene_stats[scene_scope].extend(ep_lengths)

print('\nResults (average trajectory length):')
for scene_scope in scene_stats:
    print('%s: %.2f steps'%(scene_scope, np.mean(scene_stats[scene_scope])))