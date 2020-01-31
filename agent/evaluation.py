

from agent.network import SharedNetwork, SceneSpecificNetwork
from agent.environment import THORDiscreteEnvironment
from agent.training import TrainingSaver, TOTAL_PROCESSED_FRAMES
from agent.utils import find_restore_point
import torch.nn.functional as F
import torch
import pickle
import os
import numpy as np
import re
from itertools import groupby

TASK_LIST = {
  'bathroom_02': ['26', '37', '43', '53', '69'],
  'bedroom_04': ['134', '264', '320', '384', '387'],
  'kitchen_02': ['90', '136', '157', '207', '329'],
  'living_room_08': ['92', '135', '193', '228', '254']
}

ACTION_SPACE_SIZE = 4
NUM_EVAL_EPISODES = 100
VERBOSE = False

def export_to_csv(data, file):
    import csv
    with open(file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        for k, g in groupby(sorted(data, key = lambda x: x[0]), key = lambda x: x[0]):
            g = list(g)
            header = [k, '']
            header.extend((np.mean(a) for a in list(zip(*g))[2:]))
            writer.writerow(header)
            for item in g:
                writer.writerow(list(item))
    print(f'CSV file stored "{file}"')

class Evaluation:
    def __init__(self, config):
        self.config = config
        self.device = config.get('device', torch.device('cpu'))
        self.shared_net = SharedNetwork().to(self.device)
        self.scene_nets = { key:SceneSpecificNetwork(ACTION_SPACE_SIZE).to(self.device) for key in TASK_LIST.keys() }

    @staticmethod
    def load_checkpoint(config, fail = True):
        checkpoint_path = config.get('checkpoint_path', 'model/checkpoint-{checkpoint}.pth')
        
        (base_name, restore_point) = find_restore_point(checkpoint_path, fail)
        print(f'Restoring from checkpoint {restore_point}')
        state = torch.load(open(os.path.join(os.path.dirname(checkpoint_path), base_name), 'rb'))
        evaluation = Evaluation(config)
        saver = TrainingSaver(evaluation.shared_net, evaluation.scene_nets, None, evaluation.config)
        print('Configuration')
        saver.restore(state)
        saver.print_config(offset = 4)            
        return evaluation

    def build_agent(self, scene_name):
        parent = self
        net = torch.nn.Sequential(parent.shared_net, parent.scene_nets[scene_name])
        class Agent:
            def __init__(self, initial_state, target):
                self.env = THORDiscreteEnvironment(
                    scene_name=scene_name,
                    initial_state_id = initial_state,
                    terminal_state_id = target,
                    h5_file_path=(lambda scene: parent.config["h5_file_path"].replace("{scene}", scene_name))
                )

                self.env.reset()
                self.net = net

            @staticmethod
            def get_parameters():
                return net.parameters()

            def act(self):
                with torch.no_grad():
                    state = torch.Tensor(self.env.render(mode='resnet_features')).to(parent.device)
                    target = torch.Tensor(self.env.render_target(mode='resnet_features')).to(parent.device)
                    (policy, value,) = net.forward((state, target,))
                    action = F.softmax(policy, dim=0).multinomial(1).cpu().data.numpy()[0]

                self.env.step(action)
                return (self.env.is_terminal, self.env.collided, self.env.reward)
        return Agent
        
    
    def run(self):
        scene_stats = dict()
        resultData = []
        for scene_scope, items in TASK_LIST.items():
            if len(self.config['test_scenes']) != 0 and not scene_scope in self.config['test_scenes']:
                continue

            scene_net = self.scene_nets[scene_scope]
            scene_stats[scene_scope] = list()
            for task_scope in items:
                env = THORDiscreteEnvironment(
                    scene_name=scene_scope,
                    h5_file_path=(lambda scene: self.config.get("h5_file_path", "D:\\datasets\\visual_navigation_precomputed\\{scene}.h5").replace('{scene}', scene)),
                    terminal_state_id=int(task_scope),
                )

                graph = env._get_graph_handle()
                hitting_times = graph['hitting_times'][()]
                shortest_paths = graph['shortest_path_distance'][()]

                ep_rewards = []
                ep_lengths = []
                ep_collisions = []
                ep_normalized_lengths = []
                for (i_episode, start) in enumerate(env.get_initial_states(int(task_scope))):
                    env.reset(initial_state_id = start)
                    terminal = False
                    ep_reward = 0
                    ep_collision = 0
                    ep_t = 0
                    hitting_time = hitting_times[start, int(task_scope)]
                    shortest_path = shortest_paths[start, int(task_scope)]

                    while not terminal:
                        state = torch.Tensor(env.render(mode='resnet_features'))
                        target = torch.Tensor(env.render_target(mode='resnet_features'))
                        (policy, value,) = scene_net.forward(self.shared_net.forward((state, target,)))

                        with torch.no_grad():
                            action = F.softmax(policy, dim=0).multinomial(1).data.numpy()[0]
                        env.step(action)
                        terminal = env.is_terminal

                        if ep_t == hitting_time: break
                        if env.collided: ep_collision += 1
                        ep_reward += env.reward
                        ep_t += 1                   


                    ep_lengths.append(ep_t)
                    ep_rewards.append(ep_reward)
                    ep_collisions.append(ep_collision)
                    ep_normalized_lengths.append(min(ep_t, hitting_time) / shortest_path)
                    if VERBOSE: print("episode #{} ends after {} steps".format(i_episode, ep_t))

                    
                print('evaluation: %s %s' % (scene_scope, task_scope))
                print('mean episode reward: %.2f' % np.mean(ep_rewards))
                print('mean episode length: %.2f' % np.mean(ep_lengths))
                print('mean episode collision: %.2f' % np.mean(ep_collisions))
                print('mean normalized episode length: %.2f' % np.mean(ep_normalized_lengths))
                scene_stats[scene_scope].extend(ep_lengths)
                resultData.append((scene_scope, str(task_scope), np.mean(ep_rewards), np.mean(ep_lengths), np.mean(ep_collisions), np.mean(ep_normalized_lengths),))

        print('\nResults (average trajectory length):')
        for scene_scope in scene_stats:
            print('%s: %.2f steps'%(scene_scope, np.mean(scene_stats[scene_scope])))
        
        if 'csv_file' in self.config and self.config['csv_file'] is not None:
            export_to_csv(resultData, self.config['csv_file'])

'''
# Load weights trained on tensorflow
data = pickle.load(open(os.path.join(__file__, '..\\..\\weights.p'), 'rb'), encoding='latin1')
def convertToStateDict(data):
    return {key:torch.Tensor(v) for (key, v) in data.items()}

shared_net.load_state_dict(convertToStateDict(data['navigation']))
for key in TASK_LIST.keys():
    scene_nets[key].load_state_dict(convertToStateDict(data[f'navigation/{key}']))'''