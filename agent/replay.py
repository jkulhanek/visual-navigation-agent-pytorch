from collections import namedtuple
import random

Sample = namedtuple('Transition',
                        ('state', 'action', 'value', 'policy', 'target', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, sample : Sample):
        """Saves a sample."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = Sample(sample)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)