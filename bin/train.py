#!/usr/bin/env python
import torch
import argparse

from agent.training import Training

argparse.ArgumentParser(description="")
parser = argparse.ArgumentParser(description='Deep reactive agent.')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size (default: 128)')
parser.add_argument('--entropy_beta', type=float, default=0.01,
                    help='entropy beta (default: 0.01)')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t = Training(device, args.batch_size, entropy_beta = args.entropy_beta)
t.run()



