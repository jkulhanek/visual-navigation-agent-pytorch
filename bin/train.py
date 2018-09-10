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

parser.add_argument('--grad_norm', type = float, default=40.0,
    help='gradient norm clip (default: 40.0)')

parser.add_argument('--learning_rate', type = float, default= 7 * 10e-4)
parser.add_argument('--rmsp_alpha', type = float, default = 0.99,
    help='decay parameter for RMSProp optimizer (default: 0.99)')
parser.add_argument('--rmsp_epsilon', type = float, default = 0.1,
    help='epsilon parameter for RMSProp optimizer (default: 0.99)')


args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t = Training(device, args.batch_size, entropy_beta = args.entropy_beta)
t.run()



