#!/usr/bin/env python
import torch
import argparse
import multiprocessing as mp

from agent.training import Training

if __name__ == '__main__':
    # mp.set_start_method('spawn')
    argparse.ArgumentParser(description="")
    parser = argparse.ArgumentParser(description='Deep reactive agent.')
    parser.add_argument('--entropy_beta', type=float, default=0.01,
                        help='entropy beta (default: 0.01)')

    parser.add_argument('--restore', action='store_true', help='restore from checkpoint')
    parser.add_argument('--grad_norm', type = float, default=40.0,
        help='gradient norm clip (default: 40.0)')

    parser.add_argument('--h5_file_path', type = str, default='/app/data/{scene}.h5')
    parser.add_argument('--checkpoint_path', type = str, default='/model/checkpoint-{checkpoint}.pth')

    parser.add_argument('--learning_rate', type = float, default= 0.0007001643593729748)
    parser.add_argument('--rmsp_alpha', type = float, default = 0.99,
        help='decay parameter for RMSProp optimizer (default: 0.99)')
    parser.add_argument('--rmsp_epsilon', type = float, default = 0.1,
        help='epsilon parameter for RMSProp optimizer (default: 0.1)')


    args = vars(parser.parse_args())

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')

    if args['restore']:
        t = Training.load_checkpoint(args)
    else:
        t = Training(device, args)

    t.run()



