
from diffmot import DiffMOT
import argparse
import yaml
import torch
from easydict import EasyDict

def parse_args():
    parser = argparse.ArgumentParser(
        description='Pytorch implementation of MID')
    parser.add_argument('--config', default='', help='Path to the config file')
    parser.add_argument('--dataset', default='', help='Dataset name')
    parser.add_argument('--network', choices=['ReUNet', 'ReUNet+++', 'Smaller', 'ReUNet2Plus', 'SimpleReUNet2Plus', 'SimpleReUNet2Plus1', 'SimpleReUNet2Plus2'], help='Unet version')
    parser.add_argument('--data_dir', default=None, help='Path to the data directory')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.config) as f:
       config = yaml.safe_load(f)

    for k, v in vars(args).items():
       if v is not None:
           config[k] = v
    config["exp_name"] = args.config.split("/")[-1].split(".")[0]
    config["dataset"] = args.dataset
    config = EasyDict(config)
    
    # Update config with command-line arguments if provided
    if args.data_dir is not None:
        config.data_dir = args.data_dir
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.network is not None:
        config.network = args.network

    agent = DiffMOT(config)
    
    agent.half()  # Convert all parameters and buffers to Half

    with torch.amp.autocast("cuda"):  # Updated from torch.cuda.amp.autocast
        if config.eval_mode:
            agent.eval()
        else:
            agent.train()

if __name__ == '__main__':
    main()
