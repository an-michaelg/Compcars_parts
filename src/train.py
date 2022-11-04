# -*- coding: utf-8 -*-
from network import Network
from compcars_dataloader import get_cars_dataloader
import wandb
import yaml
import argparse

CONFIG_FILENAME = 'D:/Projects/Compcars/src/config.yaml'
HW_BATCH_LIMIT = 8 # always use a base2 number


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            prog = 'Launcher for network training',
            description = 'Run training/test with setting from config YAML file',
            epilog = 'Example use: -y config.yaml')

    parser.add_argument("-y", "--yaml", help="Path to the .yaml file")
    args = parser.parse_args()
    
    if args.yaml:
        config_fn = args.yaml
    else:
        config_fn = CONFIG_FILENAME
    
    with open(config_fn) as f:
        config = yaml.safe_load(f)
    
    if config['use_wandb'] and config['mode'] != 'test':
        run = wandb.init(project="compcars_mil", entity="guangnan", config=config)
    
    tax = config['taxonomy']
    if config['mode']=="train":
        bsize = min(config['batch_size'], HW_BATCH_LIMIT)
        dataloader_tr, classind_to_gt = get_cars_dataloader(bsize, split='train', taxonomy=tax)
    dataloader_te, classind_to_gt = get_cars_dataloader(batch_size=1, split='test', taxonomy=tax)
    
    # initialize the model and any loaded weights with Network
    net = Network(config, classind_to_gt, taxonomy=tax, batch_limit=HW_BATCH_LIMIT)
    
    if config['mode']=="train":
        net.train(dataloader_tr, dataloader_te)
    net.test_comprehensive(dataloader_te, mode="test")
        
    if config['use_wandb']:
        wandb.finish()