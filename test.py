import argparse
import configparser
import copy
import json
import logging
import os
import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from utils.dataloader import Getfile
from val import evaluate
from model.unet import UNet


parser = argparse.ArgumentParser(description='Testing the UNet')
parser.add_argument('--data_path', type=str,
                    default='../data2D', help='Name of Experiment')
parser.add_argument('--test_save_path', type=str,
                    default='predicatedimg',
                    help='Name of Experiment')
parser.add_argument('--checkpoint_path', type=str,
                    default='checkpoints',
                    help='Name of Experiment')
parser.add_argument('--batch_size', '-b', dest='batch_size', metavar='B', type=int, default=16, help='Batch size')
parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
parser.add_argument('--classes', '-c', type=int, default=5, help='Number of classes')
parser.add_argument('--patch_size', type=list, default=[96, 96, 96],
                    help='patch size of network input')
parser.add_argument('--train_mode', '-trm', type=str, default="ABCT", help='train mode')
parser.add_argument('--test_mode', '-tem', type=str, default="ABCT", help='test mode to run')
parser.add_argument('--gpu', '-g', type=int, default=0, help='gpu to run')
parser.add_argument('--checkpoint', nargs='+', type=int, default=[0], help='checkpoints to run')
parser.add_argument('--seed', type=int, default=8888, help='random seed')
args = parser.parse_args()



def test(args):
    data_path = args.data_path
    test_save_path = args.test_save_path
    batch_size = args.batch_size
    num_classes = args.classes
    patch_size = args.patch_size
    checkpoint_path = args.checkpoint_path  
    train_mode = args.train_mode
    test_mode = args.test_mode
    checkpoint_path = os.path.join(checkpoint_path, train_mode)
    checkpoint = args.checkpoint

    config = configparser.ConfigParser()
    print("Loading config.ini")
    config.read('config.ini') 
    test_dir_list = config.get(test_mode, 'test_dir')
    test_dir = test_dir_list.split(", ")
    label_intensities_str = config.get(test_mode, 'label_intensities')
    label_intensities = tuple(map(float, label_intensities_str.split(',')))
    class_to_pixel_str = config.get(test_mode, 'class_to_pixel')
    class_to_pixel = json.loads(class_to_pixel_str)

    test_loader = DataLoader(
        Getfile(base_dir=data_path, val_dir=test_dir[0], domain=1, num_classes=num_classes,
                label_intensities=label_intensities, mode=None, onehot=False, num_data=0, aug=False),
        batch_size=16, shuffle=False, num_workers=8)
    print('Loading data... ', test_dir[0])

    net = UNet(in_channels=1, out_channels=args.classes,use_projection=False)
    net.to(device=device)
    net.eval()

    print("Testing begin")
    for epoch in checkpoint:
        weight_path = os.path.join(checkpoint_path,'study_seed44_unsup_all/unet2D_best_model.pth')
        print('Loading:', weight_path.split('/')[2])
        state_dict = torch.load(weight_path, map_location='cpu')
        state_dict = remove_projection_head_weights(state_dict)
        net.load_state_dict(state_dict, strict=False)
        net.to(device=device)
        net.eval()
        val_score = evaluate(test_loader, net, num_classes)

        logging.info('test_score: %f', val_score)

    return val_score

def remove_projection_head_weights(state_dict):
    keys_to_remove = [k for k in state_dict.keys() if 'projection' in k]
    for key in keys_to_remove:
        del state_dict[key]
    return state_dict

if __name__ == '__main__':
    print("Current working directory:", os.getcwd())
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    test(args)
