import argparse
import torch
import os
from trainers import sup_train, unsup_train
from utils.seg_loss import DiceFocalLoss

parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
parser.add_argument('--data_path', type=str, default='../data2D', help='Name of Experiment')
parser.add_argument('--checkpoint_path', type=str, default='checkpoints', help='Name of Experiment')
parser.add_argument('--epochs', '-e', metavar='E', type=int, default=300, help='Number of epochs')
parser.add_argument('--sup_epochs', '-se', metavar='E', type=int, default=500, help='Number of epochs')
parser.add_argument('--unsup_epochs', '-ue', metavar='E', type=int, default=500, help='Number of epochs')
parser.add_argument('--edge_epochs', metavar='E', type=int, default=200, help='Number of epochs')
parser.add_argument('--batch_size', '-b', dest='batch_size', metavar='B', type=int, default=32, help='Batch size')
parser.add_argument('--lr', '-l', metavar='LR', type=float, default=1e-4,
                    help='Main training learning rate', dest='lr')
parser.add_argument('--load', '-f', type=int, default=0, help='Load model from a .pth file')
parser.add_argument('--save_checkpoint', type=bool, default=True, help='save_checkpoint')

parser.add_argument('--classes', '-c', type=int, default=5, help='Number of classes')
parser.add_argument('--seed', type=int, default=8888, help='random seed')
parser.add_argument('--patience', type=int, default=30, help='patience for student model')
parser.add_argument('--model_type', type=str, default='unet2D', choices=['unet', 'unet2D'],
                    help='Type of the model to use')
parser.add_argument('--mode', type=str, default="CT", help='mode to run')
parser.add_argument('--gpu', type=int, default=0, help='gpu to run')
parser.add_argument('--warmup_epochs', type=int, default=5, help='warmup_epochs')
parser.add_argument('--decay_rate', type=float, default=0.95, help='decay_rate')
parser.add_argument('--remind', type=str, default='pre', help='remind me')
parser.add_argument('--threshold', type=float, default='0.5', help='entropy_threshold')
parser.add_argument('--stage', type=str, default='unsup', choices=['sup', 'unsup', 'img', 'tsne'], help='stage')
parser.add_argument('--checkpoint_name', type=str, default='checkpoint_name', help='checkpoint_name')

parser.add_argument('--contra_weight', type=float, default=5, help='')
parser.add_argument('--N', type=int, default=5, help='centroid number')
parser.add_argument('--k', type=float, default=0.2, help='Top K begin')
parser.add_argument('--start_epoch', type=int, default=50, help='alpha')

parser.add_argument('--ablation', type=str, default='all',
                    choices=['ori', 'idg', 'con', 'sel', 'idgcon', 'idgsel', 'consel', 'all'], help='alpha')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
torch.cuda.set_device(0)

import ast
import configparser
import json
import logging
import random
import time
from datetime import timedelta

import monai
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from model.unet import UNet

from utils.dataloader import Getfile, MixedDataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import os

from utils.nt_xent_loss import NTXentLoss
from utils.utils import reverse_mode


def main(args):
    total_start_time = time.time()
    learning_rate = args.lr
    data_path = args.data_path
    batch_size = args.batch_size
    num_classes = args.classes
    save_checkpoint = args.save_checkpoint
    dir_checkpoint = os.path.join(args.checkpoint_path, args.mode)

    config = configparser.ConfigParser()
    print("Loading config.ini")
    config.read('config.ini')
    train_dirs_str = config.get(args.mode, 'train_dirs')
    train_dirs = ast.literal_eval(train_dirs_str)
    val_dirs = config.get(args.mode, 'val_dir')
    val_dir = val_dirs.split(", ")
    label_intensities_str = config.get(args.mode, 'label_intensities')
    label_intensities = tuple(map(float, label_intensities_str.split(', ')))
    class_to_pixel_str = config.get(args.mode, 'class_to_pixel')
    class_to_pixel = json.loads(class_to_pixel_str)

    try:
        if_aug = False
        if_vision = False
        sup_num_data = 8000
        unsup_num_data = 8000
        if_shuffle = True
        supervised_ratio = 1
        sup_batch_size = batch_size
        unsup_batch_size = batch_size
        if args.stage == 'unsup':
            supervised_ratio = 0.25
            sup_batch_size = int(batch_size * supervised_ratio) 
            unsup_batch_size = batch_size - sup_batch_size


        start_time = time.time()
        supervised_data = Getfile(base_dir=data_path, image_dirs=train_dirs, domain=0, num_classes=num_classes,
                                  label_intensities=label_intensities, mode=args.mode, onehot=True,
                                  num_data=sup_num_data,
                                  aug=if_aug, vision=if_vision)
        supervised_dataloader = DataLoader(supervised_data, batch_size=sup_batch_size, shuffle=if_shuffle,
                                           num_workers=8,
                                           pin_memory=True, persistent_workers=True, drop_last=True)

        unsupervised_data = Getfile(base_dir=data_path, image_dirs=train_dirs, domain=1, num_classes=num_classes,
                                    label_intensities=label_intensities, mode=reverse_mode(args.mode), onehot=True,
                                    num_data=unsup_num_data, aug=if_aug, vision=if_vision)
        unsupervised_dataloader = DataLoader(unsupervised_data, batch_size=unsup_batch_size, shuffle=if_shuffle,
                                             num_workers=8, pin_memory=True, persistent_workers=True, drop_last=True)
        mixed_dataloader = MixedDataLoader(supervised_dataloader, unsupervised_dataloader,
                                           supervised_ratio=supervised_ratio)

        source_val_dataloader = DataLoader(
            Getfile(base_dir=data_path, val_dir=val_dir[0], domain=1, num_classes=num_classes,
                    label_intensities=label_intensities, mode=args.mode, onehot=False, num_data=0, aug=False),
            batch_size=64, shuffle=False, num_workers=8)
        target_val_dataloader = DataLoader(
            Getfile(base_dir=data_path, val_dir=val_dir[1], domain=1, num_classes=num_classes,
                    label_intensities=label_intensities, mode=args.mode, onehot=False, num_data=0, aug=False),
            batch_size=64, shuffle=False, num_workers=8)

        end_time = time.time()
        print(f"Data loaded in {str(timedelta(seconds=end_time - start_time))}")

        contrastive_loss = NTXentLoss(device=device, temperature=0.1, use_cosine_similarity=True)

        sup_segloss = monai.losses.DiceFocalLoss()
        seg_loss = DiceFocalLoss()
     
        if args.stage == 'sup':
            print('start supervised training')
            encoder_dropout_rate = [0, 0, 0, 0, 0.5]
            decoder_dropout_rate = [0.5, 0, 0, 0]
            sup_model = UNet(in_channels=1, out_channels=args.classes, encoder_dropout_rate=encoder_dropout_rate,
                             decoder_dropout_rate=decoder_dropout_rate,
                             use_projection=True).to(device)
            if args.load != 0:
                checkpoint_path = os.path.join(args.checkpoint_path, 'checkpoint_epoch{}.pth'.format(args.load))
                state_dict = torch.load(checkpoint_path, map_location=device)
                sup_model.load_state_dict(state_dict)
                logger.info(f'Model loaded from epoch {args.load}')
            sup_model.train()
            if args.checkpoint_name == 'ori':
                learning_rate = 1e-5
            sup_opt = optim.Adam(sup_model.parameters(), lr=learning_rate, weight_decay=0.0001)
            sup_scheduler = CosineAnnealingLR(sup_opt, T_max=args.sup_epochs, eta_min=1e-6)
            num_batch_per_epoch = len(supervised_dataloader)
            sup_train(sup_model, supervised_dataloader, source_val_dataloader, target_val_dataloader,
                      sup_opt, contrastive_loss, sup_segloss, num_batch_per_epoch, dir_checkpoint,
                      sup_scheduler, args, device)
        elif args.stage == 'unsup':
            print('start unsupervised training')
            encoder_dropout_rate = [0, 0, 0, 0, 0.5]
            decoder_dropout_rate = [0.5, 0, 0, 0]
            unsup_model = UNet(in_channels=1, out_channels=args.classes, encoder_dropout_rate=encoder_dropout_rate,
                               decoder_dropout_rate=decoder_dropout_rate,
                               use_projection=True).to(device)
            sup_model_weight = None

            if args.checkpoint_name.endswith('unsup_ori'):
                sup_model_weight = 'sup/study_sup_ori/unet2D_best_model.pth'
            elif args.checkpoint_name.endswith('unsup_all'):
                sup_model_weight = 'sup/study_sup_idgcon/unet2D_best_model.pth'
            print('Loading:', sup_model_weight)
            sup_weight_path = os.path.join(dir_checkpoint, sup_model_weight)
            state_dict = torch.load(sup_weight_path)
            unsup_model.load_state_dict(state_dict)
            unsup_opt = optim.Adam(unsup_model.parameters(), lr=learning_rate, weight_decay=0.0001)
            unsup_scheduler = CosineAnnealingLR(unsup_opt, T_max=args.unsup_epochs, eta_min=1e-6)

            unsup_train(unsup_model, mixed_dataloader, supervised_dataloader, source_val_dataloader,
                        target_val_dataloader, unsup_opt, contrastive_loss, seg_loss,
                        dir_checkpoint, unsup_scheduler, args, device)

        total_end_time = time.time()
        print(f"Total training time: {str(timedelta(seconds=total_end_time - total_start_time))} seconds")

        print(args.remind)

    except KeyboardInterrupt:
        print("Training interrupted. Saving current model state.")
        print(args.remind)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    log_file = 'training_log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    logger = logging.getLogger('main_logger')
    logger.addHandler(file_handler)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_device = torch.cuda.current_device() if torch.cuda.is_available() else None
    device_str = f'cuda:{gpu_device}' if gpu_device is not None else 'cpu'
    logger.info(f'Using device {device_str}')

    torch.backends.cudnn.enable = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    main(args)
