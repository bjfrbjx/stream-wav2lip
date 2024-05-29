import sys,os
sys.path.extend([os.getcwd(), os.path.join(os.getcwd(), "..","..")])
import argparse
import os

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from primepake_wav2lip.models.syncnet import SyncNet_color_384 as SyncNet
from trains import Sync_Dataset, load_checkpoint, syncnet_train

parser = argparse.ArgumentParser(description='Code to train the expert lip-sync discriminator')

parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", required=True)
parser.add_argument('--device', default="cpu")
parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', required=True, type=str)
parser.add_argument('--pre_path', help='Resumed from this checkpoint', default=None, type=str)
args = parser.parse_args()
from hparams import *
img_size = 384


if not torch.cuda.is_available():
    args.device = "cpu"

if __name__ == "__main__":
    # !!! 因为BatchNormal的存在，model.train()下触发_verify_batch_size，所以需要准备两个以上的样例
    checkpoint_dir = args.checkpoint_dir
    checkpoint_path = args.pre_path

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    # Dataset and Dataloader setup
    train_dataset = Sync_Dataset(f"{args.data_root}/train.txt",(img_size,img_size))
    test_dataset = Sync_Dataset(f"{args.data_root}/val.txt",(img_size,img_size))

    train_data_loader = DataLoader(train_dataset, batch_size=syncnet_batch_size, shuffle=True,num_workers=1)

    test_data_loader = DataLoader(test_dataset, batch_size=syncnet_batch_size,num_workers=1)

    # Model
    model = SyncNet().to(args.device)
    global_epoch, global_step = 0, 0
    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = Adam([p for p in model.parameters() if p.requires_grad],lr=syncnet_lr)

    if checkpoint_path is not None:
        model, global_epoch, global_step = load_checkpoint(checkpoint_path, model,args.device, optimizer)

    syncnet_train(img_size,args.device, model, train_data_loader, test_data_loader, optimizer, checkpoint_dir,
                  step_interval=1000, epochs=5000, start_step=global_step, start_epoch=global_epoch)
