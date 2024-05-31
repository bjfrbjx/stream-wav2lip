import argparse
import os
from os.path import join

import torch
from lpips import LPIPS
from torch import nn, cosine_similarity
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from primepake_wav2lip.models import SyncNet_color_384 as SyncNet
from primepake_wav2lip.models import Wav2Lip_192SAM as Wav2Lip, NLayerDiscriminator
from trains import Wav2lip_Dataset, load_checkpoint

parser = argparse.ArgumentParser(description='Code to train the Wav2Lip model WITH the visual quality discriminator')

parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', required=False,
                    default="checkpoints/wav/", type=str)
parser.add_argument('--log_dir', help='Write log files to this directory', required=False, default="logs/wav/",
                    type=str)
parser.add_argument('--exp_num', help='ID number of the experiment', required=False, default="sam", type=str)
parser.add_argument('--syncnet_checkpoint_path', help='Load the pre-trained Expert discriminator', default=None,
                    required=False, type=str)
parser.add_argument('--checkpoint_path', help='Resume generator from this checkpoint', default=None, type=str)
parser.add_argument('--disc_checkpoint_path', help='Resume quality disc from this checkpoint', default=None, type=str)
parser.add_argument('--step_interval', help='保存验证的间隔步数', default=1000, type=int)
parser.add_argument('--epoch', help='训练总轮数', default=50, type=int)
args = parser.parse_args()

global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
best_loss = 10000
print('use_cuda: {}'.format(use_cuda))

syncnet_T = 5
syncnet_mel_step_size = 16
from hparams import *
img_size = 384
syncnet_wt = 0.03


syncnet = SyncNet().to(args.device)
syncnet.eval()
for p in syncnet.parameters():
    p.requires_grad = False


def train(device, model, train_data_loader, test_data_loader, optimizer, disc_optimizer, checkpoint_dir, syncnet,
          disc, step_interval=None, epochs=None, start_step=0, start_epoch=0):
    global syncnet_wt, disc_wt
    disc_iter_start = 30000
    sync_iter_start = 250000
    from torch.nn import L1Loss, BCELoss
    recon_loss = L1Loss()
    step = start_step
    epoch = start_epoch
    _logloss = BCELoss()
    sync_loss_fn = lambda a, v, y: _logloss(cosine_similarity(a, v).unsqueeze(1), y)
    loss_fn_vgg = LPIPS(net='vgg').to(device).eval().to(device)

    def hinge_d_loss(logits_real, logits_fake):
        loss_real = torch.mean(F.relu(1. - logits_real))
        loss_fake = torch.mean(F.relu(1. + logits_fake))
        return loss_real, loss_fake

    def get_sync_loss(mel, g):
        g = g[:, :, :, g.size(3) // 2:]
        g = torch.cat([g[:, :, i] for i in range(syncnet_T)], dim=1)
        # B, 3 * T, H//2, W
        a, v = syncnet(mel, g)
        y = torch.ones(g.size(0), 1).float().to(device)
        return sync_loss_fn(a, v, y)

    def save_ckp(step, epoch):
        checkpoint_path = join(checkpoint_dir, "wav2lipSAM_gan_step{:09d}.pth".format(step))
        torch.save({
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "global_step": step,
            "global_epoch": epoch,
        }, checkpoint_path)
        print("Saved checkpoint:", checkpoint_path)
        disc_path = join(checkpoint_dir, "disc_step{:09d}.pth".format(step))
        torch.save({
            "state_dict": disc.state_dict(),
            "optimizer": disc_optimizer.state_dict(),
            "global_step": step,
            "global_epoch": epoch,
        }, disc_path)
        print("Saved checkpoint:", disc_path)

    try:
        for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
            prog_bar = tqdm(train_data_loader)
            running_sync_loss, running_l1_loss, disc_loss, running_perceptual_loss = 0., 0., 0., 0.
            running_disc_real_loss, running_disc_fake_loss, running_vgg_loss = 0., 0., 0.
            for x, indiv_mels, mel, gt in prog_bar:
                step += 1
                disc.train()
                model.train()

                x = x.to(device)
                mel = mel.to(device)
                indiv_mels = indiv_mels.to(device)
                gt = gt.to(device)

                optimizer.zero_grad()
                disc_optimizer.zero_grad()

                # x=[上半A脸+整脸B脸], indiv_mels=A时音段, gt=[整脸A脸]
                # g=【indiv_mels, x】->A话整脸
                g = model(indiv_mels, x)

                if step > disc_iter_start:
                    d_weight = 0.025
                    fake_output = disc(g)
                    perceptual_loss = -torch.mean(fake_output)
                else:
                    d_weight = 0.
                    perceptual_loss = torch.tensor(0.)

                l1loss = recon_loss(g, gt)

                vgg_loss = loss_fn_vgg(torch.cat([g[:, :, i] for i in range(g.size(2))], dim=0) * 2 - 1,
                                       torch.cat([gt[:, :, i] for i in range(gt.size(2))], dim=0) * 2 - 1).mean()

                nll_loss = l1loss + vgg_loss

                if step > sync_iter_start and syncnet_wt > 0.:
                    sync_loss = get_sync_loss(mel, g)
                else:
                    sync_loss = torch.tensor(0.)

                loss = syncnet_wt * sync_loss + d_weight * perceptual_loss + nll_loss
                loss.backward()
                optimizer.step()

                ### Remove all gradients before Training disc
                disc_optimizer.zero_grad()
                if step > disc_iter_start:
                    real_output = disc(gt)
                    fake_output = disc(g.detach())
                    disc_real_loss, disc_fake_loss = hinge_d_loss(real_output, fake_output)
                    d_loss = 0.5 * (disc_fake_loss + disc_real_loss)
                    d_loss.backward()
                    disc_optimizer.step()
                else:
                    disc_real_loss = torch.tensor(0.)
                    disc_fake_loss = torch.tensor(0.)
                running_disc_real_loss += disc_real_loss.item()
                running_disc_fake_loss += disc_fake_loss.item()
                running_l1_loss += l1loss.item()
                running_perceptual_loss += perceptual_loss.item()
                running_vgg_loss += vgg_loss.item()

                if step > sync_iter_start and syncnet_wt > 0.:
                    running_sync_loss += sync_loss.item()

                if step == 1 or step % step_interval == 0:
                    averaged_sync_loss = eval(test_data_loader, device, model, disc, get_sync_loss)
                    utils.save_sample_images(x, g, gt, step, checkpoint_dir)
                    save_ckp(step, epoch)
                    if averaged_sync_loss < 0.3:
                        break
                prog_bar.set_description('L1: {}, Sync: {}, Percep: {} | Fake: {}, Real: {}'
                                         .format(running_l1_loss / (step + 1),
                                                 running_sync_loss / (step + 1),
                                                 running_perceptual_loss / (step + 1),
                                                 running_disc_fake_loss / (step + 1),
                                                 running_disc_real_loss / (step + 1)))
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        save_ckp(step, epoch)


def eval(test_data_loader, device, model, disc, get_sync_loss):
    running_sync_loss = 0.
    step = 0
    model.eval()
    disc.eval()
    with torch.no_grad():
        for x, indiv_mels, mel, gt, vidname in test_data_loader:
            step += 1
            x = x.to(device)
            mel = mel.to(device)
            indiv_mels = indiv_mels.to(device)
            g = model(indiv_mels, x)
            sync_loss = get_sync_loss(mel, g)
            running_sync_loss += sync_loss.item()
    avg_loss = running_sync_loss / step
    print('eval Step {} | Sync: {:.6}'.format(step, avg_loss))
    return avg_loss


if __name__ == "__main__":
    checkpoint_dir = args.checkpoint_dir

    # Dataset and Dataloader setup
    train_dataset = Wav2lip_Dataset(f"{args.data_root}/train.txt", (img_size,img_size))
    test_dataset = Wav2lip_Dataset(f"{args.data_root}/val.txt", (img_size,img_size))

    train_data_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=1)

    test_data_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        num_workers=1)

    # Model
    model = Wav2Lip().to(args.device)
    disc = NLayerDiscriminator().to(args.device)

    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('total DISC trainable params {}'.format(sum(p.numel() for p in disc.parameters() if p.requires_grad)))

    optimizer = Adam([p for p in model.parameters() if p.requires_grad],
                     lr=initial_learning_rate, betas=(0.5, 0.999))
    disc_optimizer = Adam([p for p in disc.parameters() if p.requires_grad],
                          lr=disc_initial_learning_rate, betas=(0.5, 0.999))

    if args.pre_path is not None:
        model, global_epoch, global_step = load_checkpoint(args.pre_path, model, args.device, optimizer)

    if args.disc_checkpoint_path is not None:
        load_checkpoint(args.disc_checkpoint_path, disc, args.device, disc_optimizer)

    load_checkpoint(args.syncnet_path, syncnet, args.device, None)

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    # Train!
    train(args.device, model, train_data_loader, test_data_loader, optimizer, disc_optimizer, checkpoint_dir,
          syncnet, disc,
          step_interval=args.step_interval, epochs=args.epoch, start_step=global_step, start_epoch=global_epoch)
