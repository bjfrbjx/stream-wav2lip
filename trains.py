import math
import os.path
import random
from glob import glob
from os.path import basename, dirname, join, isfile

import cv2
import numpy as np
import torch
from torch import cosine_similarity, autograd
from torch.nn.functional import binary_cross_entropy
from torch.utils.data import Dataset
from tqdm import tqdm

import utils
from hparams import *


# mel augmentation
def mask_mel(crop_mel):
    block_size = 0.1
    time_size = math.ceil(block_size * crop_mel.shape[0])
    freq_size = math.ceil(block_size * crop_mel.shape[1])
    time_lim = crop_mel.shape[0] - time_size
    freq_lim = crop_mel.shape[1] - freq_size

    time_st = random.randint(0, time_lim)
    freq_st = random.randint(0, freq_lim)

    mel = crop_mel.copy()
    mel[time_st:time_st + time_size] = -4.
    mel[:, freq_st:freq_st + freq_size] = -4.
    return mel


class Sync_Dataset(Dataset):

    def id2frameFile(self,frame_id):
        frame1 = join(self.vidname, f'{frame_id:05d}.jpg')
        frame2 = join(self.vidname, f'{frame_id}.jpg')
        return frame1 if os.path.exists(frame1) else frame2

    def get_wrong_window(self, postive_img_name):
        postive_img_id=self.get_frame_id(postive_img_name)
        tl=list(range(len(self.img_names)))
        tl=tl[:postive_img_id]+tl[postive_img_id+syncnet_T:]
        tl=random.choices(tl,k=syncnet_T)
        if random.random() > 0.6:
            tl = [random.choice(tl)]*syncnet_T
        tl=[self.id2frameFile(i) for i in tl]
        return tl

    def __init__(self, work_txt, img_size):
        self.target_imgsize = img_size
        self.work_txt = work_txt
        with open(self.work_txt, "r") as f:
            self.all_videos = [i.strip() for i in f.readlines()]

    def get(self):
        return self.__getitem__(0)

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def read_window(self, window_fnames, random_flip=False):
        window = []
        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                return None
            try:
                img = cv2.resize(img, self.target_imgsize)
                if random_flip:
                    img = cv2.flip(img, 1)
            except Exception as e:
                return None

            window.append(img)

        return window

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            frame=self.id2frameFile(frame_id)
            window_fnames.append(frame)
        return window_fnames

    def crop_audio_window(self, spec, start_frame):
        # num_frames = (T x hop_size * fps) / sample_rate
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(num_mels * (start_frame_num / float(fps)))

        end_idx = start_idx + syncnet_mel_step_size
        return spec[start_idx: end_idx, :]

    def audio2mel(self,vidname):
        try:
            mel_out_path = join(vidname, "mel.npy")
            if isfile(mel_out_path):
                orig_mel = np.load(mel_out_path)
            else:
                wavpath = join(vidname, "audio.wav")
                wav = utils.load_wav(wavpath, sample_rate)
                orig_mel = utils.melspectrogram(wav).T
                np.save(mel_out_path, orig_mel)
            return orig_mel
        except Exception as e:
            return None

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while True:
            idx = random.randint(0, len(self.all_videos) - 1)
            random_flip=random.random()>0.5
            self.vidname = self.all_videos[idx]
            self.img_names = sorted((i.replace("\\", "/") for i in glob(join(self.vidname, '*.jpg'))),key=lambda x:int(basename(x).split(".")[0]))[:-syncnet_T]
            postive_img_name = random.choice(self.img_names)

            if random.choice([True, False]):
                y = torch.ones(1).float()
                window_fnames = self.get_window(postive_img_name)
            else:
                y = torch.zeros(1).float()
                window_fnames=self.get_wrong_window(postive_img_name)

            window = self.read_window(window_fnames, random_flip=random_flip)

            if window is None:
                print("sd")
            if len(window_fnames) != len(window):
                continue

            orig_mel=self.audio2mel(self.vidname)
            mel = self.crop_audio_window(orig_mel.copy(), postive_img_name)

            if (mel.shape[0] != syncnet_mel_step_size):
                continue

            # 声谱噪声增强
            if random.random() < 0.3:
                mel = mask_mel(mel)

            # H x W x 3 * T
            x = np.concatenate(window, axis=2) / 255.
            x = x.transpose(2, 0, 1)
            if self.target_imgsize[0]==self.target_imgsize[1]:
                x = x[:, x.shape[1] // 2:]

            x = torch.FloatTensor(x)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)
            return x, mel, y


class Wav2lip_Dataset(Sync_Dataset):


    def get_segmented_mels(self, spec, start_frame):
        mels = []
        start_frame_num = self.get_frame_id(start_frame)
        if start_frame_num - 1 < 0:
            return None
        for i in range(start_frame_num, start_frame_num + syncnet_T):
            m = self.crop_audio_window(spec, i - 1)
            if m.shape[0] != syncnet_mel_step_size:
                return None
            mels.append(m.T)
        mels = np.asarray(mels)
        return mels

    def prepare_window(self, window):
        # 3 x T x H x W
        x = np.asarray(window) / 255.
        x = np.transpose(x, (3, 0, 1, 2))

        return x

    def __getitem__(self, idx):
        while True:
            idx = random.randint(0, len(self.all_videos) - 1)
            random_flip=random.random()>0.5
            self.vidname = self.all_videos[idx]
            self.img_names = sorted((i.replace("\\", "/") for i in glob(join(self.vidname, '*.jpg'))),key=lambda x:int(basename(x).split(".")[0]))[:-syncnet_T]

            postive_img_name = random.choice(self.img_names)

            window_fnames = self.get_window(postive_img_name)
            wrong_window_fnames = self.get_wrong_window(postive_img_name)
            if window_fnames is None or wrong_window_fnames is None:
                continue

            window = self.read_window(window_fnames,random_flip=random_flip)
            wrong_window = self.read_window(wrong_window_fnames,random_flip=random_flip)

            orig_mel=self.audio2mel(self.vidname)
            mel = self.crop_audio_window(orig_mel.copy(), postive_img_name)

            if (mel.shape[0] != syncnet_mel_step_size):
                continue

            indiv_mels = self.get_segmented_mels(orig_mel.copy(), postive_img_name)
            if indiv_mels is None:
                continue

            window = self.prepare_window(window)
            y = window.copy()
            window[:, :, window.shape[2] // 2:] = 0.

            wrong_window = self.prepare_window(wrong_window)
            x = np.concatenate([window, wrong_window], axis=0)

            x = torch.FloatTensor(x)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)
            indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1)
            y = torch.FloatTensor(y)
            # x 【syncnet_T张半脸，syncnet_T张异音整脸】 |(6,syncnet_T,img_size,img_size)
            # mel 连续num_mel长的朗读音波【分片音波-> 判断嘴型同步】 |  (1,num_mels,syncnet_mel_step_size)
            # indiv_mels 差次的后延音波  【拼凑音波-> 生成嘴型】  | (syncnet_T,1,num_mels,syncnet_mel_step_size)
            # y 期望输出的连续syncnet_T张 同音整脸         | (3,syncnet_T,img_size,img_size)
            return x, indiv_mels, mel, y


def syncnet_eval(test_data_loader, device, model, loss_fn):
    losses = []
    with torch.no_grad():
        model.eval()
        for step, (x, mel, y) in enumerate(test_data_loader):
            x = x.to(device)
            mel = mel.to(device)
            a, v = model(mel, x)
            y = y.to(device)
            loss = loss_fn(a, v, y)
            losses.append(loss.item())
    averaged_loss = sum(losses) / len(losses)
    print(f"eval Loss:{averaged_loss}")


def syncnet_train(device, model, train_data_loader, test_data_loader, optimizer, checkpoint_dir,
                  step_interval=None, epochs=None, start_step=0, start_epoch=0):
    from torch.nn import BCELoss
    _logloss = BCELoss()
    loss_fn = lambda a, v, y: _logloss(cosine_similarity(a, v).unsqueeze(1), y)
    step = start_step
    epoch = start_epoch

    def save_ckp(step, epoch):
        checkpoint_path = join(checkpoint_dir, "syncnet_step{:09d}.pth".format(step))
        torch.save({
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "global_step": step,
            "global_epoch": epoch,
        }, checkpoint_path)
        print("Saved checkpoint:", checkpoint_path)

    try:
        for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
            running_loss = 0.
            prog_bar = tqdm(train_data_loader)
            for x, mel, y in prog_bar:
                step += 1
                model.train()
                optimizer.zero_grad()

                x = x.to(device)
                mel = mel.to(device)
                a, v = model(mel, x)
                y = y.to(device)

                loss = loss_fn(a, v, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                if step == 1 or step % step_interval == 0:
                    syncnet_eval(test_data_loader, device, model, loss_fn)
                    save_ckp(step, epoch)

                prog_bar.set_description('train Loss: {}'.format(running_loss / (step + 1)))
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        save_ckp(step, epoch)


def load_checkpoint(path, model, device, optimizer):
    print("Load checkpoint from: {}".format(path))
    checkpoint = utils._load(path, device)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)
    optimizer_state = checkpoint["optimizer"]
    if optimizer_state is not None and optimizer is not None:
        print("Load optimizer state from {}".format(path))
        optimizer.load_state_dict(checkpoint["optimizer"])
    global_step = int(checkpoint.get("global_step", 0))
    global_epoch = int(checkpoint.get("global_epoch", 0))

    return model, global_epoch, global_step


def wav2lip_eval(test_data_loader, device, model, get_sync_loss, recon_loss):
    sync_losses, recon_losses = [], []
    with torch.no_grad():
        model.eval()
        for step, (x, indiv_mels, mel, gt) in enumerate(test_data_loader):
            x = x.to(device)
            gt = gt.to(device)
            indiv_mels = indiv_mels.to(device)
            mel = mel.to(device)
            g = model(indiv_mels, x)
            sync_loss = get_sync_loss(mel, g)
            l1loss = recon_loss(g, gt)

            sync_losses.append(sync_loss.item())
            recon_losses.append(l1loss.item())

    averaged_sync_loss = sum(sync_losses) / len(sync_losses)
    averaged_recon_loss = sum(recon_losses) / len(recon_losses)

    print('eval 【L1: {}, Sync loss: {}】'.format(averaged_recon_loss, averaged_sync_loss))
    return averaged_sync_loss


def wav2lip_train(device, model, train_data_loader, test_data_loader, optimizer, checkpoint_dir, syncnet,
                  step_interval=None, epochs=None, start_step=0, start_epoch=0):
    global syncnet_wt
    from torch.nn import L1Loss, BCELoss
    recon_loss = L1Loss()
    step = start_step
    epoch = start_epoch
    _logloss = BCELoss()
    sync_loss_fn = lambda a, v, y: _logloss(cosine_similarity(a, v).unsqueeze(1), y)

    def get_sync_loss(mel, g):
        g = g[:, :, :, g.size(3) // 2:]
        g = torch.cat([g[:, :, i] for i in range(syncnet_T)], dim=1)
        # B, 3 * T, H//2, W
        a, v = syncnet(mel, g)
        y = torch.ones(g.size(0), 1).float().to(device)
        return sync_loss_fn(a, v, y)

    def save_ckp(step, epoch):
        checkpoint_path = join(checkpoint_dir, "wav2lip_step{:09d}.pth".format(step))
        torch.save({
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "global_step": step,
            "global_epoch": epoch,
        }, checkpoint_path)
        print("Saved checkpoint:", checkpoint_path)

    try:
        for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
            prog_bar = tqdm(train_data_loader)
            running_sync_loss, running_l1_loss = 0., 0.
            for x, indiv_mels, mel, gt in prog_bar:
                step += 1
                # x 【syncnet_T张半脸，syncnet_T张异音整脸】 |(6,syncnet_T,img_size,img_size)
                # mel 连续num_mel长的朗读音波【分片音波-> 判断嘴型同步】 |  (1,num_mels,syncnet_mel_step_size)
                # indiv_mels 差次的后延音波  【拼凑音波-> 生成嘴型】  | (syncnet_T,1,num_mels,syncnet_mel_step_size)
                # y 期望输出的连续syncnet_T张 同音整脸         | (3,syncnet_T,img_size,img_size)
                model.train()
                optimizer.zero_grad()

                x = x.to(device)
                mel = mel.to(device)
                indiv_mels = indiv_mels.to(device)
                gt = gt.to(device)
                # x=[上半A脸+整脸B脸], indiv_mels=A时音段, gt=[整脸A脸]
                # g=【indiv_mels, x】->A话整脸
                g = model(indiv_mels, x)

                if syncnet_wt > 0.:
                    sync_loss = get_sync_loss(mel, g)
                else:
                    sync_loss = 0.

                l1loss = recon_loss(g, gt)

                loss = syncnet_wt * sync_loss + (1 - syncnet_wt) * l1loss
                loss.backward()
                optimizer.step()
                running_l1_loss += l1loss.item()
                if syncnet_wt > 0.:
                    running_sync_loss += sync_loss.item()
                else:
                    running_sync_loss += 0.
                if step == 1 or step % step_interval == 0:
                    averaged_sync_loss = wav2lip_eval(test_data_loader, device, model, get_sync_loss, recon_loss)
                    utils.save_sample_images(x, g, gt, step, checkpoint_dir)
                    save_ckp(step, epoch)
                    if averaged_sync_loss < .75:
                        syncnet_wt = 0.03

    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        save_ckp(step, epoch)


def gan_eval(test_data_loader, device, model, disc, get_sync_loss, recon_loss):
    from torch.nn.functional import binary_cross_entropy
    running_sync_loss, running_l1_loss, running_disc_real_loss, running_disc_fake_loss, running_perceptual_loss, running_total_loss = [], [], [], [], [], []
    with torch.no_grad():
        model.eval()
        disc.eval()
        for step, (x, indiv_mels, mel, gt) in enumerate(test_data_loader):
            x = x.to(device)
            mel = mel.to(device)
            indiv_mels = indiv_mels.to(device)
            gt = gt.to(device)

            pred = disc(gt)
            disc_real_loss = binary_cross_entropy(pred, torch.ones((len(pred), 1)).to(device))

            g = model(indiv_mels, x)
            pred = disc(g)

            disc_fake_loss = binary_cross_entropy(pred, torch.zeros((len(pred), 1)).to(device))

            running_disc_real_loss.append(disc_real_loss.item())
            running_disc_fake_loss.append(disc_fake_loss.item())

            sync_loss = get_sync_loss(mel, g)

            perceptual_loss = disc.perceptual_forward(g)
            l1loss = recon_loss(g, gt)
            loss = syncnet_wt * sync_loss + disc_wt * perceptual_loss + \
                   (1. - syncnet_wt - disc_wt) * l1loss
            running_total_loss.append(loss.item())

            running_l1_loss.append(l1loss.item())
            running_sync_loss.append(sync_loss.item())
            running_perceptual_loss.append(perceptual_loss.item())

    print('total:{}, L1: {}, Sync: {}, Percep: {} | Fake: {}, Real: {}'.format(
        sum(running_total_loss) / len(running_total_loss),
        sum(running_l1_loss) / len(running_l1_loss),
        sum(running_sync_loss) / len(running_sync_loss),
        sum(running_perceptual_loss) / len(running_perceptual_loss),
        sum(running_disc_fake_loss) / len(running_disc_fake_loss),
        sum(running_disc_real_loss) / len(running_disc_real_loss)))
    return sum(running_sync_loss) / len(running_sync_loss)


def gan_train(device, model, train_data_loader, test_data_loader, optimizer, disc_optimizer, checkpoint_dir, syncnet,
              disc, step_interval=None, epochs=None, start_step=0, start_epoch=0):
    global syncnet_wt, disc_wt
    from torch.nn import L1Loss, BCELoss
    recon_loss = L1Loss()
    step = start_step
    epoch = start_epoch
    _logloss = BCELoss()
    sync_loss_fn = lambda a, v, y: _logloss(cosine_similarity(a, v).unsqueeze(1), y)

    def get_sync_loss(mel, g):
        g = g[:, :, :, g.size(3) // 2:]
        g = torch.cat([g[:, :, i] for i in range(syncnet_T)], dim=1)
        # B, 3 * T, H//2, W
        a, v = syncnet(mel, g)
        y = torch.ones(g.size(0), 1).float().to(device)
        return sync_loss_fn(a, v, y)

    def save_ckp(step, epoch):
        checkpoint_path = join(checkpoint_dir, "wav2lip_gan_step{:09d}.pth".format(step))
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
            running_disc_real_loss, running_disc_fake_loss = 0., 0.
            for x, indiv_mels, mel, gt in prog_bar:
                step+=1
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

                if syncnet_wt > 0.:
                    sync_loss = get_sync_loss(mel, g)
                else:
                    sync_loss = 0.

                perceptual_loss = disc.perceptual_forward(g)

                l1loss = recon_loss(g, gt)

                loss = syncnet_wt * sync_loss + disc_wt * perceptual_loss + (1. - syncnet_wt - disc_wt) * l1loss

                loss.backward()
                optimizer.step()

                ### Remove all gradients before Training disc
                disc_optimizer.zero_grad()

                pred = disc(gt)
                disc_real_loss = binary_cross_entropy(pred, torch.ones((len(pred), 1)).to(device))
                disc_real_loss.backward()

                pred = disc(g.detach())
                disc_fake_loss = binary_cross_entropy(pred, torch.zeros((len(pred), 1)).to(device))
                disc_fake_loss.backward()

                disc_optimizer.step()

                running_disc_real_loss += disc_real_loss.item()
                running_disc_fake_loss += disc_fake_loss.item()
                running_l1_loss += l1loss.item()
                running_perceptual_loss += perceptual_loss.item()

                if step == 1 or step % step_interval == 0:
                    averaged_sync_loss = gan_eval(test_data_loader, device, model, disc, get_sync_loss, recon_loss)
                    utils.save_sample_images(x, g, gt, step, checkpoint_dir)
                    save_ckp(step, epoch)
                    if averaged_sync_loss < .75:
                        syncnet_wt = 0.03
                prog_bar.set_description('L1: {}, Sync: {}, Percep: {} | Fake: {}, Real: {}'
                                         .format(running_l1_loss / (step + 1),
                                                 running_sync_loss / (step + 1),
                                                 running_perceptual_loss / (step + 1),
                                                 running_disc_fake_loss / (step + 1),
                                                 running_disc_real_loss / (step + 1)))
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        save_ckp(step, epoch)


def wganGP_train(device, model, train_data_loader, test_data_loader, optimizer, disc_optimizer, checkpoint_dir, syncnet,
                 disc, step_interval=None, epochs=None, start_step=0, start_epoch=0):
    global syncnet_wt, disc_wt
    from torch.nn import L1Loss, BCELoss
    recon_loss = L1Loss()
    LAMBDA = 10
    step = start_step
    epoch = start_epoch
    _logloss = BCELoss()
    sync_loss_fn = lambda a, v, y: _logloss(cosine_similarity(a, v).unsqueeze(1), y)

    def get_sync_loss(mel, g):
        g = g[:, :, :, g.size(3) // 2:]
        g = torch.cat([g[:, :, i] for i in range(syncnet_T)], dim=1)
        # B, 3 * T, H//2, W
        a, v = syncnet(mel, g)
        y = torch.ones(g.size(0), 1).float().to(device)
        return sync_loss_fn(a, v, y)

    def save_ckp(step, epoch):
        checkpoint_path = join(checkpoint_dir, "wav2lip_gan_step{:09d}.pth".format(step))
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
    def disc_grad_loss(pred):
        return -pred.mean()

    try:
        for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
            prog_bar = tqdm(train_data_loader)
            running_sync_loss, running_l1_loss, disc_loss, running_perceptual_loss = 0., 0., 0., 0.
            running_disc_real_loss, running_disc_fake_loss = 0., 0.
            for x, indiv_mels, mel, gt in prog_bar:
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

                if syncnet_wt > 0.:
                    sync_loss = get_sync_loss(mel, g)
                else:
                    sync_loss = 0.

                perceptual_loss = disc_grad_loss(disc(g))
                l1loss = recon_loss(g, gt)
                loss = syncnet_wt * sync_loss + disc_wt * perceptual_loss + (1. - syncnet_wt - disc_wt) * l1loss

                loss.backward()
                optimizer.step()

                ### 训练判别器 【wgan-gp】
                disc_optimizer.zero_grad()
                fake_img = g.detach()
                real_img = gt.detach()
                batch_num = gt.size(0)
                disc_real_loss = disc_grad_loss(disc(real_img))
                disc_fake_loss = -disc_grad_loss(disc(fake_img))
                # gradient penalty【梯度裁剪】
                alpha = torch.rand(1) * torch.ones(batch_num, 1)
                alpha = alpha.expand(batch_num, int(gt.nelement() / batch_num)).contiguous().view(batch_num, 3, syncnet_T,img_size, img_size).to(device)
                interpolates = alpha * gt + ((1 - alpha) * fake_img)
                interpolates = autograd.Variable(interpolates.to(device), requires_grad=True)

                disc_interpolates = disc(interpolates)
                gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                          grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                          create_graph=True, retain_graph=True)[0]
                gradients = gradients.view(gradients.size(0), -1)

                gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
                gradient_penalty.backward()

                disc_optimizer.step()

                running_disc_real_loss += disc_real_loss.item()
                running_disc_fake_loss += disc_fake_loss.item()
                running_l1_loss += l1loss.item()
                running_perceptual_loss += perceptual_loss.item()

                if step == 1 or step % step_interval == 0:
                    averaged_sync_loss = gan_eval(test_data_loader, device, model, disc, get_sync_loss, recon_loss)
                    utils.save_sample_images(x, g, gt, step, checkpoint_dir)
                    save_ckp(step, epoch)
                    if averaged_sync_loss < .75:
                        syncnet_wt = 0.03
                prog_bar.set_description('L1: {}, Sync: {}, Percep: {} | Fake: {}, Real: {}'
                                         .format(running_l1_loss / (step + 1),
                                                 running_sync_loss / (step + 1),
                                                 running_perceptual_loss / (step + 1),
                                                 running_disc_fake_loss / (step + 1),
                                                 running_disc_real_loss / (step + 1)))
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        save_ckp(step, epoch)
