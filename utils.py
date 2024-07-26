import os

import cv2
import librosa
import numpy as np
from facexlib.utils.face_restoration_helper import FaceRestoreHelper

osp = os.path
import torch
from gfpgan import GFPGANer
from hparams import *


class BATCH_GFP(GFPGANer):
    parse_batch_size = 8
    enhance_batch_size = 8

    def __init__(self, device='cpu',arch='clean',model_path='gfpgan/weights/GFPGANv1.3.pth'):
        super().__init__(model_path, 2, arch, 2, None, device)
        self.face_helper = FaceRestoreHelper(
            upscale_factor=2,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_mobile0.25',
            save_ext='png',
            use_parse=True,
            device=self.device,
            model_rootpath='gfpgan/weights')
        # self.gfpgan=torch.nn.parallel.DataParallel(self.gfpgan, device_ids=[0, 1], output_device=1)
        # self.face_helper.face_parse = torch.nn.parallel.DataParallel(self.face_helper.face_parse, device_ids=[0, 1], output_device=1)

    def __parse(self, face_input: torch.Tensor):
        torch.cuda.synchronize()
        with torch.no_grad():
            res = self.face_helper.face_parse(face_input)[0]
        torch.cuda.synchronize()
        return res

    def __enhance(self, tensor: torch.Tensor, weight):
        torch.cuda.synchronize()
        with torch.no_grad():
            res = self.gfpgan(tensor, return_rgb=False, weight=weight)[0]
        torch.cuda.synchronize()
        return res

    def parse_heads(self, u8bgr_heads: np.ndarray):
        tensor = torch.from_numpy(u8bgr_heads[..., (2, 1, 0)].transpose((0, 3, 1, 2))).type(torch.float32).to(
            self.device)
        face_input = tensor / 127.5 - 1
        # f16rgb=u8bgr_heads[...,::-1].transpose((0, 3, 1, 2)).astype(np.float32)/127.5-1
        # face_input = torch.from_numpy(f16rgb).to(self.device)
        while True:
            predictions = []
            try:
                for i in range(0, u8bgr_heads.shape[0], self.parse_batch_size):
                    predictions.append(self.__parse(face_input[i:i + self.parse_batch_size]))
            except RuntimeError as e:
                if self.parse_batch_size == 1:
                    raise RuntimeError('Image too big to run face detection on GPU.')
                self.parse_batch_size //= 2
                print('Recovering from OOM error; New batch size: {}'.format(self.parse_batch_size))
                continue
            return torch.concatenate(predictions, dim=0)

    def enhance_heads(self, u8bgr_heads: np.ndarray, weight=0.5):
        tensor = torch.from_numpy(u8bgr_heads[..., (2, 1, 0)].transpose((0, 3, 1, 2))).type(torch.float32).to(
            self.device)
        tensor = tensor / 127.5 - 1
        # f16rgb=u8bgr_heads[...,::-1].transpose((0, 3, 1, 2)).astype(np.float32)/127.5-1
        # tensor = torch.from_numpy(f16rgb).to(self.device)
        while True:
            predictions = []
            try:
                for i in range(0, u8bgr_heads.shape[0], self.enhance_batch_size):
                    predictions.append(self.__enhance(tensor[i:i + self.enhance_batch_size], weight))
            except RuntimeError as e:
                if self.enhance_batch_size == 1:
                    raise RuntimeError('Image too big to run face detection on GPU.')
                self.enhance_batch_size //= 2
                print('Recovering from OOM error; New batch size: {}'.format(self.enhance_batch_size))
                continue
            outputs = torch.concatenate(predictions, dim=0)
            outputs[outputs > 1] = 1
            outputs[outputs < -1] = -1
            outputs = 127.5 * outputs[:, (2, 1, 0)] + 127.5
            return np.transpose(outputs.cpu().numpy().astype(np.uint8), (0, 2, 3, 1))


import face_alignment


class FaceAlignment(face_alignment.api.FaceAlignment):
    """
    sfd : https://zhuanlan.zhihu.com/p/64859156
    dlib: https://blog.csdn.net/qq_44431690/article/details/106589485
    blazeface https://zhuanlan.zhihu.com/p/73741766
    """

    def get_detections_for_batch(self, u8bgr_images):
        images = u8bgr_images[..., ::-1].transpose((0, 3, 1, 2)).copy()
        if not isinstance(images, torch.Tensor):
            images = torch.as_tensor(images)
        detected_faces = self.face_detector.detect_from_batch(images)
        results = []

        for i, d in enumerate(detected_faces):
            if len(d) == 0:
                results.append(None)
                continue
            d = d[0]
            d = np.clip(d, 0, None)

            x1, y1, x2, y2 = map(int, d[:-1])
            results.append((x1, y1, x2, y2))
        return results


def time_print(func):
    import functools, time
    @functools.wraps(func)
    def wrapper(*args, **kargs):
        p = time.time()
        f = func(*args, **kargs)
        print(f"{func.__name__}:{time.time() - p}")
        return f

    return wrapper


def load_wav(path, sr):
    return librosa.core.load(path, sr=sr)[0]


_mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)


def melspectrogram(wav):
    def _amp_to_db(x):
        min_level = np.exp(min_level_db / 20 * np.log(10))
        return 20 * np.log10(np.maximum(min_level, x))

    def _preemphasis(wav, k, preemphasize=True):
        if preemphasize:
            from scipy import signal
            return signal.lfilter([1, -k], [1], wav)
        return wav

    def _stft(y):
        return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_size, win_length=win_size)

    def _normalize(S):
        return np.clip((2 * max_abs_value) * ((S - min_level_db) / (-min_level_db)) - max_abs_value, -max_abs_value,
                       max_abs_value)

    D = _stft(_preemphasis(wav, preemphasis))
    S = _amp_to_db(np.dot(_mel_basis, np.abs(D))) - ref_level_db
    return _normalize(S)


import torch


def _load(checkpoint_path, device):
    if device == "cuda":
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    return checkpoint


def save_sample_images(x, g, gt, global_step, checkpoint_dir):
    x = (x.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    g = (g.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    gt = (gt.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    # refs=整脸B脸   inps=上半A脸  gt=整张A脸  g=pred的A话整脸
    refs, inps = x[..., 3:], x[..., :3]
    folder = osp.join(checkpoint_dir, "samples_step{:09d}".format(global_step))
    if not os.path.exists(folder): os.mkdir(folder)
    # 原始整脸+下半掩码+
    collage = np.concatenate((refs, inps, g, gt), axis=-2)
    for batch_idx, c in enumerate(collage):
        for t in range(len(c)):
            cv2.imwrite('{}/{}_{}.jpg'.format(folder, batch_idx, t), c[t])


def batch_cv2resize(frames:np.ndarray,height:int,width:int,interpolation=cv2.INTER_LINEAR):
    # frames (batch,h,w,c)
    if frames.ndim==3:
        frames=frames[...,None]
    batch, bgf_h, bgf_w,c = frames.shape
    mid = np.transpose(frames, axes=(1, 2, 0, 3)).reshape((bgf_h, bgf_w, -1))
    mid = cv2.resize(mid, (width, height),interpolation=interpolation)
    res = np.transpose(mid.reshape((height, width, -1, c)), axes=(2, 0, 1, 3))
    if res.shape[3]==1:
        return res[...,0]
    return res

def cv2imread(filename):
    return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), -1)


if __name__=="__main__":
    from PIL import Image
    f=Image.open(r"E:\learn\stream-wav2lip\video_data\out_dir\head_faces\00000.jpg")
    f=np.uint8(f)[None,:,:,:]
    res=batch_cv2resize(f[(0,0,0),:,:,:],height=500,width=300)
    Image.fromarray(res[0],"RGB").show()