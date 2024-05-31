import os
import subprocess
from typing import Tuple

import cv2
import numpy as np
import torch
from torch.nn import Module

import utils
from utils import BATCH_GFP

torch.set_autocast_enabled(True)
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_gfp = BATCH_GFP(device)

MASK_COLORMAP = [0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 0, 0, 0]
MASK_COLORMAP2 = (0, 14, 16, 17, 18)
norm_arg = [0.5, 0.5, 0.5]
model_imgsize=96

def checkParamNum2Model(stats)->Tuple[Module,int]:
    from primepake_wav2lip.models.wav2lip import Wav2Lip_96,Wav2Lip_288
    from primepake_wav2lip.models.sam import Wav2Lip_192SAM,Wav2Lip_384SAM
    l=len(stats)
    if l==352:
        return Wav2Lip_96,96
    elif l==422:
        return Wav2Lip_288,288
    elif l==507:
        return Wav2Lip_192SAM,192
    elif l==479:
        return Wav2Lip_384SAM,384
    else:
        raise Exception("其他模型可以修改此处参数数量，引入自己的模型")
def load_model(path):
    checkpoint = torch.load(path, map_location=device)
    new_s = {}
    for k, v in checkpoint["state_dict"].items():
        new_s[k.replace('module.', '')] = v
    global model_imgsize
    Model_net,model_imgsize=checkParamNum2Model(new_s)
    model = Model_net()
    model.load_state_dict(new_s)
    return model.to(device).eval()
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[-1],output_device=-1)
    # return model.eval()


def face2background(u8bgr_bg: np.ndarray, u8bgr_face: np.ndarray, coords: np.ndarray):
    if u8bgr_bg.shape[0] != u8bgr_face.shape[0] or coords.shape[0] != u8bgr_face.shape[0]:
        raise Exception("长度不等")
    # todo 优化：不用for 使用numpy的where
    for face, bg, coord in zip(u8bgr_face, u8bgr_bg, coords):
        y1, y2, x1, x2 = coord
        bg[y1:y2, x1:x2] = cv2.resize(face, (x2 - x1, y2 - y1))
    return u8bgr_bg


def head2background(frames, u8bgr_head, invAffMats):
    def copy_head_with_mask(inv_soft_masks: np.ndarray,
                            pasted_faces: np.ndarray,
                            frames: np.ndarray):
        res = torch.from_numpy(inv_soft_masks).to(device) * torch.from_numpy(pasted_faces).to(device) \
              + torch.from_numpy(1 - inv_soft_masks).to(device) * torch.from_numpy(frames).to(device)
        return res.type(torch.uint8).cpu().numpy()

    frame_num = len(frames)
    thres = 10
    expect = [i for i in range(thres)] + [u8bgr_head.shape[1] - i - 1 for i in range(thres)]
    h_up, w_up = (frames.shape[1], frames.shape[2])
    pasted_faces = np.asarray([cv2.warpAffine(u8bgr_head[i], invAffMats[i], (w_up, h_up)) for i in range(frame_num)],
                              dtype=np.uint8)
    inv_soft_mask_list = []
    outs = batch_gfp.parse_heads(u8bgr_head)
    outs = outs.argmax(dim=1).cpu().numpy()
    mask_parsing = np.full(outs.shape, fill_value=255., dtype=np.float32)
    for idx in MASK_COLORMAP2:
        mask_parsing[outs == idx] = 0
    # for i in range(frame_num):
    #     mask_parsing[i] = cv2.GaussianBlur(mask_parsing[i], (101, 101), 11)
    #     mask_parsing[i] = cv2.GaussianBlur(mask_parsing[i], (101, 101), 11)
    mask_parsing[:, expect, :] = 0
    mask_parsing[..., expect] = 0
    for i in range(frame_num):
        mask = cv2.resize(mask_parsing[i] / 255., u8bgr_head[i].shape[:2])
        mask = cv2.warpAffine(mask, invAffMats[i], (w_up, h_up), flags=3)
        inv_soft_mask_list.append(np.expand_dims(mask, axis=-1))
    inv_soft_masks = np.asarray(inv_soft_mask_list, dtype=np.float16)
    return copy_head_with_mask(inv_soft_masks, pasted_faces, frames)


def face4mel(
        model,
        u8bgr_face: np.ndarray,
        mel_chunks: np.ndarray
):
    if mel_chunks.shape[0] != u8bgr_face.shape[0]:
        raise Exception("音频和帧数不等")
    if len(mel_chunks.shape) == 3:
        mel_chunks = np.expand_dims(mel_chunks, axis=-1)
    batch,bgf_h,bgf_w=u8bgr_face.shape[:3]
    if bgf_h!=model_imgsize or bgf_w!=model_imgsize:
        mid=np.transpose(u8bgr_face,axes=(1,2,0,3)).reshape((bgf_h,bgf_w,-1))
        mid=cv2.resize(mid,(model_imgsize,model_imgsize))
        u8bgr_face=np.transpose(mid.reshape((model_imgsize,model_imgsize,-1,3)),axes=(2,0,1,3))
    mask_faces = u8bgr_face.copy()
    mask_faces[:, u8bgr_face.shape[1] // 2:] = 0
    concat_faces = np.concatenate((mask_faces, u8bgr_face), axis=3, dtype=np.float16) / 255.
    img_batch = torch.FloatTensor(np.transpose(concat_faces, (0, 3, 1, 2))).to(device)
    mel_batch = torch.FloatTensor(np.transpose(mel_chunks, (0, 3, 1, 2))).to(device)

    with torch.no_grad():
        pred = model(mel_batch, img_batch)
        res=np.uint8(pred.detach().cpu().numpy() * 255)
    if bgf_h!=model_imgsize or bgf_w!=model_imgsize:
        mid=np.transpose(res.reshape((-1,model_imgsize,model_imgsize)),(1,2,0))
        mid = cv2.resize(mid, (bgf_w, bgf_h))
        res=np.transpose(mid,(2,0,1)).reshape(batch,3,bgf_w,bgf_h)

    res=np.transpose(res,(0, 2, 3, 1))
    return res


def enhance_mel2lip(
        model,
        u8bgr_frames: np.ndarray,
        u8bgr_heads: np.ndarray,
        invAffMats: np.ndarray,
        u8bgr_faces: np.ndarray,
        coords: np.ndarray,
        mel_chunks: np.ndarray
):
    def enhance(
            u8bgr_frames: np.ndarray,
            u8bgr_heads: np.ndarray,
            invAffMats: np.ndarray,
            u8bgr_face: np.ndarray,
            coords: np.ndarray
    ):
        """

        Args:
            u8bgr_frames: 完整帧 shape=(num,h,w,3)
            u8bgr_heads: gfp扣的头shape=(num,512,512,3)
            invAffMats: 逆仿射矩阵
            u8bgr_face: 已替嘴型的脸shape=(num,96,96,3)
            coords: 脸相对头的位置

        Returns:

        """
        u8bgr_heads = face2background(u8bgr_heads, u8bgr_face, coords)
        enhance_u8bgr_heads = batch_gfp.enhance_heads(u8bgr_heads)
        return head2background(u8bgr_frames, enhance_u8bgr_heads, invAffMats)

    u8bgr_faces = face4mel(model, u8bgr_faces, mel_chunks)
    return enhance(u8bgr_frames, u8bgr_heads, invAffMats, u8bgr_faces, coords)


def mel2lip(
        model,
        u8bgr_frames: np.ndarray,
        u8bgr_faces: np.ndarray,
        body_coords: np.ndarray,
        mel_chunks: np.ndarray
):
    u8bgr_faces = face4mel(model, u8bgr_faces, mel_chunks)
    return face2background(u8bgr_frames, u8bgr_faces, body_coords)


def audio2mel(
        fps,
        audio_file: str
):
    from utils import osp
    if not osp.exists("temp"):
        os.mkdir("temp")
    command = 'ffmpeg -y -i {} -acodec pcm_s16le -ar 16000 -ac 1  {} -loglevel error'.format(audio_file,'temp/temp2.wav')
    subprocess.call(command, shell=True)
    audio_file = 'temp/temp2.wav'
    wav = utils.load_wav(audio_file, 16000)
    wav_int16 = np.int16(wav * 32767)

    mel = utils.melspectrogram(wav)
    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

    mel_chunks = []
    mel_idx_multiplier = 80. / fps
    i = 0
    mel_step_size = 16
    while True:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])
        i += 1
    return np.asarray(mel_chunks, dtype=np.float16), wav_int16
