import argparse
import itertools
import math
import queue
import random
import time
import wave
from multiprocessing import Process, Queue
from threading import Thread
from typing import List

import cv2
import numpy as np
import pyaudio
import pyvirtualcam
import torch

from faster_enhance_wav2lip import audio2mel, mel2lip, enhance_mel2lip, load_model

parser = argparse.ArgumentParser(description='流式生成视频帧，并将数据推送到obs和话筒')

parser.add_argument('--checkpoint_path', type=str, help='wav2lip的权重文件', required=True)
parser.add_argument('--data_dir', type=str, help='预处理过的数据文件夹', required=True)
parser.add_argument('--enhance', default=False, action='store_true', help='是否启用gfpgan增强面部')
parser.add_argument('--audio_files', nargs='+', type=str, required=True, help='音频文件')
parser.add_argument('--height', type=int, default=1920, help='导出帧高度')
parser.add_argument('--width', type=int, default=1080, help='导出帧宽度')
parser.add_argument('--batch_size', type=int, default=25, help='使用模型时的批量数，会把batch_size个帧打包丢进数据池')
parser.add_argument('--data_pool_size', type=int, default=100,
                    help='数据池容量，data_pool_size*batch_size=内存缓存的帧数')

parser.add_argument('--outfile', type=str, default='obs',
                    help='输出文件，如果是obs就是流式摄像头，其他则是mp4文件.')

args = parser.parse_args()


class DataPool:
    def __init__(self, data_dir, audio_files: List, batch_size=25, batch_pool_size=100):
        self.__queue = Queue(maxsize=batch_pool_size)
        self.data_dir = data_dir
        data = np.load(f"{data_dir}/face_det.npz", mmap_mode="r")
        self.fps = data["fps"]
        self.head_coords = data["head_coords"]
        self.invAffMats = data["invAffMats"]
        self.body_coords = data["body_coords"]
        self.batch_size = batch_size
        v = [i for i in range(data["num"])]
        self.cycle = itertools.cycle(v + v[::-1])
        for i in range(random.randint(0, len(v))):
            next(self.cycle)
        self.audio_files = audio_files
        self.working_audio = None

    def put(self, data):
        self.__queue.put(data)

    def qsize(self):
        return self.__queue.qsize()

    def get(self, block=True, timeout=None):
        return self.__queue.get(block=block, timeout=timeout)

    def next_data(self, enhance):
        idx = next(self.cycle)
        bg = cv2.imread(f"{self.data_dir}/bg/{idx:05d}.jpg")
        if not enhance:
            body_face = cv2.imread(f"{self.data_dir}/body_faces/{idx:05d}.jpg")
            data = {"bg": bg,
                    "body_face": body_face,
                    "body_coord": self.body_coords[idx]
                    }
        else:
            head = cv2.imread(f"{self.data_dir}/heads/{idx:05d}.jpg")
            head_face = cv2.imread(f"{self.data_dir}/head_faces/{idx:05d}.jpg")
            data = {"bg": bg,
                    "head": head,
                    "head_face": head_face,
                    "head_coord": self.head_coords[idx],
                    "invAffMats": self.invAffMats[idx]}
        return data

    def run(self, enhance=False):
        while len(self.audio_files) > 0:
            self.working_audio = self.audio_files.pop()
            mel_chunks, wav_int16 = audio2mel(self.fps, self.working_audio)
            wav_step = math.ceil(len(wav_int16) / len(mel_chunks))
            # 每一个视频帧对应的音频段
            wav_frames = [wav_int16[i:i + wav_step] for i in range(0, len(wav_int16), wav_step)]
            for i in range(0, len(mel_chunks), self.batch_size):
                # batch_size个视频帧和音频段批量处理
                mels = mel_chunks[i:i + self.batch_size]
                wav_batch_frames = np.concatenate(wav_frames[i:i + self.batch_size], axis=0)
                size = len(mels)
                working_data = []
                for j in range(size):
                    working_data.append(self.next_data(enhance=enhance))
                if working_data[0].get("head") is not None:
                    batch_data = BatchEnhance(num=size,
                                              u8bgr_frames=[m["bg"] for m in working_data],
                                              head_coords=[m["head_coord"] for m in working_data],
                                              head_faces=[m["head_face"] for m in working_data],
                                              heads=[m["head"] for m in working_data],
                                              invAffMats=[m["invAffMats"] for m in working_data],
                                              mels=mels,
                                              wav_frames=wav_batch_frames
                                              )
                else:
                    batch_data = BatchSimple(num=size,
                                             u8bgr_frames=[m["bg"] for m in working_data],
                                             body_coords=[m["body_coord"] for m in working_data],
                                             body_faces=[m["body_face"] for m in working_data],
                                             mels=mels,
                                             wav_frames=wav_batch_frames
                                             )
                self.put(batch_data)
        self.working_audio = None

    def close(self):
        self.__queue.close()

class BatchSimple:
    def __init__(self, num, u8bgr_frames, body_coords, body_faces, mels, wav_frames):
        if num != len(u8bgr_frames):
            raise Exception("帧数不等于该批次数！")
        self.num = num
        self.frames = np.asarray(u8bgr_frames, dtype=np.uint8)
        self.coords = np.asarray(body_coords)
        self.faces = np.asarray(body_faces)
        self.mels = mels
        self.wav_frames = wav_frames


class BatchEnhance:
    def __init__(self, num, u8bgr_frames, head_coords, head_faces, heads, invAffMats, mels, wav_frames):
        if num != len(u8bgr_frames):
            raise Exception("帧数不等于该批次数！")
        self.num = num
        self.frames = np.asarray(u8bgr_frames, dtype=np.uint8)
        self.invAffMats = np.asarray(invAffMats)
        self.heads = np.asarray(heads)
        self.coords = np.asarray(head_coords)
        self.faces = np.asarray(head_faces)
        self.mels = mels
        self.wav_frames = wav_frames


class Wav2LipPool:
    def __init__(self, data_pool: DataPool, checkpoint_path):
        self.__queue = Queue(maxsize=3000)
        self.data_pool = data_pool
        self.checkpoint_path = checkpoint_path
        self.model = None
        # self.model = load_model(args.checkpoint_path)
        self.__doing = False

    def doing(self):
        return self.__doing

    def put(self, data):
        self.__queue.put(data)

    def get(self):
        return self.__queue.get()

    def full(self):
        return self.__queue.full()

    def qsize(self):
        return self.__queue.qsize()

    def run(self, height, width):
        # 不能跨进程携带model参数
        self.model = load_model(self.checkpoint_path)
        while True:
            try:
                batch_data = self.data_pool.get(block=True, timeout=5)
                self.__doing = True
                if isinstance(batch_data, BatchSimple):
                    res = mel2lip(self.model, batch_data.frames, batch_data.faces, batch_data.coords, batch_data.mels)
                elif isinstance(batch_data, BatchEnhance):
                    res = enhance_mel2lip(self.model, batch_data.frames, batch_data.heads, batch_data.invAffMats,
                                          batch_data.faces,
                                          batch_data.coords, batch_data.mels)
                else:
                    continue
                step_by_video_frame = math.ceil(len(batch_data.wav_frames) / len(res))
                if res[0].shape[:2] == (height, width):
                    for i in range(len(res)):
                        self.put([batch_data.wav_frames[i * step_by_video_frame:(i + 1) * step_by_video_frame], res[i]])
                else:
                    for i in range(len(res)):
                        self.put([batch_data.wav_frames[i * step_by_video_frame:(i + 1) * step_by_video_frame],
                                  cv2.resize(res[i], (width, height))
                                  ])
                self.__doing = False
            except:
                print(" EMPTY POOL")
                return


def export_mp4(data_dir, audio_files: List, enhance, width, height, batch_size, data_pool_size, checkpoint_path,
               outfile):
    batch_pool = DataPool(data_dir, audio_files.copy(), batch_size, data_pool_size)
    wav2lip_pool = Wav2LipPool(batch_pool, checkpoint_path)
    batch_pool_thread = Thread(target=batch_pool.run, args=(enhance,), daemon=True, name="batch_pool")
    batch_pool_thread.start()
    wav2lip_process = Process(target=wav2lip_pool.run, args=(height, width), daemon=True, name="wav2lip")
    wav2lip_process.start()
    while wav2lip_pool.qsize() <= 20:
        time.sleep(0.5)

    video_out = cv2.VideoWriter(r"temp/export.mp4", cv2.VideoWriter_fourcc(*"avc1"), int(batch_pool.fps), (width, height))
    wavfile = wave.open(r"temp/export.wav", "wb")
    frame_count = 0
    wavfile.setnchannels(1)
    wavfile.setsampwidth(2)
    wavfile.setframerate(16000)
    while batch_pool.working_audio or wav2lip_pool.qsize() > 0 or batch_pool.qsize() > 0 or wav2lip_pool.doing():
        # print(f"data:{batch_pool.qsize()},lip:{wav2lip_pool.qsize()}")
        wav_frams, u8bgr_img = wav2lip_pool.get()
        video_out.write(u8bgr_img)
        wavfile.writeframes(wav_frams.tobytes())
        frame_count += 1
    else:
        print(frame_count, batch_pool.working_audio, wav2lip_pool.qsize(), batch_pool.qsize())
        video_out.release()
        wavfile.close()
        batch_pool.close()
        batch_pool_thread.join()
        time.sleep(10)
        wav2lip_process.terminate()
        wav2lip_process.close()
    import subprocess
    cmd = fr'ffmpeg -y -i temp/export.wav -i temp/export.mp4 {outfile}'
    subprocess.call(cmd)


def run_stream(data_dir, audio_files, enhance, width, height, batch_size, data_pool_size, checkpoint_path):
    running = True

    def play_audio(queue):
        p = pyaudio.PyAudio()
        stream = p.open(
            rate=16000,
            channels=1,
            format=8,
            output=True,
            output_device_index=2,
        )
        stream.start_stream()
        while queue.qsize() <= 0:
            time.sleep(0.1)
        while running:
            stream.write(queue.get(block=True))
        stream.close()

    batch_pool = DataPool(data_dir, audio_files, batch_size, data_pool_size)
    wav2lip_pool = Wav2LipPool(batch_pool, checkpoint_path)
    batch_pool_thread = Thread(target=batch_pool.run, args=(enhance,), daemon=True, name="batch_pool")
    batch_pool_thread.start()
    wav2lip_process = Process(target=wav2lip_pool.run, args=(height, width), daemon=True, name="wav2lip")
    wav2lip_process.start()
    # 非enhance，生产速度跟得上消费速度，可以 while wav2lip_pool.qsize() <= 0:
    # enhance，生产速度远远跟不上消费速度，建议 while not wav2lip_pool.full():
    while wav2lip_pool.qsize() <= 100:
        time.sleep(0.5)
    with pyvirtualcam.Camera(width=width, height=height, fps=batch_pool.fps, print_fps=True) as cam:
        audio_tmp = queue.Queue(maxsize=3000)
        audio_thread = Thread(target=play_audio, args=(audio_tmp,), daemon=True, name="pyaudio_stream")
        audio_thread.start()
        frame_count = 0
        while batch_pool.working_audio or wav2lip_pool.qsize() > 0 or batch_pool.qsize() > 0 or wav2lip_pool.doing():
            # print(f"data:{batch_pool.qsize()},lip:{wav2lip_pool.qsize()}")
            wav_frams, u8bgr_img = wav2lip_pool.get()
            cam.send(u8bgr_img[..., ::-1])
            audio_tmp.put(wav_frams.tobytes())
            cam.sleep_until_next_frame()
            frame_count += 1
        else:
            print(frame_count, batch_pool.working_audio, wav2lip_pool.qsize(), batch_pool.qsize())
            running = False
            cam.close()
            audio_thread.join()
            batch_pool_thread.join()
            wav2lip_process.terminate()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    if args.outfile == 'obs':
        run_stream(data_dir=args.data_dir, audio_files=args.audio_files, enhance=args.enhance,
                   checkpoint_path=args.checkpoint_path,
                   width=args.width, height=args.height, batch_size=args.batch_size, data_pool_size=args.data_pool_size)
    elif args.outfile.lower().endswith(".mp4"):
        export_mp4(data_dir=args.data_dir, audio_files=args.audio_files, enhance=args.enhance,
                   checkpoint_path=args.checkpoint_path, outfile=args.outfile,
                   width=args.width, height=args.height, batch_size=args.batch_size, data_pool_size=args.data_pool_size)
    else:
        raise Exception("不支持的视频导出格式")

