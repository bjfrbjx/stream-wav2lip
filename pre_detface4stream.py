import os
import shutil
import subprocess

import cv2
import numpy as np
import torch
from PIL import Image
from face_alignment import LandmarksType
from facexlib.utils.face_restoration_helper import get_center_face

from utils import BATCH_GFP, FaceAlignment, time_print, osp

cuda_memery = 6 * 1024 * 1024 * 1024
device="cuda"
if not torch.cuda.is_available():
    device = "cpu"

###################### HPARAMS #########################
PADS = [0, 15, 0, 0]
DEVICE = device
gfp_worker = BATCH_GFP(device)

detector = FaceAlignment(LandmarksType.TWO_D, flip_input=False, device=DEVICE,face_detector="blazeface")


def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i: i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes


def face_detect(images,nosmooth=False):
    batch_size = 16

    while True:
        predictions = []
        try:
            for i in range(0, len(images), batch_size):
                y = detector.get_detections_for_batch(np.array(images[i:i + batch_size]))
                predictions.extend(y)
        except RuntimeError as e:
            import traceback
            traceback.print_exc()
            if batch_size == 1:
                raise RuntimeError('Image too big to run face detection on GPU')
            batch_size //= 2
            print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = PADS
    for rect, image in zip(predictions, images):
        if rect is None:
            from PIL import Image
            Image.fromarray(image[...,::-1],"RGB").show()
            raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)

        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    if not nosmooth:
        boxes = get_smoothened_boxes(boxes, T=5)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]
    return results

@time_print
def crop_head_with_affine(u8bgr_frames: np.ndarray):
    @time_print
    def batched_detect_faces():
        with torch.no_grad():
            final_bounding_boxes, final_landmarks = gfp_worker.face_helper.face_det.batched_detect_faces(torch.from_numpy(u8bgr_frames).to(DEVICE), 0.97)
        return final_bounding_boxes, final_landmarks
    face_template = np.array([[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
                              [201.26117, 371.41043], [313.08905, 371.15118]])
    all_crop_heads = []
    inverse_affine_matrices = []
    final_bounding_boxes, final_landmarks=batched_detect_faces()
    for i, boxes, lmks in zip(range(len(final_bounding_boxes)), final_bounding_boxes, final_landmarks):
        landmarks = []
        det_faces = []
        if len(boxes) != len(lmks):
            raise Exception("boxes,landmarks 不对应")
        for b in range(len(boxes)):
            eye_dist = np.linalg.norm([lmks[b][0] - lmks[b][2], lmks[b][1] - lmks[b][3]])
            if eye_dist < 5:
                continue
            landmarks.append(np.array([[lmks[b][i], lmks[b][i + 1]] for i in range(0, 10, 2)]))
            det_faces.append(boxes[b])
        det_face, center_idx = get_center_face(det_faces, u8bgr_frames.shape[1], u8bgr_frames.shape[2])
        affine_matrix = cv2.estimateAffinePartial2D(landmarks[center_idx], face_template, method=cv2.LMEDS)[0]
        inverse_affine_matrices.append(np.expand_dims(cv2.invertAffineTransform(affine_matrix), axis=0))
        cropped_head = cv2.warpAffine(u8bgr_frames[i], affine_matrix, (512, 512), borderMode=cv2.BORDER_REFLECT,
                                      borderValue=(135, 133, 132))
        all_crop_heads.append(cropped_head.reshape((1, 512, 512, 3)))
    u8bgr_crops = np.concatenate(all_crop_heads)
    f32_invAffMats = np.concatenate(inverse_affine_matrices)
    return u8bgr_crops, f32_invAffMats

@time_print
def pre_face_process(u8bgrframes: np.ndarray,nosmooth=False,face_size=96):
    img_batch, coords = [], []
    face_det_results = face_detect(u8bgrframes,nosmooth=nosmooth)

    for i in range(len(u8bgrframes)):
        face, coord = face_det_results[i]
        img_batch.append(cv2.resize(face, (face_size, face_size)))
        coords.append(coord)
    u8bgr_faces = np.asarray(img_batch, dtype=np.uint8)

    return u8bgr_faces, np.asarray(coords, np.uint16)


mel_step_size = 16
frame_batch = 15
print('Using {} for inference.'.format(DEVICE))

head_coords = []
body_coords = []
f32_invAffMats_batchs = []


def working(working_idx, working_frames,out_dir=".",nosmooth=False,face_size=96):
    frames = np.concatenate([np.expand_dims(u, axis=0) for u in working_frames], axis=0, dtype=np.uint8)
    print(f"\n crop_head_with_affine: {frames.shape}")
    # 躯体中取出头和复位数据
    croped_heads, f32_invAffMats = crop_head_with_affine(frames)
    for i, croped_head in enumerate(croped_heads):
        #cv2.imwrite(f"{out_dir}/heads/{working_idx[i]:05d}.jpg", croped_head)
        Image.fromarray(croped_head[..., ::-1], "RGB").save(f"{out_dir}/heads/{working_idx[i]:05d}.jpg")
    f32_invAffMats_batchs.append(f32_invAffMats)

    # 躯体中取出脸和复位数据
    body_faces, body_coord = pre_face_process(frames,nosmooth,face_size)
    for i, body_face in enumerate(body_faces):
        #cv2.imwrite(f"{out_dir}/body_faces/{working_idx[i]:05d}.jpg", body_face)
        Image.fromarray(body_face[..., ::-1], "RGB").save(f"{out_dir}/body_faces/{working_idx[i]:05d}.jpg")
    body_coords.append(body_coord)

    # 头中取出脸和复位数据
    head_faces, head_coord = pre_face_process(croped_heads,nosmooth,face_size)
    for i, head_face in enumerate(head_faces):
        #cv2.imwrite(f"{out_dir}/head_faces/{working_idx[i]:05d}.jpg", head_face)
        Image.fromarray(head_face[...,::-1],"RGB").save(f"{out_dir}/head_faces/{working_idx[i]:05d}.jpg")
    head_coords.append(head_coord)

template = 'ffmpeg -loglevel panic -y -i {} -strict -2 {}'

def main(face,nosmooth,out_dir,face_size):
    """
    输入源视频face，
    指定输出目录out_dir
    -------------
    转存out_dir下 图片和npz【因为图片太大，不能存进npz】
    npz关键信息：
    fps:原视频帧率
    croped_head：GFPGAN剪下的头
    invAffMats：逆仿射矩阵，用于恢复head在原图中的角度和位置
    concat_faces：wav2lip剪下的脸【掩码半脸+整张脸】
    coords：脸在head的相对位置

    """
    if not osp.isfile(face):
        raise ValueError('--face argument must be a valid path')

    if face.split('.')[-1].lower() not in ['mp4', 'avi', 'mov','mpg']:
        raise ValueError('--face argument must be a video file')

    video_stream = cv2.VideoCapture(face)
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    global frame_batch
    if frame_batch is None:
        frame_batch=int(video_stream.get(cv2.CAP_PROP_FPS))
    print('Reading video frames...')
    idx = 0
    if not osp.exists(out_dir):
        os.mkdir(out_dir)
    if not osp.exists(f"{out_dir}/bg"):
        os.mkdir(f"{out_dir}/bg")
    if not osp.exists(f"{out_dir}/head_faces"):
        os.mkdir(f"{out_dir}/head_faces")
    if not osp.exists(f"{out_dir}/body_faces"):
        os.mkdir(f"{out_dir}/body_faces")
    if not osp.exists(f"{out_dir}/heads"):
        os.mkdir(f"{out_dir}/heads")
    working_frames = []
    working_idx = []
    wavpath=f"{out_dir}/audio.wav"
    command = template.format(face, wavpath)
    subprocess.call(command, shell=True)
    while True:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        # cv2.imwrite(f"{args.out_dir}/bg/{idx:05d}.jpg", frame)
        Image.fromarray(frame[...,::-1],"RGB").save(f"{out_dir}/bg/{idx:05d}.jpg")
        working_frames.append(frame)
        working_idx.append(idx)
        if len(working_frames) == frame_batch:
            working(working_idx, working_frames,out_dir=out_dir,face_size=face_size,nosmooth=nosmooth)
            working_idx.clear()
            working_frames.clear()
        idx += 1
    if working_idx:
        working(working_idx, working_frames,out_dir=out_dir,face_size=face_size,nosmooth=nosmooth)


    invAffMats = np.concatenate(f32_invAffMats_batchs)
    export_body_coords = np.concatenate(body_coords)
    export_head_coords = np.concatenate(head_coords)
    np.savez(f"{out_dir}/face_det.npz",
             fps=np.uint8(int(fps)),
             head_coords=export_head_coords,
             invAffMats=invAffMats,
             body_coords=export_body_coords,
             num=len(export_head_coords)
             )
    fake_speaker=f"{out_dir}/../speaker"
    if not osp.exists(fake_speaker):
        os.mkdir(fake_speaker)
    vname=osp.basename(face).split(".")[0]
    faceFrames_dir=f"{fake_speaker}/{vname}"
    shutil.copytree(f"{out_dir}/body_faces",faceFrames_dir,dirs_exist_ok=True)
    shutil.copy(wavpath,faceFrames_dir)

def batch(face_dir,nosmooth,out_dir,face_size):
    for video in os.listdir(face_dir):
        out=osp.join(out_dir,video.split(".")[0])
        if not osp.exists(out):
            os.mkdir(out)
        video=osp.join(face_dir,video)
        main(face=video, nosmooth=nosmooth, out_dir=out, face_size=face_size)

if __name__ == '__main__':
    #main(face=r"\\Vp05-daily01\新建文件夹\s2.mpg_vcd\s2\bbaf1n.mpg",nosmooth=False,out_dir=r"\\Vp05-daily01\新建文件夹\s2.mpg_vcd\bbaf1n",face_size=288)
    batch(face_dir=r"\\Vp05-daily01\新建文件夹\s2.mpg_vcd\s2", nosmooth=False, out_dir=r"\\Vp05-daily01\新建文件夹\s2.mpg_vcd", face_size=288)
