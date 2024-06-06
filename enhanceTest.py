from time import time
from utils import BATCH_GFP
import numpy as np
from PIL import Image
batch_gfp = BATCH_GFP("cpu")
u8bgr_heads=np.uint8(Image.open(r"E:\learn\stream-wav2lip\video_data\out_dir\heads\00002.jpg"))[None,:,:,::-1]
u8bgr_bg=np.uint8(Image.open(r"E:\桌面\dsBuffer.jpg"))[None,:,:,::-1]

def test(path,arch):
    facelib = BATCH_GFP("cpu", arch=arch,model_path=path)
    res=None
    print(arch)
    for _ in range(3):
        t1=time()
        res=facelib.enhance(u8bgr_bg[0],only_center_face=True,weight=0.)[-1][...,::-1]
        print(time()-t1)
    Image.fromarray(res,"RGB").show()

if __name__ == '__main__':
    test('gfpgan/weights/codeformer.pth',"CodeFormer")
    test('gfpgan/weights/RestoreFormer.pth',"RestoreFormer")
    test('gfpgan/weights/GFPGANv1.3.pth',"clean")
    test('gfpgan/weights/GFPGANv1.4.pth', "clean")
