import torch


def check_dataset():
    from trains import  Wav2lip_Dataset
    def case(size):
        p=Wav2lip_Dataset(r"E:\work\learn\stream-wav2lip\data_dir\val.txt",(size,size))
        for i in p.get():
            print(i.shape)
        print("------\n")
    case(96)
    case(288)
    case(384)

def check_wav2lip():
    from primepake_wav2lip.models.wav2lip import Wav2Lip_96 ,Wav2Lip_288
    from primepake_wav2lip.models.sam import Wav2Lip_384SAM

    def case(model, size):
        x=torch.ones(size=(2, 6, 5, size,size), dtype=torch.float32)
        indiv_mels = torch.ones(size=(2, 5,1,80,16), dtype=torch.float32)
        mel = torch.ones(size=(2, 1, 80, 16), dtype=torch.float32)
        y=torch.ones(size=(2, 3, 5, size, size), dtype=torch.float32)
        g = model(indiv_mels, x)
        print(g.shape)

    case(Wav2Lip_96(), 96)
    case(Wav2Lip_288(), 288)
    case(Wav2Lip_384SAM(is_sam=False), 384)
def check_syncnet():
    from primepake_wav2lip.models.syncnet import SyncNet_color_144,SyncNet_color_96,SyncNet_color_384, SyncNet_color_288
    def case(model, size):
        mel = torch.ones(size=(2, 1, 80, 16), dtype=torch.float32)
        x = torch.ones(size=(2, 15, size // 2, size), dtype=torch.float32)
        a, v = model(mel, x)
        print(a.shape, v.shape)
    case(SyncNet_color_96(), 96)
    case(SyncNet_color_144(), 144)
    case(SyncNet_color_288(), 288)
    case(SyncNet_color_384(), 384)


if __name__ == "__main__":
    check_wav2lip()
