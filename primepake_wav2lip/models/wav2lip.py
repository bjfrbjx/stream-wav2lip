import torch
from torch import nn
from torch.nn import functional as F

from .conv import Conv2dTranspose, Conv2d


class Wav2Lip_96(nn.Module):
    def __init__(self):
        super(Wav2Lip_96, self).__init__()

        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(6, 16, kernel_size=7, stride=1, padding=3)),  # 96,96

            nn.Sequential(Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 48,48
                          Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 24,24
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 12,12
                          Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 6,6
                          Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 3,3
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), ),

            nn.Sequential(Conv2d(512, 512, kernel_size=3, stride=1, padding=0),  # 1, 1
                          Conv2d(512, 512, kernel_size=1, stride=1, padding=0)), ])

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0), )

        self.face_decoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(512, 512, kernel_size=1, stride=1, padding=0), ),

            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=1, padding=0),  # 3,3
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), ),

            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), ),  # 6, 6

            nn.Sequential(Conv2dTranspose(768, 384, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True), ),  # 12, 12

            nn.Sequential(Conv2dTranspose(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), ),  # 24, 24

            nn.Sequential(Conv2dTranspose(320, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), ),  # 48, 48

            nn.Sequential(Conv2dTranspose(160, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), ), ])  # 96,96

        self.output_block = nn.Sequential(Conv2d(80, 32, kernel_size=3, stride=1, padding=1),
                                          nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
                                          nn.Sigmoid())

    def forward(self, audio_sequences, face_sequences):
        # audio_sequences = (B, T, 1, 80, 16)
        B = audio_sequences.size(0)

        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)

        audio_embedding = self.audio_encoder(audio_sequences)  # B, 512, 1, 1

        feats = []
        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)
            feats.append(x)

        x = audio_embedding
        for f in self.face_decoder_blocks:
            x = f(x)
            try:
                x = torch.cat((x, feats[-1]), dim=1)
            except Exception as e:
                print(x.size())
                print(feats[-1].size())
                raise e

            feats.pop()

        x = self.output_block(x)

        if input_dim_size > 4:
            x = torch.split(x, B, dim=0)  # [(B, C, H, W)]
            outputs = torch.stack(x, dim=2)  # (B, C, T, H, W)

        else:
            outputs = x

        return outputs


class Wav2Lip_288(nn.Module):
    def __init__(self):
        super(Wav2Lip_288, self).__init__()

        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(6, 16, kernel_size=7, stride=1, padding=3)),  # 288, 288

            nn.Sequential(Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
                          Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True), ),  # 144,144

            nn.Sequential(Conv2d(32, 32, kernel_size=3, stride=2, padding=1),  # 72,72
                          Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 36,36
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 18,18
                          Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 9,9
                          Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 5,5
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), ),

            nn.Sequential(Conv2d(512, 512, kernel_size=3, stride=1, padding=0),  # 3, 3
                          Conv2d(512, 512, kernel_size=1, stride=1, padding=0)),

            nn.Sequential(Conv2d(512, 512, kernel_size=3, stride=1, padding=0),  # 1, 1
                          Conv2d(512, 512, kernel_size=1, stride=1, padding=0)), ])

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0), )

        self.face_decoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(512, 512, kernel_size=1, stride=1, padding=0), ),  # 1,1

            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=1, padding=0),  # 3,3
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), ),

            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=2, padding=1),
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), ),  # 5, 5

            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=2, padding=1),
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), ),  # 9, 9

            nn.Sequential(Conv2dTranspose(768, 384, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True), ),  # 18, 18

            nn.Sequential(Conv2dTranspose(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), ),  # 36, 36

            nn.Sequential(Conv2dTranspose(320, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), ),  # 72, 72

            nn.Sequential(Conv2dTranspose(160, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), ),  # 144,144

            nn.Sequential(Conv2dTranspose(96, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), ),  # 288,288
        ])

        self.output_block = nn.Sequential(Conv2d(80, 32, kernel_size=3, stride=1, padding=1),
                                          nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
                                          nn.Sigmoid())

    def forward(self, audio_sequences, face_sequences):
        # audio_sequences = (B, T, 1, 80, 16)
        B = audio_sequences.size(0)

        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)

        audio_embedding = self.audio_encoder(audio_sequences)  # B, 512, 1, 1

        feats = []
        x = face_sequences

        for f in self.face_encoder_blocks:
            x = f(x)
            feats.append(x)

        x = audio_embedding

        for f in self.face_decoder_blocks:
            x = f(x)
            try:
                x = torch.cat((x, feats[-1]), dim=1)
            except Exception as e:
                print(x.size())
                print(feats[-1].size())
                raise e

            feats.pop()

        x = self.output_block(x)

        if input_dim_size > 4:
            x = torch.split(x, B, dim=0)  # [(B, C, H, W)]
            outputs = torch.stack(x, dim=2)  # (B, C, T, H, W)

        else:
            outputs = x

        return outputs


class Wav2Lip_512(nn.Module):
    def __init__(self, audio_encoder=None):
        super(Wav2Lip_512, self).__init__()

        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(6, 6, kernel_size=7, stride=1, padding=3, act="leaky"),
                          Conv2d(6, 6, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
                          Conv2d(6, 6, kernel_size=3, stride=1, padding=1, residual=True, act="leaky")),

            nn.Sequential(Conv2d(6, 6, kernel_size=3, stride=2, padding=1, act="leaky"),  # 512, 512
                          Conv2d(6, 6, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
                          Conv2d(6, 6, kernel_size=3, stride=1, padding=1, residual=True, act="leaky")),

            nn.Sequential(Conv2d(6, 8, kernel_size=3, stride=2, padding=1, act="leaky"),  # 256, 256
                          Conv2d(8, 8, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
                          Conv2d(8, 8, kernel_size=3, stride=1, padding=1, residual=True, act="leaky")),

            nn.Sequential(Conv2d(8, 16, kernel_size=3, stride=2, padding=1, act="leaky"),  # 128, 128
                          Conv2d(16, 16, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
                          Conv2d(16, 16, kernel_size=3, stride=1, padding=1, residual=True, act="leaky")),

            nn.Sequential(Conv2d(16, 32, kernel_size=3, stride=2, padding=1, act="leaky"),  # 64, 64
                          Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
                          Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True, act="leaky")),

            nn.Sequential(Conv2d(32, 64, kernel_size=3, stride=2, padding=1, act="leaky"),  # 32, 32
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True, act="leaky")),

            nn.Sequential(Conv2d(64, 128, kernel_size=3, stride=2, padding=1, act="leaky"),  # 16, 16
                          Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
                          Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True, act="leaky")),

            nn.Sequential(Conv2d(128, 256, kernel_size=3, stride=2, padding=1, act="leaky"),  # 8, 8
                          Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
                          Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True, act="leaky")),

            nn.Sequential(Conv2d(256, 512, kernel_size=3, stride=2, padding=1, act="leaky"),  # 4, 4
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"), ),

            ###################
            # Modified blocks
            ##################
            nn.Sequential(Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, act="leaky"),  # 2, 2
                          Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
                          Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, residual=True, act="leaky")), ])
        ##################

        if audio_encoder is None:
            self.audio_encoder = nn.Sequential(
                Conv2d(1, 32, kernel_size=3, stride=1, padding=1, act="leaky"),
                Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
                Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),

                Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1, act="leaky"),
                Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
                Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),

                Conv2d(64, 128, kernel_size=3, stride=3, padding=1, act="leaky"),
                Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
                Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),

                Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1, act="leaky"),
                Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
                Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),

                ###################
                # Modified blocks
                ##################
                Conv2d(256, 512, kernel_size=3, stride=1, padding=1, act="leaky"),
                Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
                Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),

                Conv2d(512, 1024, kernel_size=3, stride=1, padding=0, act="relu"),
                Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0, act="relu"))
            ##################
        else:
            self.audio_encoder = audio_encoder

        for p in self.audio_encoder.parameters():
            p.requires_grad = False

        self.face_decoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0, act="leaky"), ),  #

            ###################
            # Modified blocks
            ##################
            nn.Sequential(
                Conv2dTranspose(2048, 1024, kernel_size=3, stride=2, padding=1, output_padding=1, act="leaky"),
                # + 1024
                Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"), ),  #

            nn.Sequential(
                Conv2dTranspose(1536, 1024, kernel_size=3, stride=2, padding=1, output_padding=1, act="leaky"),
                Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
                Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"), ),

            nn.Sequential(Conv2dTranspose(1280, 768, kernel_size=3, stride=2, padding=1, output_padding=1, act="leaky"),
                          Conv2d(768, 768, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
                          Conv2d(768, 768, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"), ),

            nn.Sequential(Conv2dTranspose(896, 512, kernel_size=3, stride=2, padding=1, output_padding=1, act="leaky"),
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"), ),
            ##################

            nn.Sequential(Conv2dTranspose(576, 256, kernel_size=3, stride=2, padding=1, output_padding=1, act="leaky"),
                          Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
                          Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"), ),

            nn.Sequential(Conv2dTranspose(288, 128, kernel_size=3, stride=2, padding=1, output_padding=1, act="leaky"),
                          Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
                          Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"), ),

            nn.Sequential(Conv2dTranspose(144, 80, kernel_size=3, stride=2, padding=1, output_padding=1, act="leaky"),
                          Conv2d(80, 80, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
                          Conv2d(80, 80, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"), ),

            nn.Sequential(Conv2dTranspose(88, 64, kernel_size=3, stride=2, padding=1, output_padding=1, act="leaky"),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"), ),

            nn.Sequential(Conv2dTranspose(70, 50, kernel_size=3, stride=2, padding=1, output_padding=1, act="leaky"),
                          Conv2d(50, 50, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"),
                          Conv2d(50, 50, kernel_size=3, stride=1, padding=1, residual=True, act="leaky"), ), ])

        self.output_block = nn.Sequential(Conv2d(56, 32, kernel_size=3, stride=1, padding=1, act="leaky"),
                                          Conv2d(32, 16, kernel_size=3, stride=1, padding=1, act="leaky"),
                                          nn.Conv2d(16, 3, kernel_size=1, stride=1, padding=0),
                                          nn.Tanh())

    def freeze_audio_encoder(self):
        for p in self.audio_encoder.parameters():
            p.requires_grad = False

    def forward(self, audio_sequences, face_sequences):

        B = audio_sequences.size(0)

        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)

        audio_embedding = self.audio_encoder(audio_sequences)
        audio_embedding = audio_embedding.detach()
        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)

        feats = []
        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)
            feats.append(x)

        x = audio_embedding
        cnt = 0
        for f in self.face_decoder_blocks:
            x = f(x)
            try:
                x = torch.cat((x, feats[-1]), dim=1)
            except Exception as e:
                print(x.size())
                print(feats[-1].size())
                raise e
            feats.pop()
            cnt += 1

        x = self.output_block(x)
        if input_dim_size > 4:
            x = torch.split(x, B, dim=0)  # [(B, C, H, W)]
            outputs = torch.stack(x, dim=2)  # (B, C, T, H, W)

        else:
            outputs = x

        return outputs
