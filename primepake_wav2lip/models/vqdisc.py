import functools

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.functional import binary_cross_entropy

from .conv import nonorm_Conv2d


def count_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params


class ActNorm(nn.Module):
    def __init__(self, num_features, logdet=False, affine=True,
                 allow_reverse_init=False):
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.allow_reverse_init = allow_reverse_init

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        if len(input.shape) == 2:
            input = input[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        _, _, height, width = input.shape

        if self.training and self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        h = self.scale * (input + self.loc)

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)

        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height * width * torch.sum(log_abs)
            logdet = logdet * torch.ones(input.shape[0]).to(input)
            return h, logdet

        return h

    def reverse(self, output):
        if self.training and self.initialized.item() == 0:
            if not self.allow_reverse_init:
                raise RuntimeError(
                    "Initializing ActNorm in reverse direction is "
                    "disabled by default. Use allow_reverse_init=True to enable."
                )
            else:
                self.initialize(output)
                self.initialized.fill_(1)

        if len(output.shape) == 2:
            output = output[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        h = output / self.scale - self.loc

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        return h


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)
        self.apply(weights_init)

    def to_2d(self, face_sequences):
        face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)
        return face_sequences

    def forward(self, input):
        """Standard forward."""
        input = self.to_2d(input)
        return self.main(input)


class NLayerDiscriminator_LIP(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator_LIP, self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)
        self.apply(weights_init)

    def get_lower_half(self, face_sequences):
        return face_sequences[:, :, face_sequences.size(2) // 2:]

    def to_2d(self, face_sequences):
        B = face_sequences.size(0)
        face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)
        return face_sequences

    def forward(self, input):
        """Standard forward."""
        input = self.to_2d(input)
        input = self.get_lower_half(input)
        return self.main(input)


class Wav2Lip_disc_qual_96(nn.Module):
    def __init__(self):
        super(Wav2Lip_disc_qual_96, self).__init__()

        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(nonorm_Conv2d(3, 32, kernel_size=7, stride=1, padding=3)),  # 48,96

            nn.Sequential(nonorm_Conv2d(32, 64, kernel_size=5, stride=(1, 2), padding=2),  # 48,48
                          nonorm_Conv2d(64, 64, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(nonorm_Conv2d(64, 128, kernel_size=5, stride=2, padding=2),  # 24,24
                          nonorm_Conv2d(128, 128, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(nonorm_Conv2d(128, 256, kernel_size=5, stride=2, padding=2),  # 12,12
                          nonorm_Conv2d(256, 256, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(nonorm_Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 6,6
                          nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1)),

            nn.Sequential(nonorm_Conv2d(512, 512, kernel_size=3, stride=2, padding=1),  # 3,3
                          nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1), ),

            nn.Sequential(nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=0),  # 1, 1
                          nonorm_Conv2d(512, 512, kernel_size=1, stride=1, padding=0)), ])

        self.binary_pred = nn.Sequential(nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0), nn.Sigmoid())
        self.label_noise = .0

    def get_lower_half(self, face_sequences):
        return face_sequences[:, :, face_sequences.size(2) // 2:]

    def to_2d(self, face_sequences):
        B = face_sequences.size(0)
        face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)
        return face_sequences

    def perceptual_forward(self, false_face_sequences):
        false_face_sequences = self.to_2d(false_face_sequences)
        false_face_sequences = self.get_lower_half(false_face_sequences)

        false_feats = false_face_sequences
        for f in self.face_encoder_blocks:
            false_feats = f(false_feats)

        false_pred_loss = binary_cross_entropy(self.binary_pred(false_feats).view(len(false_feats), -1),
                                               torch.ones((len(false_feats), 1)).cuda())

        return false_pred_loss

    def forward(self, face_sequences):
        face_sequences = self.to_2d(face_sequences)
        face_sequences = self.get_lower_half(face_sequences)

        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)

        return self.binary_pred(x).view(len(x), -1)


class Wav2Lip_disc_qual_288(nn.Module):
    def __init__(self):
        super(Wav2Lip_disc_qual_288, self).__init__()

        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(nonorm_Conv2d(3, 32, kernel_size=7, stride=1, padding=3)),  # 144,288

            nn.Sequential(nonorm_Conv2d(32, 64, kernel_size=5, stride=(1, 2), padding=2),  # 144, 144
                          nonorm_Conv2d(64, 64, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(nonorm_Conv2d(64, 128, kernel_size=5, stride=2, padding=2),  # 72,72
                          nonorm_Conv2d(128, 128, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(nonorm_Conv2d(128, 128, kernel_size=5, stride=2, padding=2),  # 36,36
                          nonorm_Conv2d(128, 128, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(nonorm_Conv2d(128, 256, kernel_size=5, stride=2, padding=2),  # 18,18
                          nonorm_Conv2d(256, 256, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(nonorm_Conv2d(256, 256, kernel_size=5, stride=2, padding=2),  # 9,9
                          nonorm_Conv2d(256, 256, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(nonorm_Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 5,5
                          nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1)),

            nn.Sequential(nonorm_Conv2d(512, 512, kernel_size=3, stride=2, padding=1),  # 3,3
                          nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1), ),

            nn.Sequential(nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=0),  # 1, 1
                          nonorm_Conv2d(512, 512, kernel_size=1, stride=1, padding=0)), ])

        self.binary_pred = nn.Sequential(nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0), nn.Sigmoid())
        self.label_noise = .0

    def get_lower_half(self, face_sequences):
        return face_sequences[:, :, face_sequences.size(2) // 2:]

    def to_2d(self, face_sequences):
        B = face_sequences.size(0)
        face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)
        return face_sequences

    def perceptual_forward(self, false_face_sequences):
        false_face_sequences = self.to_2d(false_face_sequences)
        false_face_sequences = self.get_lower_half(false_face_sequences)

        false_feats = false_face_sequences
        for f in self.face_encoder_blocks:
            false_feats = f(false_feats)

        false_pred_loss = F.binary_cross_entropy(self.binary_pred(false_feats).view(len(false_feats), -1),
                                                 torch.ones((len(false_feats), 1)).cuda())

        return false_pred_loss

    def forward(self, face_sequences):
        face_sequences = self.to_2d(face_sequences)
        face_sequences = self.get_lower_half(face_sequences)

        x = face_sequences

        for f in self.face_encoder_blocks:
            x = f(x)

        return self.binary_pred(x).view(len(x), -1)


class Wav2Lip_disc_qual_512(nn.Module):
    def __init__(self):
        super(Wav2Lip_disc_qual_512, self).__init__()
        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(nonorm_Conv2d(3, 8, kernel_size=7, stride=1, padding=3)),  # 512, 512

            nn.Sequential(nonorm_Conv2d(8, 16, kernel_size=5, stride=2, padding=2),  # 256, 256
                          nonorm_Conv2d(16, 16, kernel_size=5, stride=1, padding=2),
                          nonorm_Conv2d(16, 16, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(nonorm_Conv2d(16, 32, kernel_size=5, stride=2, padding=2),  # 128, 128
                          nonorm_Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
                          nonorm_Conv2d(32, 32, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(nonorm_Conv2d(32, 64, kernel_size=5, stride=2, padding=2),  # 64, 64
                          nonorm_Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
                          nonorm_Conv2d(64, 64, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(nonorm_Conv2d(64, 128, kernel_size=5, stride=2, padding=2),  # 32, 32
                          nonorm_Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
                          nonorm_Conv2d(128, 128, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(nonorm_Conv2d(128, 256, kernel_size=5, stride=2, padding=2),  # 16, 16
                          nonorm_Conv2d(256, 256, kernel_size=5, stride=1, padding=2),
                          nonorm_Conv2d(256, 256, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(nonorm_Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 8, 8
                          nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                          nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1)),

            nn.Sequential(nonorm_Conv2d(512, 512, kernel_size=3, stride=2, padding=1),  # 4, 4
                          nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                          nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1), ),

            # Modified Blocks
            nn.Sequential(nonorm_Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),  # 2, 2
                          nonorm_Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1), ),

            nn.Sequential(nonorm_Conv2d(1024, 1024, kernel_size=2, stride=1, padding=0, norm=False),  # 1, 1
                          nonorm_Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0, norm=False)), ])

        self.binary_pred = nn.Sequential(nn.Conv2d(1024, 1, kernel_size=1, stride=1, padding=0),
                                         nn.Sigmoid())
        self.label_noise = .0
        self.bce_loss = nn.BCELoss()

    def get_lower_half(self, face_sequences):
        return face_sequences[:, :, face_sequences.size(2) // 2:]

    def to_2d(self, face_sequences):
        B = face_sequences.size(0)
        face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)
        return face_sequences

    def perceptual_forward(self, false_face_sequences):
        false_face_sequences = self.to_2d(false_face_sequences)
        false_face_sequences = self.get_lower_half(false_face_sequences)

        false_feats = false_face_sequences
        for f in self.face_encoder_blocks:
            false_feats = f(false_feats)

        # TODO: change back to cuda
        # false_pred_loss = F.binary_cross_entropy_with_logits(torch.clamp(self.binary_pred(false_feats).view(len(false_feats), -1), min=-10, max=10), torch.ones((len(false_feats), 1)).cuda())
        pred = self.binary_pred(false_feats).view(len(false_feats), -1)
        target = torch.ones((len(false_feats), 1)).cuda()
        false_pred_loss = F.binary_cross_entropy(pred, target)
        # false_pred_loss = torch.mean(torch.square(pred - target))
        # false_pred_loss = F.binary_cross_entropy(self.binary_pred(false_feats).view(len(false_feats), -1),
        # torch.ones((len(false_feats), 1)).cpu())

        return false_pred_loss

    def forward(self, face_sequences):
        face_sequences = self.to_2d(face_sequences)
        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)

        return self.binary_pred(x).view(len(x), -1)
