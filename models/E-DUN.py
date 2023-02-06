import torch
import torch.nn as nn

from models.denoisingModule import EncodingBlock, EncodingBlockEnd, DecodingBlock, DecodingBlockEnd
from models.textureReconstructionModule import ConvDown, ConvUp


class MoG_DUN(nn.Module):
    def __init__(self, args):
        super(MoG_DUN, self).__init__()

        self.channel0 = args.n_colors
        self.up_factor = args.scale[0]
        self.patch_size = args.patch_size
        self.batch_size = int(args.batch_size / args.n_GPUs)

        self.Encoding_block1 = EncodingBlock(64)
        self.Encoding_block2 = EncodingBlock(64)
        self.Encoding_block3 = EncodingBlock(64)
        self.Encoding_block4 = EncodingBlock(64)

        self.Encoding_block_end = EncodingBlockEnd(64)

        self.Decoding_block1 = DecodingBlock(256)
        self.Decoding_block2 = DecodingBlock(256)
        self.Decoding_block3 = DecodingBlock(256)
        self.Decoding_block4 = DecodingBlock(256)

        self.feature_decoding_end = DecodingBlockEnd(256)

        self.act = nn.ReLU()

        self.construction = nn.Conv2d(64, 3, 3, padding=1)

        G0 = 64
        kSize = 3
        T = 4
        self.Fe_e = nn.ModuleList([nn.Sequential(*[
            nn.Conv2d(3, G0, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        ]) for _ in range(T)])

        self.RNNF = nn.ModuleList([nn.Sequential(*[
            nn.Conv2d((i + 2) * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1),
            self.act,
            nn.Conv2d(64, 3, 3, padding=1)

        ]) for i in range(T)])

        self.Fe_f = nn.ModuleList(
            [nn.Sequential(*[nn.Conv2d((2 * i + 3) * G0, G0, 1, padding=0, stride=1)]) for i in range(T - 1)])

        self.u = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.5)) for _ in range(T)])
        self.eta = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.5)) for _ in range(T)])
        self.gama = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.5)) for _ in range(T)])
        self.delta = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.1)) for _ in range(T)])
        self.gama1 = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.5)) for _ in range(T)])
        self.delta1 = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.1)) for _ in range(T)])
        self.u1 = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.5)) for _ in range(T)])

        self.conv_up = ConvUp(3, self.up_factor)
        self.conv_down = ConvDown(3, self.up_factor)

        self.NLBlock = nn.ModuleList([blockNL.blockNL(3, 15) for _ in range(4)])

    def forward(self, y):  # [batch_size ,3 ,7 ,270 ,480] ;
        Ay = torch.nn.functional.interpolate(y, scale_factor=self.up_factor, mode='bilinear', align_corners=False)
        x = Ay
        fea_list = []
        V_list = []
        outs = []
        for i in range(len(self.Fe_e)):
            fea = self.Fe_e[i](x)
            fea_list.append(fea)
            if i != 0:
                fea = self.Fe_f[i - 1](torch.cat(fea_list, 1))
            encode0, down0 = self.Encoding_block1(fea)
            encode1, down1 = self.Encoding_block2(down0)
            encode2, down2 = self.Encoding_block3(down1)
            encode3, down3 = self.Encoding_block4(down2)

            media_end = self.Encoding_block_end(down3)

            decode3 = self.Decoding_block1(media_end, encode3)
            decode2 = self.Decoding_block2(decode3, encode2)
            decode1 = self.Decoding_block3(decode2, encode1)
            decode0 = self.feature_decoding_end(decode1, encode0)

            fea_list.append(decode0)
            V_list.append(decode0)
            if i == 0:
                decode0 = self.construction(self.act(decode0))
            else:
                decode0 = self.RNNF[i - 1](torch.cat(V_list, 1))
            conv_out = x + decode0

            NL = self.NLBlock[i](x)
            e = NL - x
            e = e - self.delta1[i] * (
                    self.u1[i] * self.conv_up(self.conv_down(x + e) - y) - self.gama1[i] * (x + e - NL))

            x = x - self.delta[i] * (
                    self.conv_up(self.conv_down(x) - y + self.u[i] * self.conv_down(x + e) - y) + self.eta[i] * (
                    x - conv_out) + self.gama[i] * (x + e - NL))

            outs.append(x)

        return x
