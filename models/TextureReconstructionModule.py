import torch.nn as nn


class ConvUp(nn.Module):

    def __init__(self, ch_in, up_factor):

        super(ConvUp, self).__init__()

        body = [
            nn.Conv2d(in_channels=ch_in, out_channels=64, kernel_size=3, padding=3 // 2), nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2), nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2), nn.ReLU(),
        ]
        self.body = nn.Sequential(*body)

        if up_factor == 2:
            modules_tail = [
                nn.ConvTranspose2d(64, 64, kernel_size=3, stride=up_factor, padding=1, output_padding=1),
                nn.Conv2d(64, ch_in, 3, padding=3 // 2, bias=True)
            ]
        elif up_factor == 3:
            modules_tail = [
                nn.ConvTranspose2d(64, 64, kernel_size=3, stride=up_factor, padding=0, output_padding=0),
                nn.Conv2d(64, ch_in, 3, padding=3 // 2, bias=True)
            ]

        elif up_factor == 4:
            modules_tail = [
                nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.Conv2d(64, ch_in, 3, padding=3 // 2, bias=True)
            ]

        self.tail = nn.Sequential(*modules_tail)

    def forward(self, input):

        output = self.body(input)
        output = self.tail(output)
        return output


class ConvDown(nn.Module):

    def __init__(self, c_in, up_factor):

        super(ConvDown, self).__init__()

        body = [
            nn.Conv2d(in_channels=c_in, out_channels=64, kernel_size=3, padding=3 // 2), nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2), nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2), nn.ReLU(),
        ]
        self.body = nn.Sequential(*body)
        conv = common.default_conv
        if up_factor == 4:
            modules_tail = [
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=2),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=2),
                conv(64, c_in, 3)]
        elif up_factor == 3:
            modules_tail = [
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=up_factor),
                conv(64, c_in, 3)]
        elif up_factor == 2:
            modules_tail = [
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=up_factor),
                conv(64, c_in, 3)]
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, input):

        out = self.body(input)
        out = self.tail(out)
        return out
