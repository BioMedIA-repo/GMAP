import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_channels, filters, dropout_rate):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.filters = filters
        self.dropout_rate = dropout_rate
        factor = 2
        self.conv1 = UnetConv2D(self.in_channels, self.filters[0], self.dropout_rate[0])
        self.conv2 = Down(self.filters[0], self.filters[1], self.dropout_rate[1])
        self.conv3 = Down(self.filters[1], self.filters[2], self.dropout_rate[2])
        self.conv4 = Down(self.filters[2], self.filters[3], self.dropout_rate[3])
        self.center = Down(self.filters[3], self.filters[4] // factor, self.dropout_rate[4])

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        center = self.center(conv4)
        return conv1, conv2, conv3, conv4, center


class Decoder(nn.Module):
    def __init__(self, out_channels, filters, dropout_rate):
        super(Decoder, self).__init__()
        self.out_channels = out_channels
        self.filters = filters
        self.dropout_rate = dropout_rate

        factor = 2

        self.up_concat4 = Up(self.filters[4], self.filters[3] // factor, self.dropout_rate[0])
        self.up_concat3 = Up(self.filters[3], self.filters[2] // factor, self.dropout_rate[1])
        self.up_concat2 = Up(self.filters[2], self.filters[1] // factor, self.dropout_rate[2])
        self.up_concat1 = Up(self.filters[1], self.filters[0], self.dropout_rate[3])

        self.final = OutConv(self.filters[0], self.out_channels)

    def forward(self, conv1, conv2, conv3, conv4, center):
        dec4 = self.up_concat4(center, conv4)
        dec3 = self.up_concat3(dec4, conv3)
        dec2 = self.up_concat2(dec3, conv2)
        dec1 = self.up_concat1(dec2, conv1)
        final = self.final(dec1)
        soft_final = F.softmax(final, dim=1)
        return soft_final


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, encoder_dropout_rate=None, decoder_dropout_rate=None,
                 use_projection=True):
        super(UNet, self).__init__()
        if encoder_dropout_rate is None:
            encoder_dropout_rate = [0, 0, 0, 0, 0]
        if decoder_dropout_rate is None:
            decoder_dropout_rate = [0, 0, 0, 0]
        self.filters = [16, 32, 64, 128, 256]
        self.encoder_dropout_rate = encoder_dropout_rate
        self.decoder_dropout_rate = decoder_dropout_rate

        self.use_projection = use_projection
        if self.use_projection:
            proj_out_dim = 128
            self.projection = ProjectionLayer(output_dim=proj_out_dim)  # 只指定输出维度
        self.encoder = Encoder(in_channels, self.filters, self.encoder_dropout_rate)
        self.decoder = Decoder(out_channels, self.filters, self.decoder_dropout_rate)

    def forward(self, inputs):
        conv1, conv2, conv3, conv4, center = self.encoder(inputs)
        soft_final = self.decoder(conv1, conv2, conv3, conv4, center)
        # return conv1, conv2, conv3, conv4, center
        return soft_final


class ProjectionLayer(nn.Module):
    def __init__(self, output_dim=128):
        super(ProjectionLayer, self).__init__()
        self.output_dim = output_dim
        self.mid_dim = 512
        self.linear = nn.Linear(32768, self.mid_dim)
        self.proj = nn.Sequential(
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.mid_dim, output_dim)
        )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1).cuda() 
        x = self.linear(x)
        x = self.proj(x)
        x = F.normalize(x, p=2, dim=1)
        return x


class UnetConv2D(nn.Module):
    def __init__(self, in_size, out_size, dropout_rate=0.):
        super(UnetConv2D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.LeakyReLU(inplace=True),
        )
        self.dropout = nn.Dropout(p=dropout_rate)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.dropout(outputs)
        outputs = self.conv2(outputs)
        return outputs


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            UnetConv2D(in_channels, out_channels, dropout_rate=dropout_rate)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = UnetConv2D(in_channels, out_channels, dropout_rate=dropout_rate)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

