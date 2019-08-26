import torch
import torch.nn as nn
import torch.nn.functional as F


class InvertBottleneckConv(nn.Module):
    def __init__(self, in_c, out_c, e=2, kernel_size=3, activation=nn.ReLU6, bias=True):
        super(InvertBottleneckConv, self).__init__()
        self.act = activation(inplace=True)
        self.conv1 = nn.Conv2d(in_c, in_c*e, kernel_size, 1, kernel_size//2, bias=bias)
        self.conv2 = nn.Conv2d(in_c*e, in_c*e, 3, 1, 1, bias=bias, groups=out_c*e)
        self.conv3 = nn.Conv2d(in_c*e, out_c, kernel_size, 1, kernel_size//2, bias=bias)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        return x


class FeatureExtractor(nn.Module):
    def __init__(self, in_channels=32, f=1, activation=nn.ReLU6, bias=True):
        super(FeatureExtractor, self).__init__()
        self.act = activation(inplace=True)

        self.cell1_conv1 = nn.Conv2d(in_channels*f, 16*f, 1, 1, 0, bias=bias)
        self.cell1_conv2 = InvertBottleneckConv(16*f, 16*f, kernel_size=3, activation=activation, bias=bias)

        self.cell2_conv1 = nn.Conv2d((in_channels+16)*f, 48*f, 1, 1, 0, bias=bias)
        self.cell2_conv2 = InvertBottleneckConv(48*f, 48*f, kernel_size=1, activation=activation, bias=bias)
        self.cell2_conv3 = InvertBottleneckConv(48*f, 48*f, kernel_size=1, activation=activation, bias=bias)

        self.cell3_conv1 = nn.Conv2d((in_channels+16+48)*f, 16*f, 1, 1, 0, bias=bias)
        self.cell3_conv2 = nn.Conv2d(16*f, 16*f, 1, 1, 0, bias=bias)

        self.cell4_conv1 = nn.Conv2d(16*f, 32*f, 1, 1, 0, bias=bias)
        self.cell4_conv2 = InvertBottleneckConv(32*f, 32*f, kernel_size=3, activation=activation, bias=bias)
        self.cell4_conv3 = InvertBottleneckConv(32*f, 32*f, kernel_size=3, activation=activation, bias=bias)
        self.cell4_conv4 = InvertBottleneckConv(32*f, 32*f, kernel_size=3, activation=activation, bias=bias)
        self.cell4_conv5 = InvertBottleneckConv(32*f, 32*f, kernel_size=3, activation=activation, bias=bias)

        self.cell5_conv1 = nn.Conv2d((48+16+32)*f, 64*f, 3, 1, 1, bias=bias)
        self.cell5_conv2 = nn.Conv2d(64*f, 64*f, 3, 1, 1, bias=bias)

        self.cell6_conv1 = nn.Conv2d((16+32+64)*f, 16*f, 3, 1, 1, bias=bias, groups=4)
        self.cell6_conv2 = nn.Conv2d(16*f, 16*f, 3, 1, 1, bias=bias, groups=4)
        self.cell6_conv3 = nn.Conv2d(16*f, 16*f, 3, 1, 1, bias=bias, groups=4)
        self.cell6_conv4 = nn.Conv2d(16*f, 16*f, 3, 1, 1, bias=bias, groups=4)

        self.cell7_conv1 = nn.Conv2d((in_channels+48+16+16)*f, 16*f, 3, 1, 1, bias=bias)

    def forward(self, x):
        c0 = x

        # cell1
        x = self.cell1_conv1(x)
        t0 = x
        x = self.cell1_conv2(x)
        x += t0
        c1 = x
        x = torch.cat([x, c0], dim=1)

        # cell2
        x = self.cell2_conv1(x)
        t0 = x
        x = self.cell2_conv2(x)
        t1 = x+t0
        x = self.cell2_conv3(t1)
        x += t1
        x += t0
        c2 = x
        x = torch.cat([x, c1, c0], dim=1)

        # cell3
        x = self.act(self.cell3_conv1(x))
        t0 = x
        x = self.act(self.cell3_conv2(x))
        x += t0

        # cell4
        x = self.cell4_conv1(x)
        t0 = x
        x = self.cell4_conv2(x)
        t1 = x+t0
        x = self.cell4_conv3(t1)
        t1 += x
        x = self.cell4_conv4(t1)
        t1 += x
        x = self.cell4_conv5(t1)
        x += t1
        c4 = x
        x = torch.cat([x, c2, c1], dim=1)

        # cell5
        x = self.act(self.cell5_conv1(x))
        x = self.act(self.cell5_conv2(x))
        c5 = x
        x = torch.cat([x, c4, c1], dim=1)

        # cell6
        x = self.act(self.cell6_conv1(x))
        x = self.act(self.cell6_conv2(x))
        x = self.act(self.cell6_conv3(x))
        x = self.act(self.cell6_conv4(x))
        c6 = x
        x = torch.cat([x, c2, c1, c0], dim=1)

        # cell7
        x = self.act(self.cell7_conv1(x))

        x = torch.cat([x, c6, c4, c1], dim=1)
        return x


class FALSRB(nn.Module):
    def __init__(self, activation=nn.ReLU6, bias=True):
        super(FALSRB, self).__init__()
        self.act = activation(inplace=True)

        self.input_conv = nn.Conv2d(1, 32, 3, 1, 1, bias=bias)
        self.feature_extractor = FeatureExtractor(in_channels=32, activation=activation, bias=bias)
        self.y_conv1 = nn.Conv2d(16+16+16+32, 32, 3, 1, 1, bias=bias)
        self.pixelshuffle = nn.PixelShuffle(2)
        self.y_conv2 = nn.Conv2d(8, 1, 3, 1, 1, bias=bias)

        self.rgb2ypbpr = torch.FloatTensor([0.299, -0.147, 0.615, 0.587, -0.289, -0.515, 0.114, 0.436, 0.100]).view(3, 3)
        self.ypbpr2rgb = torch.FloatTensor([1., 1., 1., 0., -0.394, 2.032, 1.140, -0.581, 0.]).view(3, 3)

    def rgb2ypbpr_transform(self, x):
        rgb2ypbpr = self.rgb2ypbpr.to(x.device)
        return x.permute(0, 2, 3, 1).matmul(rgb2ypbpr).permute(0, 3, 1, 2)

    def ypbpr2rgb_transform(self, x):
        ypbpr2rgb = self.ypbpr2rgb.to(x.device)
        return x.permute(0, 2, 3, 1).matmul(ypbpr2rgb).permute(0, 3, 1, 2)

    def forward(self, x):
        # 0-1 input
        x_scale = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        pbpr = self.rgb2ypbpr_transform(x_scale)[:, 1:]

        y = self.rgb2ypbpr_transform(x)[:, 0:1]
        x = self.act(self.input_conv(y))
        f0 = x
        x = self.feature_extractor(x)
        x = self.act(self.y_conv1(x))
        x = self.pixelshuffle(x+f0)
        y = self.y_conv2(x)

        ypbpr = torch.cat([y, pbpr], dim=1)

        return self.ypbpr2rgb_transform(ypbpr)


def test():
    data = torch.randn(2, 3, 32, 32).cuda()
    model = FALSRB().cuda()
    output = model(data)
    print(output.shape)


if __name__ == "__main__":
    test()
