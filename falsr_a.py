import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    def __init__(self, in_channels=32, activation=nn.ReLU6, bias=True):
        super(FeatureExtractor, self).__init__()
        self.act = activation(inplace=True)

        self.cell1_conv1 = nn.Conv2d(32, 64, 3, 1, 1, bias=bias)
        self.cell1_conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=bias)
        self.cell1_conv3 = nn.Conv2d(64, 64, 3, 1, 1, bias=bias)
        self.cell1_conv4 = nn.Conv2d(64, 64, 3, 1, 1, bias=bias)

        self.cell2_conv1 = nn.Conv2d(64, 48, 1, 1, 0, bias=bias)

        self.cell3_conv1 = nn.Conv2d(48, 64, 3, 1, 1, bias=bias)
        self.cell3_conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=bias)
        self.cell3_conv3 = nn.Conv2d(64, 64, 3, 1, 1, bias=bias)
        self.cell3_conv4 = nn.Conv2d(64, 64, 3, 1, 1, bias=bias)

        self.cell4_conv1 = nn.Conv2d(64+48, 64, 3, 1, 1, bias=bias)
        self.cell4_conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=bias)
        self.cell4_conv3 = nn.Conv2d(64, 64, 3, 1, 1, bias=bias)
        self.cell4_conv4 = nn.Conv2d(64, 64, 3, 1, 1, bias=bias)

        self.cell5_conv1 = nn.Conv2d(64+48+in_channels, 64, 3, 1, 1, bias=bias)
        self.cell5_conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=bias)
        self.cell5_conv3 = nn.Conv2d(64, 64, 3, 1, 1, bias=bias)
        self.cell5_conv4 = nn.Conv2d(64, 64, 3, 1, 1, bias=bias)

        self.cell6_conv1 = nn.Conv2d(64, 64, 1, 1, 0, bias=bias)
        self.cell6_conv2 = nn.Conv2d(64, 64, 1, 1, 0, bias=bias)
        self.cell6_conv3 = nn.Conv2d(64, 64, 1, 1, 0, bias=bias)
        self.cell6_conv4 = nn.Conv2d(64, 64, 1, 1, 0, bias=bias)

        self.cell7_conv1 = nn.Conv2d(64+64+48+in_channels, 64, 3, 1, 1, bias=bias)
        self.cell7_conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=bias)
        self.cell7_conv3 = nn.Conv2d(64, 64, 3, 1, 1, bias=bias)
        self.cell7_conv4 = nn.Conv2d(64, 64, 3, 1, 1, bias=bias)

    def forward(self, x):
        c0 = x

        # cell1
        x = self.act(self.cell1_conv1(x))
        t = x
        x = self.act(self.cell1_conv2(x))
        x = self.act(self.cell1_conv3(x))
        x = self.act(self.cell1_conv4(x))
        x += t
        c1 = x

        # cell2
        x = self.act(self.cell2_conv1(x))
        c2 = x

        # cell3
        x = self.act(self.cell3_conv1(x))
        t = x
        x = self.act(self.cell3_conv2(x))
        x = self.act(self.cell3_conv3(x))
        x = self.act(self.cell3_conv4(x))
        x += t
        c3 = x
        x = torch.cat([x, c2], dim=1)

        # cell4
        x = self.act(self.cell4_conv1(x))
        t = x
        x = self.act(self.cell4_conv2(x))
        x = self.act(self.cell4_conv3(x))
        x = self.act(self.cell4_conv4(x))
        x += t
        c4 = x
        x = torch.cat([x, c2, c0], dim=1)

        # cell5
        x = self.act(self.cell5_conv1(x))
        t = x
        x = self.act(self.cell5_conv2(x))
        x = self.act(self.cell5_conv3(x))
        x = self.act(self.cell5_conv4(x))
        x += t
        c5 = x

        # cell6
        x = self.act(self.cell6_conv1(x))
        x = self.act(self.cell6_conv2(x))
        x = self.act(self.cell6_conv3(x))
        x = self.act(self.cell6_conv4(x))
        c6 = x
        x = torch.cat([x, c5, c2, c0], dim=1)

        # cell7
        x = self.act(self.cell7_conv1(x))
        t = x
        x = self.act(self.cell7_conv2(x))
        x = self.act(self.cell7_conv3(x))
        x = self.act(self.cell7_conv4(x))
        x += t

        x = torch.cat([x, c0, c1, c2, c3, c4, c5, c6], dim=1)
        return x


class FALSRA(nn.Module):
    def __init__(self, activation=nn.ReLU6, bias=True):
        super(FALSRA, self).__init__()
        self.act = activation(inplace=True)

        self.input_conv = nn.Conv2d(1, 32, 3, 1, 1, bias=bias)
        self.feature_extractor = FeatureExtractor(in_channels=32, activation=activation, bias=bias)
        self.y_conv1 = nn.Conv2d(464, 32, 3, 1, 1, bias=bias)
        self.pixelshuffle = nn.PixelShuffle(2)
        self.y_conv2 = nn.Conv2d(8, 1, 3, 1, 1, bias=bias)

        self.rgb2ypbpr = torch.FloatTensor([0.299, -0.14713, 0.615, 0.587, -0.28886, -0.51499, 0.114, 0.436, -0.10001]).view(3, 3)
        self.ypbpr2rgb = torch.FloatTensor([1., 1., 1., 0., -0.39465, 2.03211, 1.13983, -0.58060, 0.]).view(3, 3)

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
    model = FALSRA().cuda()
    output = model(data)
    print(output.shape)

    #data = torch.ones(2, 3, 32, 32).cuda()
    pbpr = model.rgb2ypbpr_transform(data)
    rgb = model.ypbpr2rgb_transform(pbpr)
    print("diff:", (data - rgb).abs().sum() / data.size().numel())
    print("diff:", (data - rgb).abs().sum() / data.abs().sum())
    print("diff:", ((data - rgb).abs() / (data.abs() + 1e-8) ).sum() / data.size().numel() )
    print("diff:", ((data - rgb).abs() / (data.abs() + 1e-8) ).max())

if __name__ == "__main__":
    test()
