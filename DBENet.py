import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#  dense down block
class Ddown(nn.Module):
    def __init__(self, in_C):
        super(Ddown, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_C, in_C, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_C),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_C * 2, in_C, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_C),
            nn.ReLU(inplace=True)
        )

        self.conv1_1 = nn.Conv2d(in_C * 4, in_C * 2, kernel_size=1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x1 = self.conv1(x)  # 32 -> 32
        x2 = torch.cat([x1, x], 1)  # 32 -> 64
        x3 = self.conv2(x2)  # 64 -> 32
        x4 = torch.cat([x1, x2, x3], 1)  # 128
        x5 = self.pool(self.conv1_1(x4))  # 128 -> 64

        return x5

# dense up block
class Dup(nn.Module):
    def __init__(self, in_C):
        super(Dup, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_C, in_C, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_C),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_C * 2, in_C, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_C),
            nn.ReLU(inplace=True)
        )

        self.conv1_1 = nn.Conv2d(in_C * 4, in_C // 2, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)  # 32 -> 32
        x2 = torch.cat([x1, x], 1)  #
        x3 = self.conv2(x2)
        x4 = torch.cat([x1, x2, x3], 1)
        x5 = nn.UpsamplingBilinear2d(scale_factor=2)(self.conv1_1(x4))
        # x5 = F.interpolate(self.conv1_1(x4), scale_factor=2, mode='bilinear', align_corners=True)

        return x5


# channels shuffle
class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, int(C / g), H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)


# residual down block
class Sdown(nn.Module):
    def __init__(self, in_C):
        super(Sdown, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_C, in_C, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_C),
            nn.ReLU(inplace=True)
        )
        self.conv1_1 = nn.Conv2d(in_C, in_C * 2, 1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x1 = self.conv(x)
        return self.pool(self.conv1_1(x + x1))

# residual up block
class Sup(nn.Module):
    def __init__(self, in_C):
        super(Sup, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_C, in_C, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_C),
            nn.ReLU(inplace=True)
        )
        self.conv1_1 = nn.ConvTranspose2d(in_C, in_C // 2, 2, 2)

    def forward(self, x):
        x1 = self.conv(x)
        return self.conv1_1(x + x1)


#  cross-channel interaction component
class CCIC(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(CCIC, self).__init__()
        
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        
        padding = kernel_size // 2
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=kernel_size, padding=padding, bias=False
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg(x).view([b, 1, c])
        y = self.conv(y)
        y = self.sig(y).view([b, c, 1, 1])

        return y * x


#  channel excitation component
class CEC(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CEC, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, X_input):
        b, c, _, _ = X_input.size()  

        y = self.avg_pool(X_input)  
        y = y.view(b, c)  

        y = self.linear1(y) 
        y = self.linear2(y)  
        y = y.view(b, c, 1, 1)  

        return y * X_input


#  Ensemble Attention Module
class EAM(nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()

        self.cec = CEC(in_channels)
        self.ccic = CCIC(in_channels)

        self.conv = conv1_1(in_channels)
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        x1 = self.cec(x)
        x2 = self.ccic(x)

        x3 = x1 + x2
        x4 = self.conv(x * x3)

        return self.upsampling(x4)


class conv1_1(nn.Module):
    def __init__(self, in_C):
        super(conv1_1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_C, in_C // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_C // 2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DBENet(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super(DBENet, self).__init__()
        self.conv0 = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(inplace=True))

        self.D1 = Ddown(32)  # H×W×32         ->  H/2×W/2×64
        self.D2 = Ddown(64)  # H/2×W/2×64     ->  H/4×W/4×128
        self.D3 = Ddown(128)  # H/4×W/4×128    ->  H/8×W/8×256
        self.D4 = Ddown(256)  # H/8×W/8×256    ->  H/16×W/16×512

        self.U1 = Dup(512)  # H/16×W/16×512  ->  H/8×W/8×256
        self.U2 = Dup(256)  # H/8×W/8×256    ->  H/4×W/4×128
        self.U3 = Dup(128)  # H/4×W/4×128    ->  H/2×W/2×64
        self.U4 = Dup(64)  # H/2×W/2×64     ->  H×W×32

        self.Q1 = Sdown(32)  # H×W×32         ->  H/2×W/2×64
        self.Q2 = Sdown(64)  # H/2×W/2×64     ->  H/4×W/4×128
        self.Q3 = Sdown(128)  # H/4×W/4×128    ->  H/8×W/8×256
        self.Q4 = Sdown(256)  # H/8×W/8×256    ->  H/16×W/16×512

        self.O1 = Sup(512)  # H/16×W/16×512  ->  H/8×W/8×256
        self.O2 = Sup(256)  # H/8×W/8×256    ->  H/4×W/4×128
        self.O3 = Sup(128)  # H/4×W/4×128    ->  H/2×W/2×64
        self.O4 = Sup(64)  # H/2×W/2×64     ->  H×W×32

        self.att1 = EAM(512)
        self.att2 = EAM(256)
        self.att3 = EAM(128)
        self.att4 = EAM(64)


        self.c1 = conv1_1(512)
        self.c2 = conv1_1(256)
        self.c3 = conv1_1(128)
        self.c4 = conv1_1(64)

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv1_1 = nn.Conv2d(32, num_classes, 1)
        self.soft = nn.Softmax(dim=1)
        self.shuffle = ChannelShuffle(2)

    def forward(self, x):
        x = self.conv0(x)
        d1 = self.D1(x)  
        d2 = self.D2(d1)  
        d3 = self.D3(d2)  
        d4 = self.D4(d3)  
        d5 = nn.ReLU(inplace=True)(d4)

        q1 = self.Q1(x)  
        q2 = self.Q2(q1)  
        q3 = self.Q3(q2)  
        q4 = self.Q4(q3)  
        q5 = nn.ReLU(inplace=True)(q4)

        x1 = self.eca1(d4)
        y1 = self.se1(q4)
        xy1 = self.upsample(self.c1(x1 + y1))  
        u1 = self.U1(d5)
        o1 = self.O1(q5)
        u1_1 = xy1 + u1 + self.shuffle(d3)
        o1_1 = xy1 + o1

        x2 = self.eca2(u1_1)
        y2 = self.se2(o1_1)
        xy2 = self.upsample(self.c2(x2 + y2))  
        u2 = self.U2(u1_1)
        o2 = self.O2(o1_1)
        u2_1 = xy2 + u2 + self.shuffle(d2)
        o2_1 = xy2 + o2

        x3 = self.eca3(u2_1)
        y3 = self.se3(o2_1)
        xy3 = self.upsample(self.c3(x3 + y3))  
        u3 = self.U3(u2_1)
        o3 = self.O3(o2_1)
        u3_1 = xy3 + u3 + self.shuffle(d1)
        o3_1 = xy3 + o3

        u4 = self.U4(u3_1)
        o4 = self.O4(o3_1)

        out = u4 + o4

        out = self.soft(self.conv1_1(out))

        return out
