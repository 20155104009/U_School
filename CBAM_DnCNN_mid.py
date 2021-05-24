import torch
import torch.nn as nn
import torch.nn.functional as F
#import PixelUnShuffle



def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        #max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        #out = avg_out + max_out
        return self.sigmoid(avg_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes=64, planes=64, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.ca = ChannelAttention(32)
        self.sa = SpatialAttention()

        #self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out1 = self.ca(out[:,:32,:,:]) * out[:,:32,:,:]
        out2 = self.sa(out[:,32:,:,:]) * out[:,32:,:,:]
        out = torch.cat((out1,out2),1)

        out = self.conv2(out)
        out = self.bn2(out)





        out += residual
        out = self.relu(out)

        return out

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel=64, stride=1):
        super(ResidualBlock, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=stride, padding=1, bias=False,dilation=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=stride, padding=2, bias=False,dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=stride, padding=5, bias=False,dilation=5),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=stride, padding=2, bias=False,dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

        )
        self.block5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=stride, padding=1, bias=False,dilation=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),


        )





    def forward(self, x):
        y = x

        x = self.block1(x)

        x1_1 = self.block2(x)
        x1_2 = self.block2(x)

        x2_1 = self.block3(x1_1)
        x2_2 = self.block3(x1_1)
        x2_3 = self.block3(x1_2)
        x2_4 = self.block3(x1_2)


        x3_1 = self.block4(torch.add(x2_1,x2_2))
        x3_2 = self.block4(torch.add(x2_3,x2_4))
        x4_1 = self.block5(torch.add(x3_1,x3_2))



        return y+x4_1


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.inchannel = 64
        self.inchannel_1 = 64

        self.layer1_1 = BasicBlock(64, 64, stride=1)
        self.layer1_2 = BasicBlock(64, 64, stride=1)
        self.layer1_3 = BasicBlock(64, 64, stride=1)
        self.layer1_4 = BasicBlock(64, 64, stride=1)
        self.layer1_5 = BasicBlock(64, 64, stride=1)
        self.layer1_6 = BasicBlock(64, 64, stride=1)
        self.layer1_7 = BasicBlock(64, 64, stride=1)

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
        )

        self.conv5 = nn.Sequential(

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False,dilation=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False),

        )

        self.gate = nn.Conv2d(64 * 3, 3, 3, 1, 1, bias=True)


    def forward(self, x):   #batchSize * c * k*w * k*h   128*1*40*40
        y = x
        out = self.conv1(x)
        y1 = self.layer1_1(out)
        y2 = self.layer1_2(y1)
        y3 = self.layer1_3(y2)

        y4 = self.layer1_4(y3)
        y5 = self.layer1_5(y4)
        y6 = self.layer1_6(y5)
        y7 = self.layer1_7(y6)

        #gates = self.gate(torch.cat((y1, y2, y3), dim=1))
        #gated_y = y1 * gates[:, [0], :, :] + y2 * gates[:, [1], :, :] + y3 * gates[:, [2], :, :]

        out = self.conv5(y7)
        return out


import torch, torchvision
model=Net()
from torchsummary import summary
summary(model, (1,40, 40))