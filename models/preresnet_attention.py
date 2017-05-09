'''Preactivated version of ResNet18/34/50/101/152 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                nn.ReLU(),
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += self.shortcut(x)
        return out

class Hourglass(nn.Module):
    def __init__(self, block, num_blocks, planes, depth):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.residual = self._make_layer(block, num_blocks, planes)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)

    def _make_layer(self, block, num_blocks, planes):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(planes*block.expansion, planes))
        return nn.Sequential(*layers)

    def _hour_glass(self, n, x):
        up1 = self.residual(x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.residual(low1)

        if n > 1:
            low2 = self._hour_glass(n-1, low1)
        else:
            low2 = self.residual(low1)
        low3 = self.residual(low2)
        up2 = self.upsample(low3)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hour_glass(self.depth, x)

# def test_hourglass():
#     net = Hourglass(Bottleneck, 1, 64, 4)
#     y = net(Variable(torch.randn(1,256,64,64)))
#     print(y.size())


class Attention(nn.Module):
    # implementation of Wang et al. "Residual Attention Network for Image Classification". CVPR, 2017.
    def __init__(self, block, p, t, r, planes, depth):
        super(Attention, self).__init__()
        self.p = p
        self.t = t
        out_planes = planes*block.expansion
        self.residual = block(out_planes, planes)
        self.hourglass = Hourglass(block, r, planes, depth)
        self.fc1 = nn.Conv2d(out_planes, out_planes, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(out_planes, 1, kernel_size=1, bias=False)

    def get_mask(self):
        return self.mx

    def forward(self, x):
        # preprocessing
        for i in range(0, self.p):
            x = self.residual(x)

        # trunk branch
        tx = x
        for i in range(0, self.p):
            tx = self.residual(tx)

        # mask branch
        self.mx = F.sigmoid(self.fc2(self.fc1(self.hourglass(x))))

        # residual attented feature
        out = tx + tx*self.mx.expand_as(tx)

        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(3,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.att1 = Attention(block, 1, 2, 1, 64, 4)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.att2 = Attention(block, 1, 2, 1, 128, 3)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.att3 = Attention(block, 1, 2, 1, 256, 2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def get_mask(self): # get attention mask
        masks = []
        masks.append(self.att1.get_mask())
        masks.append(self.att2.get_mask())
        masks.append(self.att3.get_mask())
        return masks

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.att1(out)
        out = self.layer2(out)
        out = self.att2(out)
        out = self.layer3(out)
        out = self.att3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def PreResNetAtt18():
    return ResNet(BasicBlock, [2,2,2,2])

def PreResNetAtt34():
    return ResNet(BasicBlock, [3,4,6,3])

def PreResNetAtt50():
    return ResNet(Bottleneck, [3,4,6,3])

def PreResNetAtt101():
    return ResNet(Bottleneck, [3,4,23,3])

def PreResNetAtt152():
    return ResNet(Bottleneck, [3,8,36,3])


# net = PreResNetAtt18()
# y = net(Variable(torch.randn(1,3,32,32)))
# print(net.get_mask())