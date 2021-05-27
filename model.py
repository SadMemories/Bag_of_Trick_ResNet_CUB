import torch.nn as nn
from torch.nn import init
import torch
from torch.utils.tensorboard import SummaryWriter


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample != None:
            residual = self.downsample(x)

        out += residual
        out = self.relu2(out)
        return out


class BottleBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleBlock, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu3(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes):
        super(ResNet, self).__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        # self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv1_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(kernel_size=7)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # init.kaiming_normal(self.fc.weight)
        # for key in self.state_dict():
        #     if key.split('.')[-1] == 'weight':
        #         if 'conv' in key:
        #             init.kaiming_normal(self.state_dict()[key], mode='fan_out')
        #         if 'bn' in key:
        #             if "SpatialGate" in key:
        #                 self.state_dict()[key][...] = 0
        #             else:
        #                 self.state_dict()[key][...] = 1
        #     elif key.split('.')[-1] == 'bias':
        #         self.state_dict()[key][...] = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, planes, num_block, stride=1):

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # if stride == 2:
            #     downsample = nn.Sequential(
            #         nn.AvgPool2d(kernel_size=2, stride=2),
            #         nn.Conv2d(self.inplanes, planes * block.expansion,
            #                   kernel_size=1, stride=1, bias=False),
            #         nn.BatchNorm2d(planes * block.expansion)
            #     )
            # else:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = [block(self.inplanes, planes, stride, downsample)]

        self.inplanes = planes * block.expansion
        for i in range(1, num_block):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        # out = out.view(out.size(0), -1)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out


def get_resnet(depth, num_classes):
    assert depth in [18, 34, 50, 101], 'depth should be 18 or 34 or 50 or 101'

    if depth == 18:
        return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)
    elif depth == 34:
        return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)
    elif depth == 50:
        return ResNet(BottleBlock, [3, 4, 6, 3], num_classes)
    else:
        return ResNet(BottleBlock, [3, 4, 23, 3], num_classes)


# if __name__ == '__main__':
#     writer = SummaryWriter()
#     model = get_resnet(50, 200)
#     input = torch.zeros([32, 3, 224, 224])
#     writer.add_graph(model, input)
#     writer.close()