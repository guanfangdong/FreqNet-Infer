import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.fft import irfftn, rfftn

import freq_reg_plus_v3 as frp
from freq_reg_plus_v3._dct import idct, dct
from freq_reg_plus_v3._dctbase import dct2Base, idct2Base

gnl_minrate = 0.007

class VGG(nn.Module):

    def __init__(self, cfg, batch_norm=False, num_classes=10):
        super(VGG, self).__init__()
        self.block0 = self._make_layers(cfg[0], batch_norm, 3)
        self.block1 = self._make_layers(cfg[1], batch_norm, cfg[0][-1])
        self.block2 = self._make_layers(cfg[2], batch_norm, cfg[1][-1])
        self.block3 = self._make_layers(cfg[3], batch_norm, cfg[2][-1])
        self.block4 = self._make_layers(cfg[4], batch_norm, cfg[3][-1])

        self.pool4 = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = frp.Linear(512, num_classes, minrate=1.0)
        
        self.reLU_1 = frp.ReLU()
        self.reLU_2 = frp.ReLU()
        self.reLU_3 = frp.ReLU()
        self.reLU_4 = frp.ReLU()
        self.reLU_5 = frp.ReLU()

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.block0)
        feat_m.append(self.pool0)
        feat_m.append(self.block1)
        feat_m.append(self.pool1)
        feat_m.append(self.block2)
        feat_m.append(self.pool2)
        feat_m.append(self.block3)
        feat_m.append(self.pool3)
        feat_m.append(self.block4)
        feat_m.append(self.pool4)
        return feat_m

    def get_bn_before_relu(self):
        bn1 = self.block1[-1]
        bn2 = self.block2[-1]
        bn3 = self.block3[-1]
        bn4 = self.block4[-1]
        return [bn1, bn2, bn3, bn4]

    def forward(self, x, return_features=False):
        h = x.shape[2]
        x = self.reLU_1(self.block0(x))
        x = frp.avg_pool2d(x, 2, 2)
        x = self.block1(x)
        x = self.reLU_2(x)
        x = frp.avg_pool2d(x, 2, 2)
        x = self.block2(x)
        x = self.reLU_3(x)
        x = frp.avg_pool2d(x, 2, 2)
        x = self.block3(x)
        x = self.reLU_4(x)
        if h == 64:
            x = frp.avg_pool2d(x, 2, 2)
        x = self.block4(x)
        x = self.reLU_5(x)
        x = self.pool4(x)
        features = x.view(x.size(0), -1)
        x = self.classifier(features)
        if return_features:
            return x, features
        else:
            return x

    @staticmethod
    def _make_layers(cfg, batch_norm=False, in_channels=3):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = frp.Conv2d(in_channels, v, kernel_size=3, padding=1, minrate=gnl_minrate, bias=False)
                if batch_norm:
                    layers += [conv2d, frp.BatchNorm2d(v), frp.ReLU()]
                else:
                    layers += [conv2d, frp.ReLU()]
                in_channels = v
        layers = layers[:-1]
        return nn.Sequential(*layers)



cfg = {'S': [[64], [128], [256], [512], [512]]}

def vgg8_bn(**kwargs):
    """VGG 8-layer model (configuration "S")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(cfg['S'], batch_norm=True, **kwargs)
    return model


def load_datasets():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=16, prefetch_factor=8, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=16, prefetch_factor=8, pin_memory=True)

    return trainloader, testloader

    
def test_epoch(net, device, testloader, criterion):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad(): 
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    avg_loss = test_loss / len(testloader)
    accuracy = 100. * correct / total

    print(f'\nTest set: Average loss: {avg_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.3f}%)')
    return accuracy

    

def main(argc, argv):        
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    global gnl_minrate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainloader, testloader = load_datasets()
    
    print('==> Building model..')
    net = vgg8_bn().to(device)
    # load pretrained model
    net.load_state_dict(torch.load(f'./model/vgg8_fr_plus_custom_BN_AP0.007.pth'))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    acc = test_epoch(net, device, testloader, criterion)
    print(f'Initial accuracy: {acc:.3f}')
    



if __name__ == '__main__':
    argc = len(sys.argv)
    argv = sys.argv
    main(argc, argv)
