import copy
import math
import os
import random
import time
from collections import OrderedDict

import requests

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms
import torchsummary as summary

import matplotlib.pyplot as plt
from PIL import Image

import sys

from torch.multiprocessing import Lock
sys.path.append(r"/home/COCI/")
from src.COCI_Iterator import COCIIterator
# from ft.OCI_Checkpoint import OCICheckpoint

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


batch_size = 64
train_dataset = datasets.CIFAR10(root=r"../data/",
                                 train=True,
                                 transform=transforms.ToTensor(),
                                 download=True)

test_dataset = datasets.CIFAR10(root=r'../data/',
                                train=False,
                                transform=transforms.ToTensor())


train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, grayscale):
        self.inplanes = 64
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # because MNIST is already 1x1 here:
        # disable avg pooling
        # x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas


def resnet101(num_classes):
    """Constructs a ResNet-34 model."""
    model = ResNet(block=Bottleneck,
                   layers=[3, 4, 23, 3],
                   num_classes=num_classes,
                   grayscale=False)
    return model

NUM_EPOCHS = 100

model = resnet101(num_classes=10)

model = model.to(DEVICE)
# model.load_state_dict(torch.load(ckpath))
summary.summary(model, input_size=(3, 32, 32), device='cuda')

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

valid_loader = test_loader


def compute_accuracy_and_loss(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    cross_entropy = 0.
    for i, (features, targets) in enumerate(data_loader):
        features = features.to(device)
        targets = targets.to(device)

        logits, probas = model(features)
        cross_entropy += F.cross_entropy(logits, targets).item()
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float() / num_examples * 100, cross_entropy / num_examples


def poisson_error_list(ft_lambda):
    e_list = []
    e_slot = 0
    for i in range(10):
        # r = random.random()
        r = 0.04
        e_interval = - math.log(1 - r) / ft_lambda
        e_slot += e_interval
        e_list.insert(i, e_slot)
    return e_list
error_num = 0
error_list = poisson_error_list(ft_lambda=0.008)
# error_list[0] = 1.0
print('error_list:{}'.format(error_list))

start_time = time.time()
train_acc_lst, valid_acc_lst = [], []
train_loss_lst, valid_loss_lst = [], []

# ck, the unit is 'min'
# OCI = OCIIterator(model_name='resnet101',
#                   dataloader=train_loader,
#                   ft_lambda=0.0042,  # 4 hours a error
#                   ck_mode='MANUAL',
#                   ts=0.0015,  # the unit is min
#                   theta_1=-0.078,
#                   theta_2=0.787,
#                   epoch=NUM_EPOCHS,
#                   ft_strategy='OCI',
#                   fit_interval=10000,
#                   profile_threshold=50,
#                   model=model,
#                   optimizer=optimizer)
COCI = COCIIterator(model_name='resnet101',
                  dataloader=train_loader,
                  ft_lambda=0.0042,  # 4 hours a error
                  epoch=NUM_EPOCHS,
                  model=model,
                  optimizer=optimizer)
flag, start_epoch= COCI.recovery()
ck_t_error = []


# the model is loaded if there is a saved model
if flag:
    checkpoint = COCI.get_checkpoint()
    pre_train_t = checkpoint['ck_time']
    start_time = time.time() - pre_train_t
    print('loading epoch {} successfully！'.format(start_epoch))
else:
    start_epoch = 1
    start_time = time.time()
for epoch in range(start_epoch, NUM_EPOCHS):

    model.train()

    for batch_idx, (features, targets) in enumerate(train_loader):
        # fit error
        if error_num < len(error_list) - 1:
            if error_list[error_num] <= (time.time() - start_time) / 60 and (time.time() - start_time) / 60 < error_list[error_num + 1]:
                error_num += 1
                print('error{} is happen!!'.format(error_num))
                # recovery_loss = OCI.loss_list[-1]
                # print('loss where error happen is {}'.format(recovery_loss))
                COCI.recovery(gpu=1)
                # print('loss for taking ck is {}'.format(ck_loss - recovery_loss))
                # progress_cost_recovery_sum += ck_loss - recovery_loss
                ck_start_t = COCI.ck_start_t[-1]
                ck_t_error.append(ck_start_t)

        # if init_iteration > 0:
        #     init_iteration -= 1
        #     continue

        iter_start_t = time.time()
        ### PREPARE MINIBATCH
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)
        # print('from CPU to GPU takes {}'.format(time.time() - iter_start_t))

        ### FORWARD AND BACK PROP
        t = time.time()
        logits, probas = model(features)
        # print('forward takes {}'.format(time.time() - t))
        cost = F.cross_entropy(logits, targets)
        # print('forward takes {}'.format(time.time() - iter_start_t))

        optimizer.zero_grad()

        t = time.time()
        cost.backward()
        # print('backward takes {}'.format(time.time() - t))


        ### UPDATE MODEL PARAMETERS
        # optimizer.step()
        t = time.time()
        COCI.optimizer_step(cost, model, optimizer)
        # print('optimizer step takes {}s'.format(time.time() - t))

        ### LOGGING
        if not batch_idx % len(train_loader):

            # ck = OCICheckpoint(model=model, optimizer=optimizer)
            # meta_state={'epoch':epoch, 'iter':batch_idx}
            # ck_state = 'idle'
            # flag = ck._snapshot_GPU(lock=Lock(), meta_state=meta_state)
            # print("flag is {} and ck_state is {}s".format(flag, ck_state))
            # ckpath = r'/WdHeDisk/users/wangzc/COCI_pytorch/DNN/ResNet101/checkpoint/ck_ResNet101_0.chk'
            # ckname_dict = OrderedDict()
            # ck._persist(ckpath, snapshot=ck.latest_snapshot, lock=Lock(), ckname_dict=ckname_dict)

            print(f'Epoch: {epoch + 1:03d}/{NUM_EPOCHS:03d} | '
                  f'Batch {batch_idx:04d}/{len(train_loader):04d} |'
                  f' Cost: {cost:.4f}')
            # print('model size is {}'.format(sys.getsizeof(optimizer.state_dict())))

    elapsed = (time.time() - start_time) / 60
    print(f'Total Training Time: {elapsed:.2f} min')


# no need to build the computation graph for backprop when computing accuracy
model.eval()
with torch.set_grad_enabled(False):
    train_acc, train_loss = compute_accuracy_and_loss(model, train_loader, device=DEVICE)
    valid_acc, valid_loss = compute_accuracy_and_loss(model, valid_loader, device=DEVICE)
    train_acc_lst.append(train_acc)
    valid_acc_lst.append(valid_acc)
    train_loss_lst.append(train_loss)
    valid_loss_lst.append(valid_loss)
    print(f'Epoch: {epoch + 1:03d}/{NUM_EPOCHS:03d} Train Acc.: {train_acc:.2f}%'
          f' | Validation Acc.: {valid_acc:.2f}%')
elapsed = (time.time() - start_time) / 60
print(f'Total Training Time: {elapsed:.2f} min')
# torch.save()
ts_list = COCI.ts_list
sum = 0
for ts in ts_list:
    sum += ts
print('ck cost is {}min'.format(sum))


def loss_function(theta_1, theta_2, t):
    return math.exp(theta_1 * t + theta_2)
