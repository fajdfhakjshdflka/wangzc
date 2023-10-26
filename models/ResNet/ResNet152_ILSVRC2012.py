from __future__ import print_function, division

import json

import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torchvision import models
from torch.autograd import Variable
import torchsummary as summary
import os
import csv
import codecs
import numpy as np
import time
from thop import profile

sys.path.append(r"/home/COCI/")
from src.COCI_Iterator import COCIIterator

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
EPOCH = 10
batch_size = 64
classes_num = 1000
learning_rate = 1e-3

DEVICE = torch.device("cuda:0")

'''定义Transform'''
train_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

val_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

train_dir = r"/home/OCI/DNN/data/ILSVRC2012/train"
train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=8,
                                               pin_memory=True)  # ,num_workers=16,pin_memory=False

val_dir = "/home/OCI/DNN/data/ILSVRC2012/val"
val_datasets = datasets.ImageFolder(val_dir, transform=val_transforms)
val_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size=batch_size, shuffle=True, num_workers=8,
                                             pin_memory=True)  # ,num_workers=16,pin_memory=True


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = torch.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=classes_num):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.classifier = nn.Linear(512 * block.expansion, num_classes)  #

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out_x1 = self.relu(out)
        out_x2 = self.maxpool(out_x1)
        out1 = self.layer1(out_x2)  # 56*56      4
        out2 = self.layer2(out1)  # 28*28        4
        out3 = self.layer3(out2)  # 14*14        4
        out4 = self.layer4(out3)  # (512*7*7)     4
        # out5 = F.avg_pool2d(out4, 4)
        out5 = self.avgpool(out4)
        out6 = out5.view(out5.size(0), -1)
        out7 = self.classifier(out6)
        return out7


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


# --------------------training process---------------------------------
model = ResNet152()
model_name = r'ResNet152'
training_result_dir = r'./training_result'
model = model.to(DEVICE)
# summary.summary(model, input_size=(3, 224, 224), device='cuda')

params = [{'params': md.parameters()} for md in model.children()
          if md in [model.classifier]]
# optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
StepLR = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
# loss_func = nn.CrossEntropyLoss()
loss_func = F.cross_entropy

# OCI = OCIIterator(model_name='resnet101',
#                   dataloader=train_dataloader,
#                   ft_lambda=0.0042,  # 4 hours a error
#                   ck_mode='MANUAL',
#                   ts=0.0015,  # the unit is min
#                   theta_1=-0.078,
#                   theta_2=0.787,
#                   epoch=EPOCH,
#                   ft_strategy='CCM',
#                   fit_interval=10000,
#                   profile_threshold=50,
#                   model=model,
#                   optimizer=optimizer)
COCI = COCIIterator(model_name='resnet101',
                  dataloader=train_dataloader,
                  ft_lambda=0.0042,  # 4 hours a error
                  epoch=EPOCH,
                  model=model,
                  optimizer=optimizer)
flag, start_epoch= COCI.recovery()
# the model is loaded if there is a saved model
if flag:
    checkpoint = COCI.get_checkpoint()
    pre_train_t = checkpoint['ck_time']
    train_time_list = checkpoint['iter_list']
    train_loss_list = checkpoint['loss_list']
    start_time = time.time() - pre_train_t
    print('loading epoch {} successfully！'.format(start_epoch))
else:
    start_epoch = 1
    start_time = time.time()
    train_time_list = []
    train_loss_list = []

def train_res(model, train_dataloader, since_t, epoch, train_time_list, train_loss_list):
    model.train()
    # print('epoch {}'.format(epoch + 1))
    # training-----------------------------
    train_loss = 0.
    train_acc = 0.
    print(f'len(train_dataloader):{len(train_dataloader)} and len(train_datasets):{len(train_datasets)}')
    print()
    for batch_idx, (batch_x, batch_y) in enumerate(train_dataloader):
        # iter_since = time.time()
        batch_x = Variable(batch_x).to(DEVICE)
        batch_y = Variable(batch_y).to(DEVICE)
        optimizer.zero_grad()
        out = model(batch_x)
        loss = loss_func(out, batch_y)
        # print(f'loss is {loss.item()}')
        train_loss += loss.item()
        pred = torch.max(out, 1)[1]
        train_correct = (pred == batch_y).sum()
        train_acc += train_correct.item()
        loss.backward()
        optimizer.step()
        # OCI.optimizer_step(loss, model, optimizer)
        # iter_end = time.time()
        # print(f'{batch_idx + 1:05d} iter takes {iter_end - iter_since:.4f}s')

        current_t = (time.time() - since_t) / 60
        train_time_list.append(current_t)
        train_loss_list.append(loss.item())
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch + 1:03d}/{EPOCH:03d} | '
                  f'Batch {batch_idx:04d}/{len(train_dataloader):04d} | '
                  f'Cost: {loss:.4f} | '
                  f'Time: {current_t:.4f}min')

        if batch_idx != 0 and batch_idx % 1000 == 0:
            print(f'batch_idx iter takes {(time.time() - since_t) / 60}min')

    return train_time_list, train_loss_list
    # print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(train_datasets)),
    #                                                train_acc / (len(train_datasets))))
    # train_Loss_list.append(train_loss / (len(val_datasets)))
    # train_Accuracy_list.append(100 * train_acc / (len(val_datasets)))


# evaluation--------------------------------
def val(model, val_dataloader):
    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    for batch_idx, (batch_x, batch_y) in enumerate(val_dataloader):
        batch_x = Variable(batch_x, volatile=True).to(DEVICE)
        batch_y = Variable(batch_y, volatile=True).to(DEVICE)
        out = model(batch_x)
        loss = loss_func(out, batch_y)
        eval_loss += loss.item()
        pred = torch.max(out, 1)[1]
        num_correct = (pred == batch_y).sum()
        eval_acc += num_correct.item()
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(val_datasets)),
                                                  eval_acc / (len(val_datasets))))
    # Loss_list.append(eval_loss / (len(val_datasets)))
    # Accuracy_list.append(100 * eval_acc / (len(val_datasets)))


def main():
    model.to(DEVICE)

    for epoch in range(start_epoch, EPOCH):
        since = time.time()
        # print('epoch {}'.format(epoch))
        train_time_list, train_loss_list = train_res(model, train_dataloader, start_time, epoch, train_time_list, train_loss_list)
        time_elapsed = time.time() - since
        print('An epoch takes in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        js = json.dumps(train_time_list)  # the unit is min
        f = open(os.path.join(training_result_dir, model_name + '_time.txt'), 'w+')
        f.write(js)
        f.close()

        js = json.dumps(train_loss_list)
        f = open(os.path.join(training_result_dir, model_name + '_loss.txt'), 'w+')
        f.write(js)
        f.close()

        state = {'model': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'epoch': epoch,
                 'pre_train_t': time.time() - start_time,
                 'train_time_list': train_time_list,
                 'train_loss_list': train_loss_list}
        torch.save(state, log_dir)


    js = json.dumps(train_time_list)  # the unit is min
    f = open(os.path.join(training_result_dir, model_name + '_time.txt'), 'w+')
    f.write(js)
    f.close()

    js = json.dumps(train_loss_list)
    f = open(os.path.join(training_result_dir, model_name + '_loss.txt'), 'w+')
    f.write(js)
    f.close()


if __name__ == '__main__':
    main()

