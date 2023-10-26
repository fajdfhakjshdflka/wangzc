import json
import os

import torch
from torch import optim
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import time
import math

from torch.nn.parallel import DistributedDataParallel as ddp
from torch.utils.data.distributed import DistributedSampler as ds
import torch.distributed as distri

import sys
sys.path.append(r"/home/COCI/")
from src.COCI_Iterator import COCIIterator

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
EPOCH = 1
batch_size = 16
classes_num = 1000
learning_rate = 1e-3

def setup_DDP(backend="nccl", verbose=False):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # If the OS is Windows or macOS, use gloo instead of nccl
    distri.init_process_group(backend=backend)
    # set distributed device
    DEVICE = torch.device("cuda:{}".format(local_rank))
    if verbose:
        print(f"local rank: {local_rank}, global rank: {rank}, world size: {world_size}")
    return rank, local_rank, world_size, DEVICE
print("Setup_DDP step")
rank, local_rank, world_size, DEVICE = setup_DDP(verbose=True)
print("\n my rank is",rank)
this_rank = distri.get_rank()

print('device=', DEVICE)
#DEVICE = torch.device("cuda:0")
'''Define Transform'''
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

train_dir = r"/home/COCI/data/ILSVRC2012/train"

train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
train_sampler = ds(train_datasets, shuffle=True)

#train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=8,
#                                               pin_memory=True)  # ,num_workers=16,pin_memory=False
train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=False, drop_last=True, sampler=train_sampler)

val_dir = "/home/COCI/data/ILSVRC2012/val"
val_datasets = datasets.ImageFolder(val_dir, transform=val_transforms)
val_sampler = ds(val_datasets, shuffle=False)
#val_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size=batch_size, shuffle=True, num_workers=8,
#                                             pin_memory=True)  # ,num_workers=16,pin_memory=True
val_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size=batch_size, shuffle=False, drop_last=True, sampler=val_sampler)

models_dir = os.path.expanduser('~/.torch/models')
model_name = {
    'vgg11': 'vgg11-bbd30ac9.pth',
    'vgg11_bn': 'vgg11_bn-6002323d.pth',
    'vgg13': 'vgg13-c768596a.pth',
    'vgg13_bn': 'vgg13_bn-abd245e5.pth',
    'vgg16': 'vgg16-397923af.pth',
    'vgg16_bn': 'vgg16_bn-6c64b313.pth',
    'vgg19': 'vgg19-dcbb9e9d.pth',
    'vgg19_bn': 'vgg19_bn-c79401a0.pth',
}

class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


def vgg11(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A']), **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['vgg11'])))
    return model


def vgg11_bn(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['vgg11_bn'])))
    return model


def vgg13(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B']), **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['vgg13'])))
    return model


def vgg13_bn(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['vgg13_bn'])))
    return model


def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['vgg16'])))
    return model


def vgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['vgg16_bn'])))
    return model


def vgg19(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E']), **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['vgg19'])))
    return model


def vgg19_bn(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['vgg19_bn'])))
    return model


def eval(model, loss_func, dataloader):

    model.eval()
    loss, accuracy = 0, 0

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)

            logits = model(batch_x)
            error = loss_func(logits, batch_y)
            loss += error.item()

            probs, pred_y = logits.data.max(dim=1)
            accuracy += (pred_y==batch_y.data).float().sum()/batch_y.size(0)

    loss /= len(dataloader)
    accuracy = accuracy*100.0/len(dataloader)
    return loss, accuracy


# def poisson_error_list(ft_lambda):
#     e_list = []
#     e_slot = 0
#     for i in range(10):
#         r = random.random()
#         e_interval = - math.log(1 - r) / ft_lambda
#         e_slot += e_interval
#         e_list.insert(i, e_slot)
#     return e_list
# error_num = 0
# error_list = poisson_error_list(ft_lambda=0.008)
# error_list[0] = 1.0
# print('error_list:{}'.format(error_list))
ck_t_error = []

model = vgg19_bn()
# if torch.cuda.device_count() > 1:
#   print("Let's use", torch.cuda.device_count(), "GPUs!")
#   # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
#   model = nn.DataParallel(model)
model = model.to(DEVICE)
# model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
model = ddp(model, device_ids=[local_rank], output_device=local_rank)

model = model.module
model_Name = r'vgg19_bn'
distributed = r'_distributed'
training_result_dir = r'./training_result'
# model = model.to(DEVICE)
# summary.summary(model, input_size=(3, 224, 224), device='cuda')  # image input size(3,224,224)

params = [{'params': md.parameters()} for md in model.children()
          if md in [model.classifier]]
# optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
StepLR = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
# loss_func = nn.CrossEntropyLoss()
loss_func = nn.CrossEntropyLoss().to(DEVICE)

# ck, the unit is 'min'
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

group = distri.new_group([0, 1, 2, 3])
def train_res(model, train_dataloader, since_t, epoch, train_time_list, train_loss_list):
    model.train()
    # print('epoch {}'.format(epoch + 1))
    # training-----------------------------
    train_loss = 0.
    train_acc = 0.
    print(f'len(train_dataloader):{len(train_dataloader)} and len(train_datasets):{len(train_datasets)}')
    print()
    for batch_idx, (batch_x, batch_y) in enumerate(train_dataloader):
        if rank == 0 and batch_idx % 100 == 0:
            print('-------------------------------------------------------')
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

        # AllReduce
        # for p in model.parameters():
        #     p.grad = p.grad / world_size
        #     distri.all_reduce(p.grad, op=distri.ReduceOp.SUM, group=group, async_op=False)

        # optimizer.step()
        COCI.optimizer_step(loss, model, optimizer)

        # torch.distributed.barrier()

        current_t = (time.time() - since_t) / 60
        train_time_list.append(current_t)
        train_loss_list.append(loss.item())
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch:03d}/{EPOCH:03d} | '
                  f'Batch {batch_idx:04d}/{len(train_dataloader):04d} | '
                  f'Cost: {loss:.4f} | '
                  f'Time: {current_t:.4f}min | '
                  f'Rank: {rank:02d}')

        # if batch_idx != 0 and batch_idx % 1000 == 0:
        # print(f'batch_idx iter takes {(time.time() - since_t) / 60}min')

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

    for epoch in range(start_epoch, EPOCH + 1):
        since = time.time()
        # print('epoch {}'.format(epoch))
        train_time_list, train_loss_list = train_res(model, train_dataloader, start_time, epoch, train_time_list,
                                                     train_loss_list)
        time_elapsed = time.time() - since
        print('An epoch takes in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        if rank == 0:
            js = json.dumps(train_time_list)  # the unit is min
            f = open(os.path.join(training_result_dir, model_Name + '_ddp_time_4A30_0.txt'), 'w+')
            f.write(js)
            f.close()

            js = json.dumps(train_loss_list)
            f = open(os.path.join(training_result_dir, model_Name + '_ddp_loss_4A30_0.txt'), 'w+')
            f.write(js)
            f.close()


if __name__ == '__main__':
    main()
