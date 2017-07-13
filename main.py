'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from tqdm import tqdm

import os
import argparse

from models import *
from utils import progress_bar
from torch.autograd import Variable


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--sort', action='store_true', help='sort the dataset')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
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


def flatten(x):
    return x.view(x.numel(), -1)


class SortingNetwork(nn.Module):
    def __init__(self):
        super(SortingNetwork, self).__init__()
        self.vgg = models.vgg16(pretrained=True)
        self.vgg.features = nn.Sequential(
            *(self.vgg.features[i] for i in range(29)))

    def forward(self, image):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        normalized_image = normalize(image.data)

        feature_representation = self.vgg.features(Variable(normalized_image))

        return flatten(feature_representation)


    def __str__(self):
        return str(self.vgg.features)

sorting_network = SortingNetwork()

if use_cuda:
    sorting_network.cuda()


trainset = torchvision.datasets.CIFAR10(root='./data',
                                        train=True,
                                        download=True,
                                        transform=transform_train)


def sorting_network_prediction(tensor):
    variable_tensor = Variable(tensor)

    if use_cuda:
        variable_tensor = variable_tensor.cuda()

    tensor_prediction = sorting_network(variable_tensor)

    return tensor_prediction.data


def find_average_tensor(dataset):
    print('Finding average tensor of dataset...')
    temp_dataloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=1,
                                                  num_workers=1)
    sum_tensor = torch.FloatTensor(torch.zeros(1, 512, 2, 2))

    if use_cuda:
        sum_tensor = sum_tensor.cuda()

    dataset_size = len(temp_dataloader)

    for batch_idx, (inputs, targets) in enumerate(tqdm(temp_dataloader)):
        if use_cuda:
            inputs = inputs.cuda()

        input_prediction = sorting_network_prediction(inputs)

        sum_tensor = input_prediction + flatten(sum_tensor)

    average_tensor = sum_tensor / dataset_size

    if use_cuda:
        average_tensor = average_tensor.cpu()

    return average_tensor


from scipy.spatial.distance import cosine

def key_function(dataset_sample, average_prediction):

    prediction = sorting_network_prediction(torch.unsqueeze(dataset_sample[0], 0))
    squeezed_prediction = torch.squeeze(prediction)

    similarity = cosine(average_prediction.cpu().numpy(), squeezed_prediction.cpu().numpy())

    return similarity


if args.sort:
    average_prediction = find_average_tensor(trainset)

    trainset = sorted(trainset, key=lambda x: key_function(x, average_prediction))[::-1]

# sorted(trainset, key=)
train_batch_size = 128
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=train_batch_size,
                                          shuffle=False,
                                          num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data',
                                       train=False,
                                       download=False,
                                       transform=transform_test)
test_batch_size = 100
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=test_batch_size,
                                         shuffle=False,
                                         num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('==> Building model..')
    # net = VGG('VGG19')
    net = ResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()


if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100. * correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.t7')
            best_acc = acc


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
