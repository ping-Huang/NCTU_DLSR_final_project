'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import argparse
import os
from datetime import datetime

from benchmark import benchmarking
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
import torchvision.transforms as transforms

#from models import ShuffleNetV2
from models import resnet20_cifar
from utils import progress_bar
from collections import OrderedDict


# pylint: disable=invalid-name

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--use_cpu', action='store_true',help='use_cpu')
args = parser.parse_args()

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
if args.use_cpu:
    device = 'cpu'
else:
    device = 'cuda' 
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
total_epoch = 300  # The number of total epochs to run

# Data
print('==> Preparing data..')

cinic_directory = './data/cinic-10'
cinic_mean = [0.47889522, 0.47227842, 0.43047404]
cinic_std = [0.24205776, 0.23828046, 0.25874835]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=cinic_mean, std=cinic_std)
])

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=cinic_mean, std=cinic_std)
])

trainset = torchvision.datasets.ImageFolder(
    root=(cinic_directory + '/train'), transform=train_transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

validset = torchvision.datasets.ImageFolder(
    root=(cinic_directory + '/valid'), transform=transform)
validloader = torch.utils.data.DataLoader(
    validset, batch_size=128, shuffle=False, num_workers=2)

testset = torchvision.datasets.ImageFolder(
    root=(cinic_directory + '/test'), transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=2)

classes = ('airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
#net = VGG('VGG16')
#net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
#net = ShuffleNetV2(0.5)
net = resnet20_cifar()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=1e-4)
scheduler = CosineAnnealingLR(
    optimizer=optimizer, T_max=total_epoch, eta_min=0)


#checkpoint = torch.load('./checkpoint/checkpoint_finetuned.pth.tar')
#net.load_state_dict(checkpoint['state_dict'])
#print(checkpoint.keys())

def train(epoch):
    ''' Trains the net on the train dataset for one entire iteration '''
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def validate(epoch):
    ''' Validates the net's accuracy on validation dataset and saves if better
        accuracy than previously seen. '''
    global best_acc
    net.eval()
    valid_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(validloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            valid_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(validloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (valid_loss, 100.*correct/total, correct, total))
                         #% (valid_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        torch.save(net, './checkpoint/ckpt.pt')
        best_acc = acc
    print('Best: %.3f' % best_acc)


def test():
    ''' Final test of the best performing net on the testing dataset. '''
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    #checkpoint = torch.load('./checkpoint/ckpt.t7')
    checkpoint = torch.load('./checkpoint/resnet20_82.52/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    print('Test best performing net from epoch {} with accuracy {:.3f}%'.format(
        checkpoint['epoch'], checkpoint['acc']))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

def main():
    start_time = datetime.now()
    print('Runnning training and test for {} epochs'.format(total_epoch))

    # Run training for specified number of epochs
    for epoch in range(start_epoch, start_epoch+total_epoch):
        scheduler.step()
        train(epoch)
        validate(epoch)

    time_elapsed = datetime.now() - start_time
    print('Training time elapsed (hh:mm:ss.ms) {}\n'.format(time_elapsed))

    # Run final test on never before seen data
    test()
@benchmarking(team=6, task=0, model=net, preprocess_fn=None)
def inference(net, data_loader,**kwargs):

    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    #checkpoint = torch.load('./checkpoint/ckpt.t7')
    if args.use_cpu:
        checkpoint = torch.load('./checkpoint/resnet20_82.52/ckpt.t7')
        
        new_state_dict = OrderedDict()
        for k, v in checkpoint['net'].items():
            name = k[7:] # remove `module.`
            print(name)
            new_state_dict[name] = v
        # load params
        net.load_state_dict(new_state_dict)
    else:
        checkpoint = torch.load('./checkpoint/resnet20_82.52/ckpt.t7')
        net.load_state_dict(checkpoint['net'])
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    print('Test best performing net from epoch {} with accuracy {:.3f}%'.format(
        checkpoint['epoch'], checkpoint['acc']))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            #progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
if __name__ == '__main__':
    #main()
    inference()
