import numpy as np
import torch
import os
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.autograd import Variable
import argparse
from torchsummary import summary
import matplotlib.pyplot as plt
from fvcore.nn import FlopCountAnalysis
from tensorboardX import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from loss.labelsmoothing_loss import LabelSmoothingCrossEntropy
from tools.utils import set_seeds
from models.cnn import CNN
from models.mlp import MLP_WISDM


parser = argparse.ArgumentParser(description='train-WISDM')
parser.add_argument('--output_dir', type=str, default='logs/model/WISDM/')
parser.add_argument('--resume', default=False, help='checkpoint path')
parser.add_argument('--model', type=str, choices=['cnn', 'mlp'], default='cnn', help='Model type to train')

args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
set_seeds(3)

acc = 0
acc_best = 0
accuracy_list = []
epoch_list = []
f1_list = []

train_x = torch.from_numpy(np.load('./data/wisdm/train_x.npy', encoding="latin1")).float()
train_y = torch.from_numpy(np.load('./data/wisdm/train_y.npy', encoding="latin1")).long()
test_x = torch.from_numpy(np.load('./data/wisdm/test_x.npy', encoding="latin1")).float()
test_y = torch.from_numpy(np.load('./data/wisdm/test_y.npy', encoding="latin1")).long()

train_x = train_x.unsqueeze(1)
train_y = torch.topk(train_y, 1)[1].squeeze(1)
test_x = test_x.unsqueeze(1)
test_y = torch.topk(test_y, 1)[1].squeeze(1)

padding = nn.ZeroPad2d(padding=(0, 0, 0, 0))
train_x = padding(train_x)
test_x = padding(test_x)

data_train = TensorDataset(train_x, train_y)
data_test = TensorDataset(test_x, test_y)
data_train_loader = torch.utils.data.DataLoader(data_train, batch_size=512, shuffle=True, num_workers=2)
data_test_loader = torch.utils.data.DataLoader(data_test, batch_size=2560, shuffle=True, num_workers=0)

if args.model == 'cnn':
    net = CNN().cuda()
elif args.model == 'mlp':
    net = MLP_WISDM().cuda()

criterion = LabelSmoothingCrossEntropy(eps=0.1, reduction='mean').cuda()
learning_rate = 1e-3
optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=1e-7)

def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(epoch):
    adjust_learning_rate(optimizer, epoch)
    net.train()
    loss_list, batch_list = [], []
    for i, (datas, labels) in enumerate(data_train_loader):
        datas, labels = Variable(datas).cuda(), Variable(labels).cuda()

        optimizer.zero_grad()

        output = net(datas)
        loss = criterion(output, labels)

        loss_list.append(loss.data.item())
        batch_list.append(i + 1)

        if i == 1:
            print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.data.item()))

        loss.backward()
        optimizer.step()

def test():
    global acc, acc_best
    net.eval()
    total_correct = 0
    avg_loss = 0.0
    with torch.no_grad():
        for i, (datas, labels) in enumerate(data_test_loader):
            datas, labels = Variable(datas).cuda(), Variable(labels).cuda()
            output = net(datas)
  
            avg_loss += criterion(output, labels).sum()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()

    avg_loss /= len(data_test)
    acc = float(total_correct) / len(data_test)
    if acc_best < acc:
        acc_best = acc
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.data.item(), acc))

    f11 = f1_score(labels.cpu().numpy(), pred.cpu().numpy(), average='weighted')
    f12 = f1_score(labels.cpu().numpy(), pred.cpu().numpy(), average='micro')
    f13 = f1_score(labels.cpu().numpy(), pred.cpu().numpy(), average='macro')
    print('Accuracy: %.8f' % acc, '| F1: %.8f' % f11, '| Micro: %.8f' % f12, '| Macro: %.8f' % f13)

def train_and_test(epoch):
    train(epoch)
    test()

def main():
    epoch = 200
    for e in range(1, epoch):
        train_and_test(e)
    torch.save(net, args.output_dir + f'WISDM_{args.model}_model.pth')

if __name__ == '__main__':
    summary(net, (1, 200, 3))
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Total Number of params: {} | Trainable Number of params: {}'.format(total_num, trainable_num))

    tensor = (torch.rand(1, 1, 200, 3)).cuda()
    flops = FlopCountAnalysis(net, tensor)
    print("FLOPs: ", flops.total())

    main()
