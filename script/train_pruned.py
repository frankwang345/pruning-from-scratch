from torchvision import transforms
import datasets
import torch.nn.functional as F
import torch
import argparse
import os

import models
import misc

print = misc.logger.info

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--arch', '-a', default='vgg16_bn', type=str)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--mm', default=0.9, type=float)
parser.add_argument('--wd', default=1e-4, type=float)
parser.add_argument('--epochs', default=160, type=int)
parser.add_argument('--log_interval', default=100, type=int)
parser.add_argument('--train_batch_size', default=128, type=int)
parser.add_argument('--sparsity_level', '-s', default=0.2, type=float)
parser.add_argument('--pruned_ratio', '-p', default=0.7, type=float)
parser.add_argument('--expanded_inchannel', '-e', default=80, type=int)
parser.add_argument('--seed', default=None, type=int)
parser.add_argument('--budget_train', action='store_true')

args = parser.parse_args()
args.seed = misc.set_seed(args.seed)

if args.budget_train:
    args.epochs = int(1 / (1 - args.pruned_ratio) * args.epochs)

args.num_classes = 10

args.device = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

args.logdir = 'seed-%d/%s-%s/channel-%d-pruned-%.2f' % (
    args.seed, args.dataset, args.arch, args.expanded_inchannel, args.pruned_ratio
)

if args.budget_train:
    args.logdir += '-B'

misc.prepare_logging(args)

print('==> Preparing data..')

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = datasets.CIFAR10(root='./data/cifar10', type='train+val', transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=2)

testset = datasets.CIFAR10(root='./data/cifar10', type='test', transform=transform_val)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

print('==> Initializing model...')
pruned_cfg = misc.load_pickle('logs/seed-%d/%s-%s/channel-%d-sparsity-%.2f/pruned_cfg-%.2f.pkl' % (
    args.seed, args.dataset, args.arch, args.expanded_inchannel, args.sparsity_level, args.pruned_ratio
))

model = models.__dict__[args.arch](args.num_classes, args.expanded_inchannel, pruned_cfg)

model = model.to(args.device)

optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.mm, weight_decay=args.wd)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[int(args.epochs * 0.5), int(args.epochs * 0.75)], gamma=0.1
)


def train(epoch):
    model.train()
    for i, (data, target) in enumerate(trainloader):
        data = data.to(args.device)
        target = target.to(args.device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        pred = output.max(1)[1]
        acc = (pred == target).float().mean()

        if i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}, Accuracy: {:.4f}'.format(
                epoch, i, len(trainloader), loss.item(), acc.item()
            ))

def evaluate(loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(args.device), target.to(args.device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1)[1]
            correct += (pred == target).float().sum().item()

    test_loss /= len(loader.dataset)
    acc = correct / len(loader.dataset)
    print('Val set: Average loss: {:.4f}, Accuracy: {:.4f}\n'.format(
        test_loss, acc
    ))
    return acc

for epoch in range(args.epochs):
    scheduler.step()
    train(epoch)
    evaluate(testloader)

torch.save(model.state_dict(), os.path.join(args.logdir, 'best_checkpoint.pth'))
test_acc = evaluate(testloader)
print('Final saved model test accuracy = %.4f' % test_acc)

