import torch
import argparse
import shutil
import os
import torch.nn.parallel
import torch.distributed as dist
import torch.nn as nn

import models
import datasets
import misc
import math
from apex.parallel import DistributedDataParallel as DDP
import warnings
warnings.filterwarnings("ignore")

def print(msg):
    if args.local_rank == 0:
        misc.logger.info(msg)

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='data/imagenet', type=str)
parser.add_argument('--arch', '-a', default='mobilenet_v1', type=str)
parser.add_argument('--lr_gamma', default=0.975, type=float)
parser.add_argument('--lr_scheduler', default='cos', type=str)
parser.add_argument('--lr', default=0.05, type=float)
parser.add_argument('--mm', default=0.9, type=float)
parser.add_argument('--wd', default=4e-5, type=float)
parser.add_argument('--epochs', default=150, type=int)
parser.add_argument('--log_interval', default=50, type=int)
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N', help='mini-batch size per process (default: 256)')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--sparsity_level', '-s', default=0.5, type=float)
parser.add_argument('--pruned_ratio', '-p', default=0.5, type=float)
parser.add_argument('--expanded_inchannel', '-e', default=40, type=int)
parser.add_argument('--multiplier', '-m', default=1.0, type=float)
parser.add_argument('--budget_train', action='store_true')
parser.add_argument('--label_smooth', action='store_true')

args = parser.parse_args()
if args.budget_train:
    args.epochs = 200 if args.arch == 'resnet50' else 300

args.logdir = 'imagenet-%s/channel-%d-pruned-%.2f' % (
    args.arch, args.expanded_inchannel, args.pruned_ratio
)

if args.budget_train:
    args.logdir += '-B'
if args.label_smooth:
    args.logdir += '-smooth'

torch.backends.cudnn.benchmark = True
args.distributed = False
if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1

args.gpu = 0
args.world_size = 1

if args.distributed:
    args.gpu = args.local_rank % torch.cuda.device_count()
    torch.cuda.set_device(args.gpu)
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')
    args.world_size = torch.distributed.get_world_size()

if args.local_rank == 0:
    misc.prepare_logging(args)

print("=> Using model {}".format(args.arch))
pruned_cfg = misc.load_pickle('logs/imagenet-%s/channel-%d-sparsity-%.2f/pruned_cfg-%.2f.pkl' % (
    args.arch, args.expanded_inchannel, args.sparsity_level, args.pruned_ratio
))

model = models.__dict__[args.arch](1000, args.expanded_inchannel, args.multiplier, pruned_cfg)
model = model.cuda()
model = DDP(model, delay_allreduce=True)
# define loss function (criterion) and optimizer
criterion = nn.CrossEntropyLoss().cuda()
if args.label_smooth:
    class CrossEntropyLabelSmooth(nn.Module):
        def __init__(self, num_classes, epsilon):
            super(CrossEntropyLabelSmooth, self).__init__()
            self.num_classes = num_classes
            self.epsilon = epsilon
            self.logsoftmax = nn.LogSoftmax(dim=1)

        def forward(self, inputs, targets):
            log_probs = self.logsoftmax(inputs)
            targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
            targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
            loss = (-targets * log_probs).mean(0).sum()
            return loss
    criterion = CrossEntropyLabelSmooth(num_classes=1000, epsilon=0.1).cuda()

print('==> Preparing data..')
train_loader, train_sampler = datasets.get_imagenet_loader(
    os.path.join(args.data, 'train'), args.batch_size, type='train', mobile_setting=(not args.arch == 'resnet50')
)
test_loader = datasets.get_imagenet_loader(
    os.path.join(args.data, 'val'), 100, type='test', mobile_setting=(not args.arch == 'resnet50')
)

def get_lr_scheduler(optimizer):
    """get learning rate"""

    if args.lr_scheduler == 'multistep':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[int(0.3*args.epochs), int(0.6*args.epochs), int(0.9*args.epochs)],
            gamma=0.1)

    elif args.lr_scheduler == 'cos':
        lr_dict = {}
        for i in range(args.epochs):
            lr_dict[i] = 0.5 * (1 + math.cos(math.pi * i / args.epochs))
        lr_lambda = lambda epoch: lr_dict[epoch]
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda)
    return lr_scheduler

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= args.world_size
    return rt

def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]

args.lr = args.lr*float(args.batch_size*args.world_size)/256.
# all depthwise convolution (N, 1, x, x) has no weight decay
# weight decay only on normal conv and fc
model_params = []
for params in model.parameters():
    ps = list(params.size())
    if len(ps) == 4 and ps[1] != 1:
        weight_decay = args.wd
    elif len(ps) == 2:
        weight_decay = args.wd
    else:
        weight_decay = 0
    item = {'params': params, 'weight_decay': weight_decay,
            'lr': args.lr, 'momentum': args.mm,
            'nesterov': True}
    model_params.append(item)

optimizer = torch.optim.SGD(model_params)
lr_scheduler = get_lr_scheduler(optimizer)

def train(train_loader, model, criterion, optimizer, epoch):
    losses = misc.AverageMeter()
    top1 = misc.AverageMeter()
    top5 = misc.AverageMeter()

    # switch to train mode
    prefetcher = datasets.DataPrefetcher(train_loader)
    model.train()

    input, target = prefetcher.next()
    i = -1
    while input is not None:
        i += 1

        output = model(input)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.log_interval == 0:
            prec1, prec5 = misc.accuracy(output.data, target, topk=(1, 5))

            # Average loss and accuracy across processes for logging
            reduced_loss = reduce_tensor(loss.data)
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)

            # to_python_float incurs a host<->device sync
            losses.update(to_python_float(reduced_loss), input.size(0))
            top1.update(to_python_float(prec1), input.size(0))
            top5.update(to_python_float(prec5), input.size(0))

            torch.cuda.synchronize()

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.10f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), loss=losses, top1=top1, top5=top5))

        input, target = prefetcher.next()


def validate(val_loader, model, criterion, epoch):
    losses = misc.AverageMeter()
    top1 = misc.AverageMeter()
    top5 = misc.AverageMeter()

    # switch to evaluate mode
    prefetcher = datasets.DataPrefetcher(val_loader)
    model.eval()

    input, target = prefetcher.next()
    i = -1
    while input is not None:
        i += 1
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = misc.accuracy(output.data, target, topk=(1, 5))

        reduced_loss = reduce_tensor(loss.data)
        prec1 = reduce_tensor(prec1)
        prec5 = reduce_tensor(prec5)

        losses.update(to_python_float(reduced_loss), input.size(0))
        top1.update(to_python_float(prec1), input.size(0))
        top5.update(to_python_float(prec5), input.size(0))

        input, target = prefetcher.next()

    print(' * Test Epoch {0}, Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}\n'
          .format(epoch, top1=top1, top5=top5))

    return top1.avg


# main
best_prec1 = 0
for epoch in range(args.epochs):
    if args.distributed:
        train_sampler.set_epoch(epoch)

    lr_scheduler.step()
    # train for one epoch
    train(train_loader, model, criterion, optimizer, epoch)

    # evaluate on validation set
    prec1 = validate(test_loader, model, criterion, epoch)

    # remember best prec@1 and save checkpoint
    if args.local_rank == 0:
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        torch.save({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, os.path.join(args.logdir, 'checkpoint.pth.tar'))

        if is_best:
            shutil.copyfile(os.path.join(args.logdir, 'checkpoint.pth.tar'),
                            os.path.join(args.logdir, 'model_best.pth.tar'))
            print(' * Save best model @ Epoch {}\n'.format(epoch))
