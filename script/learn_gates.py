from torchvision import transforms
import datasets
import torch.nn.functional as F
import numpy as np
import torch
import argparse
import os

from gate import default_graph, apply_func, replace_func
import models
import misc

print = misc.logger.info

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--arch', '-a', default='vgg16_bn', type=str)
parser.add_argument('--sparsity_level', '-s', default=0.2, type=float)
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--lambd', default=0.5, type=float)
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--log_interval', default=100, type=int)
parser.add_argument('--train_batch_size', default=128, type=int)
parser.add_argument('--expanded_inchannel', '-e', default=80, type=int)
parser.add_argument('--seed', default=None, type=int)

args = parser.parse_args()
args.seed = misc.set_seed(args.seed)
args.num_classes = 10

args.device = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

args.logdir = 'seed-%d/%s-%s/channel-%d-sparsity-%.2f' % (
    args.seed, args.dataset, args.arch, args.expanded_inchannel, args.sparsity_level
)

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

trainset = datasets.CIFAR10(root='./data/cifar10', type='train', transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=2)

valset = datasets.CIFAR10(root='./data/cifar10', type='val', transform=transform_val)
valloader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False, num_workers=2)

print('==> Initializing model...')
model = models.__dict__[args.arch](args.num_classes, args.expanded_inchannel)

if args.arch.find('vgg') != -1:
    from gate import init_convbn_gates, new_convbn_forward, collect_convbn_gates
    init_func = init_convbn_gates
    new_forward = new_convbn_forward
    collect_gates = collect_convbn_gates
    module_type = 'ConvBNReLU'

elif args.arch.find('resnet') != -1:
    from gate import init_basicblock_gates, new_basicblock_forward, collect_basicblock_gates
    init_func = init_basicblock_gates
    new_forward = new_basicblock_forward
    collect_gates = collect_basicblock_gates
    module_type = 'BasicBlock'

elif args.arch.find('densenet') != -1:
    from gate import init_channel_selection_gates, new_channel_selection_forward, collect_channel_selection_gates
    init_func = init_channel_selection_gates
    new_forward = new_channel_selection_forward
    collect_gates = collect_channel_selection_gates
    module_type = 'ChannelSelection'

else:
    raise NotImplementedError


print('==> Transforming model...')

apply_func(model, module_type, init_func)
apply_func(model, module_type, collect_gates)
replace_func(model, module_type, new_forward)

model = model.to(args.device)

gates_params = default_graph.get_tensor_list('gates_params')
optimizer = torch.optim.Adam(gates_params, lr=args.lr)

def train(epoch):
    model.train()
    for i, (data, target) in enumerate(trainloader):
        data = data.to(args.device)
        target = target.to(args.device)

        optimizer.zero_grad()
        output = model(data)
        loss_ce = F.cross_entropy(output, target)
        loss_reg = args.lambd * (torch.cat(gates_params).abs().mean() - args.sparsity_level) ** 2
        loss = loss_ce + loss_reg

        loss.backward()
        optimizer.step()

        for p in gates_params:
            p.data.clamp_(0, 1)

        if i % args.log_interval == 0:
            concat_channels = torch.cat(gates_params)
            sparsity = (concat_channels != 0).float().mean()
            mean_gate = concat_channels.mean()
            acc = (output.max(1)[1] == target).float().mean()

            print('Train Epoch: %d [%d/%d]\tLoss: %.4f, Loss_CE: %.4f, Loss_REG: %.4f, '
                  'Sparsity: %.4f, Mean gate: %.4f, Accuracy: %.4f' % (
                epoch, i, len(trainloader), loss.item(), loss_ce.item(), loss_reg.item(),
                sparsity.item(), mean_gate.item(), acc.item()
            ))


def test():
    model.eval()
    test_loss_ce = []
    correct = 0
    with torch.no_grad():
        for data, target in valloader:
            default_graph.clear_all_tensors()

            data, target = data.to(args.device), target.to(args.device)
            output = model(data)

            test_loss_ce.append(F.cross_entropy(output, target).item())

            pred = output.max(1)[1]
            correct += (pred == target).float().sum().item()

    test_sparsity = (torch.cat(gates_params) != 0).float().mean()
    acc = correct / len(valloader.dataset)
    print('Test set: Loss_CE: %.4f, '
          'Sparsity: %.4f, Accuracy: %.4f\n' % (
        np.mean(test_loss_ce),
        test_sparsity.item(), acc
    ))
    return acc, test_sparsity

best_acc = 0
for epoch in range(args.epochs):
    train(epoch)
    acc, test_sparsity = test()
    if test_sparsity <= args.sparsity_level and acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), os.path.join(args.logdir, 'checkpoint.pth'))

        temp_params = []
        for i in range(len(gates_params)):
            temp_params.append(gates_params[i].data.clone().cpu())

        misc.dump_pickle(temp_params, os.path.join(args.logdir, 'channel_gates.pkl'))

if best_acc == 0:
    torch.save(model.state_dict(), os.path.join(args.logdir, 'checkpoint.pth'))

    temp_params = []
    for i in range(len(gates_params)):
        temp_params.append(gates_params[i].data.clone().cpu())

    misc.dump_pickle(temp_params, os.path.join(args.logdir, 'channel_gates.pkl'))
