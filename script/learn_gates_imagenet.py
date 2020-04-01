from torchvision import transforms
import datasets
import torch
import argparse
import os

from gate import default_graph, apply_func, replace_func
from gate import init_convbn_gates, collect_convbn_gates, new_convbn_forward
import models
import misc

print = misc.logger.info

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--data', default='data/imagenet', type=str)
parser.add_argument('--arch', '-a', default='mobilenet_v1', type=str)
parser.add_argument('--sparsity_level', '-s', default=0.5, type=float)
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--lambd', default=0.05, type=float)
parser.add_argument('--log_interval', default=100, type=int)
parser.add_argument('--eval_interval', default=500, type=int)
parser.add_argument('--train_batch_size', default=100, type=int)
parser.add_argument('--expanded_inchannel', '-e', default=40, type=int)
parser.add_argument('--multiplier', '-m', default=1.0, type=float)
parser.add_argument('--seed', default=None, type=int)

args = parser.parse_args()
args.seed = misc.set_seed(args.seed)
args.num_classes = 1000

args.device = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

args.logdir = 'imagenet-%s/channel-%d-sparsity-%.2f' % (
    args.arch, args.expanded_inchannel, args.sparsity_level
)

misc.prepare_logging(args)

print('==> Preparing data..')

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.25, 1.0)),
    transforms.RandomHorizontalFlip(),
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
])

train_loader = torch.utils.data.DataLoader(
    datasets.ImageNet(args.data, 'train', transform_train),
    batch_size=args.train_batch_size, shuffle=True, num_workers=32,
    pin_memory=True, collate_fn=datasets.fast_collate
)
test_loader = torch.utils.data.DataLoader(
    datasets.ImageNet(args.data, 'val', transform_test),
    batch_size=50, shuffle=False, num_workers=32,
    pin_memory=True, collate_fn=datasets.fast_collate
)
print('==> Initializing model...')
model = models.__dict__[args.arch](args.num_classes, args.expanded_inchannel, args.multiplier)

if args.arch == 'mobilenet_v1':
    from gate import init_conv_depthwise_gates, new_conv_depthwise_forward, collect_conv_depthwise_gates
    init_func = init_conv_depthwise_gates
    new_forward = new_conv_depthwise_forward
    collect_gates = collect_conv_depthwise_gates
    module_type = 'ConvDepthWise'
elif args.arch == 'mobilenet_v2':
    from gate import init_inverted_block_gates, new_inverted_block_forward, collect_inverted_block_gates
    init_func = init_inverted_block_gates
    new_forward = new_inverted_block_forward
    collect_gates = collect_inverted_block_gates
    module_type = 'InvertedBlock'
elif args.arch == 'resnet50':
    from gate import init_bottleneck_gates, new_bottleneck_forward, collect_bottleneck_gates
    init_func = init_bottleneck_gates
    new_forward = new_bottleneck_forward
    collect_gates = collect_bottleneck_gates
    module_type = 'Bottleneck'
else:
    raise NotImplementedError


print('==> Transforming model...')
model_params = []
for params in model.parameters():
    ps = list(params.size())
    if len(ps) == 4 and ps[1] != 1:
        weight_decay = 1e-4
    elif len(ps) == 2:
        weight_decay = 1e-4
    else:
        weight_decay = 0
    item = {'params': params, 'weight_decay': weight_decay,
            'lr': 0.045, 'momentum': 0.9,
            'nesterov': True}
    model_params.append(item)

apply_func(model, 'ConvBNReLU', init_convbn_gates)
apply_func(model, module_type, init_func)
apply_func(model, 'ConvBNReLU', collect_convbn_gates)
apply_func(model, module_type, collect_gates)
replace_func(model, 'ConvBNReLU', new_convbn_forward)
replace_func(model, module_type, new_forward)

model = model.to(args.device)
criterion = torch.nn.CrossEntropyLoss().to(args.device)

gates_params = default_graph.get_tensor_list('gates_params')
optimizer = torch.optim.Adam(gates_params, lr=args.lr)

def test():
    test_losses = misc.AverageMeter()
    test_top1 = misc.AverageMeter()
    test_top5 = misc.AverageMeter()

    model.eval()
    prefetcher = datasets.DataPrefetcher(test_loader)
    with torch.no_grad():
        data, target = prefetcher.next()
        while data is not None:
            default_graph.clear_all_tensors()

            data, target = data.to(args.device), target.to(args.device)
            output = model(data)

            loss = criterion(output, target)
            prec1, prec5 = misc.accuracy(output, target, topk=(1, 5))
            test_losses.update(loss.item(), data.size(0))
            test_top1.update(prec1.item(), data.size(0))
            test_top5.update(prec5.item(), data.size(0))

            data, target = prefetcher.next()

    test_sparsity = (torch.cat(gates_params) != 0).float().mean().item()
    print(' * Test set: Loss_CE: %.4f, '
          'Sparsity: %.4f, Top1 acc: %.4f, Top5 acc: %.4f\n' % (
        test_losses.avg, test_sparsity, test_top1.avg, test_top5.avg
    ))
    return test_top1.avg, test_sparsity

best_acc = 0
top1 = misc.AverageMeter()
top5 = misc.AverageMeter()

prefetcher = datasets.DataPrefetcher(train_loader)
data, target = prefetcher.next()
i = -1
while data is not None:
    i += 1

    model.train()
    optimizer.zero_grad()
    output = model(data)
    loss_ce = criterion(output, target)
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
        prec1, prec5 = misc.accuracy(output, target, topk=(1, 5))
        top1.update(prec1.item(), data.size(0))
        top5.update(prec5.item(), data.size(0))

        print('Train Iter [%d/%d]\tLoss: %.4f, Loss_CE: %.4f, Loss_REG: %.4f, '
              'Sparsity: %.4f, Mean gate: %.4f, Top1 acc: %.4f, Top5 acc: %.4f' % (
            i, len(train_loader), loss.item(), loss_ce.item(), loss_reg.item(),
            sparsity.item(), mean_gate.item(), top1.avg, top5.avg
        ))

    if i % args.eval_interval == 0 and i > 0:
        acc, test_sparsity = test()
        if test_sparsity <= args.sparsity_level and acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(args.logdir, 'checkpoint.pth'))

            temp_params = []
            for j in range(len(gates_params)):
                temp_params.append(gates_params[j].data.clone().cpu())

            misc.dump_pickle(temp_params, os.path.join(args.logdir, 'channel_gates.pkl'))

    data, target = prefetcher.next()

if best_acc == 0:
    torch.save(model.state_dict(), os.path.join(args.logdir, 'checkpoint.pth'))

    temp_params = []
    for j in range(len(gates_params)):
        temp_params.append(gates_params[j].data.clone().cpu())

    misc.dump_pickle(temp_params, os.path.join(args.logdir, 'channel_gates.pkl'))
