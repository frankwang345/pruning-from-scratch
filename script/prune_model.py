import torch
import misc
import argparse
import os
import flop_counter
import models

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--arch', '-a', default='vgg16_bn', type=str)
parser.add_argument('--sparsity_level', '-s', default=0.2, type=float)
parser.add_argument('--pruned_ratio', '-p', default=0.5, type=float)
parser.add_argument('--max_iter', default=10, type=int)
parser.add_argument('--expanded_inchannel', '-e', default=80, type=int)
parser.add_argument('--seed', default=None, type=int)

args = parser.parse_args()
args.seed = misc.set_seed(args.seed)

args.device = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

args.eps = 0.001
args.num_classes = 10

args.logdir = 'logs/seed-%d/%s-%s/channel-%d-sparsity-%.2f' % (
    args.seed, args.dataset, args.arch, args.expanded_inchannel, args.sparsity_level
)

gates_params = misc.load_pickle(os.path.join(args.logdir, 'channel_gates.pkl'))

def calculate_flops(model, input_size=(1, 3, 32, 32)):
    model = flop_counter.add_flops_counting_methods(model)
    model.eval().start_flops_count()
    inp = torch.randn(*input_size)
    out = model(inp)
    flops = model.compute_average_flops_cost()
    return flops

print('==> Initializing full model...')
model = models.__dict__[args.arch](args.num_classes)

full_flops = calculate_flops(model)
print('Full model FLOPS = %.4f (M)' % (full_flops / 1e6))

all_gates = torch.cat(gates_params)
gates_lens = [len(p) for p in gates_params]

start_pruned_ratio = 0
end_pruned_ratio = 1

pruned_cfg = models.expanded_cfg(args.expanded_inchannel)[args.arch]

for j in range(args.max_iter):
    cur_pruned_ratio = (start_pruned_ratio + end_pruned_ratio) / 2
    reserved_channel_num = round(len(all_gates) * (1 - cur_pruned_ratio))
    reserved_index = all_gates.topk(reserved_channel_num)[1]
    mask = torch.zeros(len(all_gates))
    mask[reserved_index] = 1
    masks = torch.split_with_sizes(mask, gates_lens)

    counter = 0
    for i in range(len(pruned_cfg)):
        if pruned_cfg[i] == 'M':
            continue
        else:
            pruned_cfg[i] = masks[counter].sum().long().item()
            counter += 1

    model = models.__dict__[args.arch](args.num_classes, args.expanded_inchannel, pruned_cfg)

    pruned_flops = calculate_flops(model)
    actual_pruned_ratio = 1 - pruned_flops / full_flops
    print('Iter %d, start %.2f, end %.2f, pruned ratio = %.4f' % (
        j, start_pruned_ratio, end_pruned_ratio, actual_pruned_ratio
    ))

    if abs(actual_pruned_ratio - args.pruned_ratio) / args.pruned_ratio <= args.eps:
        print('Successfully reach the target pruned ratio with FLOPS = %.4f (M)' % (
            pruned_flops / 1e6
        ))
        break

    if actual_pruned_ratio > args.pruned_ratio:
        end_pruned_ratio = cur_pruned_ratio
    else:
        start_pruned_ratio = cur_pruned_ratio

misc.dump_pickle(pruned_cfg, os.path.join(args.logdir, 'pruned_cfg-%.2f.pkl' % args.pruned_ratio))
misc.dump_pickle(masks, os.path.join(args.logdir, 'masks-%.2f.pkl' % args.pruned_ratio))
