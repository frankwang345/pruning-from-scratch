from PIL import Image
import os
import shutil
import pickle as pkl
import time
from datetime import datetime
import numpy as np
import torch
import random


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class Logger(object):
    def __init__(self):
        self._logger = None

    def init(self, logdir, name='log'):
        if self._logger is None:
            import logging
            if not os.path.exists(logdir):
                os.makedirs(logdir)
            log_file = os.path.join(logdir, name)
            if os.path.exists(log_file):
                os.remove(log_file)
            self._logger = logging.getLogger()
            self._logger.setLevel('INFO')
            fh = logging.FileHandler(log_file)
            ch = logging.StreamHandler()
            self._logger.addHandler(fh)
            self._logger.addHandler(ch)

    def info(self, str_info):
        now = datetime.now()
        display_now = str(now).split(' ')[1][:-3]
        self.init(os.path.expanduser('~/tmp_log'), 'tmp.log')
        self._logger.info('[' + display_now + ']' + ' ' + str_info)

logger = Logger()


def ensure_dir(path, erase=False):
    if os.path.exists(path) and erase:
        print("Removing old folder {}".format(path))
        shutil.rmtree(path)
    if not os.path.exists(path):
        print("Creating folder {}".format(path))
        os.makedirs(path)


def load_pickle(path, verbose=True):
    begin_st = time.time()
    with open(path, 'rb') as f:
        if verbose:
            print("Loading pickle object from {}".format(path))
        v = pkl.load(f)
    if verbose:
        print("=> Done ({:.4f} s)".format(time.time() - begin_st))
    return v


def dump_pickle(obj, path):
    with open(path, 'wb') as f:
        print("Dumping pickle object to {}".format(path))
        pkl.dump(obj, f, protocol=pkl.HIGHEST_PROTOCOL)


def prepare_logging(args):
    args.logdir = os.path.join('./logs', args.logdir)

    logger.init(args.logdir, 'log')

    ensure_dir(args.logdir)
    logger.info("=================FLAGS==================")
    for k, v in args.__dict__.items():
        logger.info('{}: {}'.format(k, v))
    logger.info("========================================")


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def set_seed(seed=None):
    if seed is None:
        seed = random.randint(0, 9999)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    return seed