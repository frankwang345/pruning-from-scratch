from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
import misc
import torchvision.transforms as tfm
import torchvision.datasets as ds
import torch


class CIFAR10(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    def __init__(self, root, type='train',
                 transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.type = type

        # now load the picked numpy arrays
        train_data = []
        train_labels = []
        for fentry in self.train_list:
            f = fentry[0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            train_data.append(entry['data'])
            if 'labels' in entry:
                train_labels += entry['labels']
            else:
                train_labels += entry['fine_labels']
            fo.close()

        train_data = np.concatenate(train_data)
        train_data = train_data.reshape((50000, 3, 32, 32))
        train_data = train_data.transpose((0, 2, 3, 1))  # convert to HWC

        if self.type == 'train':
            self.data = train_data[:45000]
            self.labels = train_labels[:45000]
        elif self.type == 'val':
            self.data = train_data[45000:]
            self.labels = train_labels[45000:]
        elif self.type == 'train+val':
            self.data = train_data
            self.labels = train_labels
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.data = entry['data']
            if 'labels' in entry:
                self.labels = entry['labels']
            else:
                self.labels = entry['fine_labels']
            fo.close()
            self.data = self.data.reshape((10000, 3, 32, 32))
            self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class ImageNet(data.Dataset):
    def __init__(self, root, type='train', transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.type = type
        all_train_image_list = misc.load_pickle(os.path.join(self.root, 'train_img_list.pkl'))
        all_test_image_list = misc.load_pickle(os.path.join(self.root, 'val_img_list.pkl'))
        self.train_image_list = []
        self.train_labels = []
        self.val_image_list = []
        self.val_labels = []
        self.test_image_list = []
        self.test_labels = []
        for i in range(1000):
            self.train_image_list += all_train_image_list[i][:-50]
            self.train_labels += [i] * len(all_train_image_list[i][:-50])
            self.val_image_list += all_train_image_list[i][-50:]
            self.val_labels += [i] * 50
            self.test_image_list += all_test_image_list[i]
            self.test_labels += [i] * 50

        if self.type == 'train':
            self.data = self.train_image_list
            self.labels = self.train_labels
        elif self.type == 'val':
            self.data = self.val_image_list
            self.labels = self.val_labels
        elif self.type == 'train+val':
            self.data = self.train_image_list + self.val_image_list
            self.labels = self.train_labels + self.val_labels
        elif self.type == 'test':
            self.data = self.test_image_list
            self.labels = self.test_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img_path = self.data[item]
        target = self.labels[item]
        img = misc.pil_loader(img_path)
        if self.transform is not None:
            img = self.transform(img)

        return img, target


imagenet_pca = {
    'eigval': np.asarray([0.2175, 0.0188, 0.0045]),
    'eigvec': np.asarray([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ])
}


class Lighting(object):
    def __init__(self, alphastd,
                 eigval=imagenet_pca['eigval'],
                 eigvec=imagenet_pca['eigvec']):
        self.alphastd = alphastd
        assert eigval.shape == (3,)
        assert eigvec.shape == (3, 3)
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0.:
            return img
        rnd = np.random.randn(3) * self.alphastd
        rnd = rnd.astype('float32')
        v = rnd
        old_dtype = np.asarray(img).dtype
        v = v * self.eigval
        v = v.reshape((3, 1))
        inc = np.dot(self.eigvec, v).reshape((3,))
        img = np.add(img, inc)
        if old_dtype == np.uint8:
            img = np.clip(img, 0, 255)
        img = Image.fromarray(img.astype(old_dtype), 'RGB')
        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'


def fast_collate(batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros( (len(imgs), 3, h, w), dtype=torch.uint8 )
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] += torch.from_numpy(nump_array)

    return tensor, targets


def get_imagenet_loader(root, batch_size, type='train', mobile_setting=True):
    crop_scale = 0.25 if mobile_setting else 0.08
    jitter_param = 0.4
    lighting_param = 0.1
    if type == 'train':
        transform = tfm.Compose([
            tfm.RandomResizedCrop(224, scale=(crop_scale, 1.0)),
            tfm.ColorJitter(
                brightness=jitter_param, contrast=jitter_param,
                saturation=jitter_param),
            Lighting(lighting_param),
            tfm.RandomHorizontalFlip(),
        ])

    elif type == 'test':
        transform = tfm.Compose([
            tfm.Resize(256),
            tfm.CenterCrop(224),
        ])

    dataset = ds.ImageFolder(root, transform)
    sampler = data.distributed.DistributedSampler(dataset)
    data_loader = data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, sampler=sampler, collate_fn=fast_collate
    )
    if type == 'train':
        return data_loader, sampler

    elif type == 'test':
        return data_loader


class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(async=True)
            self.next_target = self.next_target.cuda(async=True)
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target