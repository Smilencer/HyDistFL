import random
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]  # 返回的是第item个样本的标签与图像
        if isinstance(image, torch.Tensor):
            image = image.clone().detach()
        else:
            # image = torch.tensor(image)
            image = torch.stack(image)
        if isinstance(label, torch.Tensor):
            label = label.clone().detach()
        else:
            label = torch.tensor(label)
        # print(image.shape)
        return image, label
    
def get_datasets(data_name, dataroot, normalize=True, val_size=10000):
    """
    get_datasets returns train/val/test data splits of CIFAR10/100 datasets
    :param data_name: name of dataset, choose from [cifar10, cifar100]
    :param dataroot: root to data dir
    :param normalize: True/False to normalize the data
    :param val_size: validation split size (in #samples)
    :return: train_set, val_set, test_set (tuple of pytorch dataset/subset)
    """

    if data_name =='cifar10':
        normalization = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        data_obj = CIFAR10
    elif data_name == 'cifar100':
        normalization = transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
        data_obj = CIFAR100
    else:
        raise ValueError("choose data_name from ['mnist', 'cifar10', 'cifar100']")

    trans = [transforms.ToTensor()]

    if normalize:
        trans.append(normalization)

    transform = transforms.Compose(trans)

    dataset = data_obj(
        dataroot,
        train=True,
        download=True,
        transform=transform
    )

    test_set = data_obj(
        dataroot,
        train=False,
        download=True,
        transform=transform
    )

    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    return train_set, val_set, test_set

    
def gen_random_loaders(data_name, data_path, num_users, bz):

    
    dataloaders = []
    train_set, shared_set, test_set = get_datasets(data_name, data_path, normalize=True)
    x_train, y_train = train_set.dataset.data, train_set.dataset.targets
    # train_set = np.array(train_set)
    y_train = np.array(y_train)
    n_train = y_train.shape[0]

    min_size = 0
    min_require_size = 10
    K = 10
    if data_name == 'cifar100':
        K = 100
    
    N = y_train.shape[0]

    net_dataidx_map = {}
    while min_size < min_require_size:
        idx_batch = [[] for _ in range(num_users)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0] 
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(0.5, num_users))
            proportions = np.array([p * (len(idx_j) < N / (num_users+1)) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    
    
    for j in range(num_users):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]

        
    train_loader = []
    for j in range(num_users):
        temp = DataLoader(DatasetSplit(train_set.dataset,idx_batch[j]),batch_size=32,shuffle=True, drop_last=True)
        train_loader.append(temp)
    # print(len(train_loader))
    shared_loader = DataLoader(shared_set,batch_size=32,shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set,batch_size=32,shuffle=True)
    
 

    return train_loader, shared_loader, test_loader

train_loader, shared_loader, test_loader=gen_random_loaders("cifar100", "/root/HyPeFL/datasets/cifar100", 10, 32)
