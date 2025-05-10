import json
import math
import os
import pickle
import random
from collections import OrderedDict
from typing import Dict, List, OrderedDict, Tuple, Union

import numpy as np
import torch
from path import Path
from torch.utils.data import DataLoader, Subset, random_split

# +
import numpy as np
import torch
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import to_pil_image
PROJECT_DIR = Path(__file__).parent.parent.parent.abspath()
LOG_DIR = PROJECT_DIR / "logs"
TEMP_DIR = PROJECT_DIR / "temp"
DATASETS_DIR = PROJECT_DIR / "datasets"
transform_train = transforms.Compose([
    transforms.Resize([128, 128]),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize([128, 128]),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 不是这里面的参数值
def get_dataloader(
    dataset: str,
    client_id: int,
    batch_size=20,
    valset_ratio=0.2,
    testset_ratio=0.1,
    only_dataset=False,
) -> Dict[str, Union[DataLoader, Subset]]:
    args_dict = json.load(open(DATASETS_DIR / "args.json", "r"))
    client_num_in_each_pickles = args_dict["client_num_in_each_pickles"]
    pickles_dir = DATASETS_DIR / dataset / "pickles"
    if os.path.isdir(pickles_dir) is False:
        raise RuntimeError("Please preprocess and create pickles first.")

    pickle_path = (
        pickles_dir / f"{math.floor(client_id / client_num_in_each_pickles)}.pkl"
    )
    with open(pickle_path, "rb") as f:
        subset = pickle.load(f)
    client_dataset = subset[client_id % client_num_in_each_pickles]
    val_samples_num = int(len(client_dataset) * valset_ratio)
    test_samples_num = int(len(client_dataset) * testset_ratio)
    train_samples_num = len(client_dataset) - val_samples_num - test_samples_num
    trainset, valset, testset = random_split(
        client_dataset, [train_samples_num, val_samples_num, test_samples_num]
    )
    if only_dataset:
        return {"train": trainset, "val": valset, "test": testset}
    trainloader = DataLoader(trainset, batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size)
    testloader = DataLoader(testset, batch_size)
    return {"train": trainloader, "val": valloader, "test": testloader}


def get_client_id_indices(
    dataset,
) -> Union[Tuple[List[int], List[int], int], Tuple[List[int], int]]:
    seperation = {"id":[],"total":None}
    seperation["id"] = [0,1,2,3,4,5,6,7,8,9]
    seperation["total"] = 10
    return seperation["id"], seperation["total"]


def fix_random_seed(seed: int) -> None:
    torch.cuda.empty_cache()
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def clone_parameters(
    src: Union[OrderedDict[str, torch.Tensor], torch.nn.Module]
) -> OrderedDict[str, torch.Tensor]:
    if isinstance(src, OrderedDict):
        return OrderedDict(
            {
                name: param.clone().detach().requires_grad_(param.requires_grad)
                for name, param in src.items()
            }
        )
    if isinstance(src, torch.nn.Module):
        return OrderedDict(
            {
                name: param.clone().detach().requires_grad_(param.requires_grad)
                for name, param in src.state_dict(keep_vars=True).items()
            }
        )


# +AD数据集
class AdDataset(Dataset):
    def __init__(self, train=True, transform=None, path=None):

        self_transform = ToTensor()

        # dataset = torchvision.datasets.ImageFolder(path)
        # 创建 ImageFolder 数据集
        dataset = ImageFolder(root=path, transform=transform_train)
        
        # 创建 DataLoader 加载数据
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

        # 获取所有训练集的标签和特征
        train_labels = []
        train_features = []
        for images, labels in train_loader:
            if images.shape[1] != 3:  # 判断图像的通道数是否为3
                # print(image.shape)
                images = transforms.Grayscale(num_output_channels=3)(
                    images)  # 此时，将使用 transforms.Grayscale() 转换函数将图像转换为灰度图像，并将通道数设置为3。

            train_features.append(images)
            train_labels.extend(labels.tolist())
        train_labels = train_labels  # 将列表转换为原始标签形式
        train_features = torch.cat(train_features, dim=0)  # 将图像张量拼接为一个张量

        # print(train_features.shape) torch.Size([1600, 3, 181, 217])


        self.images = train_features
        self.labels = torch.tensor(train_labels)
        self.transform = transform_train

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        image = self.images[idx]


        image = to_pil_image(image)
        image = self.transform(image)

        return image, label
    
    
# +
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
    
# +
def get_ad_data(n_parties, beta):
    root = r'/root/HyPeFL/datasets/mri_png'
    # root = r'/root/HyPeFL/datasets/PET_jpeg'
    dataset = AdDataset(train=True, path=root)
    train_ratio = 0.6  # 60% 的数据作为训练集，10% 的数据作为测试集, 20%公共数据集
    # 计算划分的样本数量
    train_size = int(train_ratio * len(dataset))
    test_size = int(0.1 * len(dataset))
    share_size = int(len(dataset)-train_size-test_size)
    # 使用 random_split 函数进行划分
    train_dataset, test_dataset, shared_dataset = random_split(dataset, [train_size, test_size, share_size])
    
    min_size = 0
    min_require_size = 10
    K = 3
    y_train = train_dataset.dataset.labels
    y_train = np.array(y_train)
    N = y_train.shape[0]
    # 存储参与方（parties）和其对应的数据索引
    net_dataidx_map = {}

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(n_parties)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, n_parties))
            proportions = np.array(
                [p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(n_parties):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]
        # print(net_dataidx_map)

    data_list = [None] * n_parties
    for i in range(n_parties):
        idxs = net_dataidx_map[i][:len(net_dataidx_map[i])]
        data_list[i] = DatasetSplit(dataset, idxs)
        
    loader_list = []
    for i in range(n_parties):
        loader = DataLoader(data_list[i], batch_size=32, shuffle=False)
        loader_list.append(loader)
    shared_loader = DataLoader(shared_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return loader_list, shared_loader, test_loader

# loader_list, shared_loader, test_loader = get_ad_data(2, 1)
# for x,y in loader_list[0]:
#     print(x.shape)



# # +
# def get_dataloader2(
#     dataset,
#     client_id: int,
#     batch_size=20,
#     valset_ratio=0.2,
#     testset_ratio=0.1,
#     only_dataset=False,
# ) -> Dict[str, Union[DataLoader, Subset]]:

#     client_dataset = dataset[client_id]
#     val_samples_num = int(len(client_dataset) * valset_ratio)
#     test_samples_num = int(len(client_dataset) * testset_ratio)
#     train_samples_num = len(client_dataset) - val_samples_num - test_samples_num
#     trainset, valset, testset = random_split(
#         client_dataset, [train_samples_num, val_samples_num, test_samples_num]
#     )
#     if only_dataset:
#         return {"train": trainset, "val": valset, "test": testset}
#     trainloader = DataLoader(trainset, batch_size, shuffle=True)
#     valloader = DataLoader(valset, batch_size, shuffle=False)
#     testloader = DataLoader(testset, batch_size, shuffle=False)
#     return {"train": trainloader, "val": valloader, "test": testloader}
