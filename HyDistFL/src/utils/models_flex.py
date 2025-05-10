import os
import pickle
from collections import OrderedDict
from typing import Dict, List, OrderedDict, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

from .util import TEMP_DIR
from .util import get_dataloader
from torch.utils.data import Subset, TensorDataset, DataLoader

class Linear(nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))

        nn.init.uniform_(self.weight)
        nn.init.constant_(self.bias, 0.0)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


# 自定义的cnn嵌入层
class CNNEmbed(nn.Module):
    def __init__(self, embed_y, dim_y, embed_dim, device, in_channels=3, n_kernels=16):
        super(CNNEmbed, self).__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # embed_y表示是否将标签 y 嵌入到输入中
        in_channels += embed_y * dim_y

        self.conv1 = nn.Conv2d(in_channels, n_kernels, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(n_kernels, 2 * n_kernels, 5)
        self.fc1 = nn.Linear(2 * n_kernels * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, embed_dim)

        self.embed_dim = embed_dim
        self.embed_y = embed_y
        self.device = device

    def forward(self, dl: DataLoader):  # 修改输入为DataLoader
        embedding = torch.zeros(self.embed_dim).to(self.device)
        num_batches = len(dl)
        l = 0
        for i, B in enumerate(dl):
            l += len(B)
            B = tuple(t.to(self.device) for t in B)
            embedding += self._cnn_forward(B).sum(0)
            if i + 1 == num_batches:
                break
                
        embedding = embedding / l  
        with torch.no_grad():
                m, s = torch.mean(embedding), torch.std(embedding)
        embedding = (embedding - m) / s
        return embedding

    def _cnn_forward(self, B):
        x, y = B
        if self.embed_y:   # 是否将标签也嵌入到输入中 10
            y = y.view(y.size(0), y.size(1), 1, 1)
            c = torch.zeros((x.size(0), y.size(1), x.size(2), x.size(3))).to(self.device)
            c += y
            inp = torch.cat((x, c), dim=1)
        else:
            inp = x
        x = x.to(self.device)
        # 原有卷积网络前向传播逻辑
        x = self.pool(F.relu(self.conv1(inp)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.to(self.device)
        

class HyperNetwork(nn.Module):
    def __init__(
        self,
        embed_y: bool,      # 是否嵌入标签
        dim_y: int,         # 标签维度 
        # 原来的超网络参数
        embedding_dim: int,
        client_num: int,
        hidden_dim: int,
        backbone: nn.Module,
        K: int,
        in_channels: int=3, # 图像输入通道数
        n_kernels: int=16,  # 卷积核数量、
        gpu=True,
    ):
        super(HyperNetwork, self).__init__()
        self.device = torch.device(
            "cuda" if gpu and torch.cuda.is_available() else "cpu"
        )
        self.K = K
        self.client_num = client_num
        self.embedding = CNNEmbed(
            embed_y=embed_y,
            dim_y=dim_y,
            embed_dim=embedding_dim,
            device=self.device,
            in_channels=in_channels,
            n_kernels=n_kernels
        ).to(self.device)
        self.blocks_name = set(n.split(".")[0] for n, _ in backbone.named_parameters())
        self.cache_dir = TEMP_DIR / "hn"
        if not os.path.isdir(self.cache_dir):
            os.system(f"mkdir -p {self.cache_dir}")

        if os.listdir(self.cache_dir) != client_num:
            for client_id in range(client_num):
                with open(self.cache_dir / f"{client_id}.pkl", "wb") as f:
                    pickle.dump(
                        {
                            "mlp": nn.Sequential(
                                nn.Linear(embedding_dim, hidden_dim),
                                nn.ReLU(),
                                nn.Linear(hidden_dim, hidden_dim),
                                nn.ReLU(),
                                nn.Linear(hidden_dim, hidden_dim),
                                nn.ReLU(),
                            ),
                            # all negative tensor would be outputted sometimes if fc is torch.nn.Linear, which used kaiming init.
                            # so here use U(0,1) init instead.
                            "fc": {
                                name: Linear(hidden_dim, client_num)
                                for name in self.blocks_name
                            },
                        },
                        f,
                    )
        # for tracking the current client's hn parameters
        self.current_client_id: int = None
        self.mlp: nn.Sequential = None
        self.fc_layers: Dict[str, Linear] = {}
        self.retain_blocks: List[str] = []

    # mlp
    def mlp_parameters(self) -> List[nn.Parameter]:
        return list(filter(lambda p: p.requires_grad, self.mlp.parameters()))

    # 全连接层
    def fc_layer_parameters(self) -> List[nn.Parameter]:
        params_list = []
        for block, fc in self.fc_layers.items():
            if block not in self.retain_blocks:
                params_list += list(filter(lambda p: p.requires_grad, fc.parameters()))
        return params_list

    # 嵌入层
    def emd_parameters(self) -> List[nn.Parameter]:
        return list(self.embedding.parameters())

    def forward(self, client_id: int) -> Tuple[Dict[str, torch.Tensor], List[str]]:
        self.current_client_id = client_id
        self.retain_blocks = []
        # 生成embedding
        loaders = get_dataloader("cifar10", client_id, 32, 0.1, 0.3)
        train_set = loaders["train"]
        emd = self.embedding(
            train_set
        )
        self.load_hn()
        feature = self.mlp(emd)
        alpha = {
            block: F.relu(self.fc_layers[block](feature)) for block in self.blocks_name
        }
        default_weight = torch.tensor(
            [i == client_id for i in range(self.client_num)],
            dtype=torch.float,
            device=self.device,
        )

        if self.K > 0:
            blocks_name = []
            self_weights = []
            with torch.no_grad():
                for name, weight in alpha.items():
                    blocks_name.append(name)
                    self_weights.append(weight[client_id])
                _, topk_weights_idx = torch.topk(torch.tensor(self_weights), self.K)
            for i in topk_weights_idx:
                alpha[blocks_name[i]] = default_weight
                self.retain_blocks.append(blocks_name[i])

        return alpha, self.retain_blocks

    # 保存超网络参数
    def save_hn(self):
        for block, param in self.fc_layers.items():
            self.fc_layers[block] = param.cpu()
        with open(self.cache_dir / f"{self.current_client_id}.pkl", "wb") as f:
            pickle.dump(
                {"mlp": self.mlp.cpu(), "fc": self.fc_layers}, f,
            )
        self.mlp = None
        self.fc_layers = {}
        self.current_client_id = None

    def load_hn(self) -> Tuple[nn.Sequential, OrderedDict[str, Linear]]:
        with open(self.cache_dir / f"{self.current_client_id}.pkl", "rb") as f:
            parameters = pickle.load(f)
        self.mlp = parameters["mlp"].to(self.device)
        for block, param in parameters["fc"].items():
            self.fc_layers[block] = param.to(self.device)

    def clean_models(self):
        if os.path.isdir(self.cache_dir):
            os.system(f"rm -rf {self.cache_dir}")


# (input_channels, first fc layer's input features, classes)
ARGS = {
    "cifar10": (3, 8192, 10),
    "cifar100": (3, 8192, 100),
    "emnist": (1, 6272, 62),
    "fmnist": (1, 6272, 10),
}


# NOTE: unknown CNN model structure
# Really don't know the specific structure of CNN model used in pFedLA.
# Structures below are from FedBN's.
# "cifar10", "cifar100"
class CNNWithBatchNorm(nn.Module):
    def __init__(self, dataset):
        super(CNNWithBatchNorm, self).__init__()
        self.block1 = nn.ModuleDict(
            {
                "conv": nn.Conv2d(ARGS[dataset][0], 64, 5, 1, 2),
                "bn": nn.BatchNorm2d(64),
                "relu": nn.ReLU(True),
                "pool": nn.MaxPool2d(2),
            }
        )
        self.block2 = nn.ModuleDict(
            {
                "conv": nn.Conv2d(64, 64, 5, 1, 2),
                "bn": nn.BatchNorm2d(64),
                "relu": nn.ReLU(True),
                "pool": nn.MaxPool2d(2),
            }
        )
        self.block3 = nn.ModuleDict(
            {
                "conv": nn.Conv2d(64, 128, 5, 1, 2),
                "bn": nn.BatchNorm2d(128),
                "relu": nn.ReLU(True),
            }
        )

        self.block4 = nn.ModuleDict(
            {"fc": nn.Linear(ARGS[dataset][1], 2048), "relu": nn.ReLU(True)}
        )
        self.block5 = nn.ModuleDict({"fc": nn.Linear(2048, 512), "relu": nn.ReLU(True)})
        self.block6 = nn.ModuleDict({"fc": nn.Linear(512, ARGS[dataset][2])})

    def forward(self, x):
        x = self.block1["conv"](x)
        x = self.block1["bn"](x)
        x = self.block1["relu"](x)
        x = self.block1["pool"](x)

        x = self.block2["conv"](x)
        x = self.block2["bn"](x)
        x = self.block2["relu"](x)
        x = self.block2["pool"](x)

        x = self.block3["conv"](x)
        x = self.block3["bn"](x)
        x = self.block3["relu"](x)

        x = x.view(x.shape[0], -1)
        x1 = x
        
        x = self.block4["fc"](x)
        x = self.block4["relu"](x)

        x = self.block5["fc"](x)
        x = self.block5["relu"](x)

        x = self.block6["fc"](x)
        return x1, x # x1  是表征

# "cifar10", "cifar100"
class CNNWithoutBatchNorm(nn.Module):
    def __init__(self, dataset):
        super(CNNWithoutBatchNorm, self).__init__()
        self.block1 = nn.ModuleDict(
            {
                "conv": nn.Conv2d(ARGS[dataset][0], 64, 5, 1, 2),
                "relu": nn.ReLU(True),
                "pool": nn.MaxPool2d(2),
            }
        )
        self.block2 = nn.ModuleDict(
            {
                "conv": nn.Conv2d(64, 64, 5, 1, 2),
                "relu": nn.ReLU(True),
                "pool": nn.MaxPool2d(2),
            }
        )
        self.block3 = nn.ModuleDict(
            {"conv": nn.Conv2d(64, 128, 5, 1, 2), "relu": nn.ReLU(True),}
        )

        self.block4 = nn.ModuleDict(
            {"fc": nn.Linear(ARGS[dataset][1], 2048), "relu": nn.ReLU(True)}
        )
        self.block5 = nn.ModuleDict({"fc": nn.Linear(2048, 512), "relu": nn.ReLU(True)})
        self.block6 = nn.ModuleDict({"fc": nn.Linear(512, ARGS[dataset][2])})

    def forward(self, x):
        x = self.block1["conv"](x)
        x = self.block1["relu"](x)
        x = self.block1["pool"](x)

        x = self.block2["conv"](x)
        x = self.block2["relu"](x)
        x = self.block2["pool"](x)

        x = self.block3["conv"](x)
        x = self.block3["relu"](x)

        x = x.view(x.shape[0], -1)
        x1 = x

        x = self.block4["fc"](x)
        x = self.block4["relu"](x)

        x = self.block5["fc"](x)
        x = self.block5["relu"](x)

        x = self.block6["fc"](x)
        return x1, x # x1  是表征

