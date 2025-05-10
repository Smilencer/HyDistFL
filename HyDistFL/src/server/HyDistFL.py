import copy

from path import Path

_CURRENT_DIR = Path(__file__).parent.abspath()
import sys

sys.path.append(_CURRENT_DIR.parent)
sys.path.append(_CURRENT_DIR.parent / "data")


import os
import pickle
import random
from collections import OrderedDict
from typing import List, OrderedDict, Tuple

import torch
from client.HyPeFL import HyPeFLClient
from rich.progress import track
from tqdm import tqdm
from utils.args import get_HyPeFL_args
from utils.models import HyperNetwork
from utils.util import get_dataloader
from base import ServerBase
from makedata import gen_random_loaders

class HyPeFLServer(ServerBase):
    def __init__(self):
        super(HyPeFLServer, self).__init__(get_HyPeFL_args(), "HyPeFL")
        # 生成日志文件
        self.log_name = "{}_{}_{}_{}_{}.html".format(
            self.algo,   # 算法名称
            self.args.dataset,
            self.args.global_epochs,
            self.args.local_epochs,
            self.args.k,
        )
        if self.global_params_dict is not None:
            del self.global_params_dict  # 确保不会使用全局模型参数
        if os.listdir(self.temp_dir) != []:
            # 检查临时目录中是否存在一个名为 "clients_model.pt" 的文件。
            if os.path.exists(self.temp_dir / "clients_model.pt"):
                # 如果存在，它将加载这个文件中保存的客户端模型参数列表。
                self.client_model_params_list = torch.load(
                    self.temp_dir / "clients_model.pt"
                )
                self.logger.log("Find existed clients model...")
        else:
            self.logger.log("Initializing clients model...")
            # 初始化一个客户端模型参数列表。
            self.client_model_params_list = [
                # 使用与数据集相关的骨干网络（self.backbone(self.args.dataset)) 的参数来填充这个列表。列表中的每个元素代表一个客户端的模型参数。
                list(self.backbone(self.args.dataset).state_dict().values())
                for _ in range(self.client_num_in_total)
            ]
    
        # 与数据集相关的骨干网络，用于一些初始化操作。
        _dummy_model = self.backbone(self.args.dataset)
        self.data, self.shared_loader, self.test_loader = gen_random_loaders("cifar100","/root/HyPeFL/datasets/cifar100",10,self.args.batch_size)
        self.dummy_model = _dummy_model
        
        # 创建超网络实例
        self.hypernet = HyperNetwork(
            embedding_dim=self.args.embedding_dim,
            client_num=self.client_num_in_total,
            hidden_dim=self.args.hidden_dim,
            backbone=_dummy_model,
            K=self.args.k,
            gpu=self.args.gpu,
        )

        # 创建客户端实例
        self.trainer = HyPeFLClient(
            backbone=_dummy_model,
            dataset=self.data,
            testset=self.test_loader,
            batch_size=self.args.batch_size,
            valset_ratio=self.args.valset_ratio,
            testset_ratio=self.args.testset_ratio,
            local_epochs=self.args.local_epochs,
            local_lr=self.args.local_lr,
            logger=self.logger,
            gpu=self.args.gpu,
        )

        # 创建了所有模型参数的名称列表和可训练参数的名称列表
        self.all_params_name = [name for name in _dummy_model.state_dict().keys()]
        self.trainable_params_name = [
            name
            for name, param in _dummy_model.state_dict(keep_vars=True).items()
            if param.requires_grad
        ]

    def train(self) -> None:
        self.logger.log("=" * 30, "TRAINING", "=" * 30, style="bold green")
        progress_bar = (
            track(
                range(self.global_epochs),
                "[bold green]Training...",
                console=self.logger,
            )
            if not self.args.log
            else tqdm(range(self.global_epochs), "Training...")
        )

        # +最开始在第0轮的时候初始化旧模型为当前模型
        # +加载上一轮模型结构，结构一样
        old_model = copy.deepcopy(self.dummy_model)
        old_model = old_model.to(self.device)
        # +加载上一轮模型权重
        old_net_pool = [None] * 10
        selected_clients = random.sample(self.client_id_indices, 10)
      
        for client_id in selected_clients:
            # -5
            (
                # 当前客户端的本地模型参数
                client_local_params,
                # 应该保留的模型层
                retain_blocks,
            ) = self.generate_client_model_parameters(client_id)
            print(client_id)
            old_net_pool[client_id] = client_local_params
            # count += 1
            # self.logger.log(count)
            self.tocpu(client_id)

        
        
        # -3
        for E in progress_bar:

            if E % self.args.verbose_gap == 0:
                self.logger.log("=" * 30, f"ROUND: {E}", "=" * 30)

            # 从所有客户端中随机选择一定数量的客户端
            selected_clients = random.sample(
                self.client_id_indices, self.args.client_num_per_round
            )
            
             # +公共数据集
            shared_data = self.shared_loader

            # +获取每个客户端在公共数据集上表征的平均
            shared_features = []
            # print("=======================")
            for client_id in selected_clients:
                # -5
                (
                    # 当前客户端的本地模型参数
                    client_local_params,
                    # 应该保留的模型层
                    retain_blocks,
                ) = self.generate_client_model_parameters(client_id)
                s_model = copy.deepcopy(self.dummy_model)
                s_model.load_state_dict(client_local_params)
                temp = []
                with torch.no_grad():
                    for x, y in shared_data:
                        # print(x.shape[0])
                        features, _ = s_model(x)
                        # 当前客户端的所有表征累加
                        if features.shape[0] == self.args.batch_size:
                            temp.append(features)
                        else:
                            padding_tensor = torch.zeros((self.args.batch_size - features.shape[0], 8192))
                            features = torch.cat((features,padding_tensor), dim=0)
                            temp.append(features)

                temp = torch.stack(temp)
                # temp_tensor = torch.cat(temp, dim=0)
                shared_features.append(temp)
                self.tocpu(client_id)

            # 列表转化为张量
            shared_features = torch.stack(shared_features)
            # 平均表征
            # self.logger.log(average_features.shape)
            average_features = shared_features.mean(dim=0)

            # print("=========")
            # -4
            for client_id in selected_clients:
                # -5
                (
                    # 当前客户端的本地模型参数
                    client_local_params,
                    # 应该保留的模型层
                    retain_blocks,
                ) = self.generate_client_model_parameters(client_id)
                # self.logger.log('循环成功执行了')
                # -6
                # 客户端本地训练
                diff, stats = self.trainer.train(
                    client_id=client_id,
                    model_params=client_local_params,
                    # +加入公共数据集
                    shared_data=shared_data,
                    old_model=old_model,
                    old_net_pool=old_net_pool,
                    average_feature=average_features,
                    verbose=(E % self.args.verbose_gap) == 0,
                )
                self.all_clients_stats[client_id][f"ROUND: {E}"] = (
                    f"retain {retain_blocks}, {stats['loss_before']:.4f} -> {stats['loss_after']:.4f}",
                )

                # 更新超网络的参数
                # -8
                self.update_hypernetwork(client_id, diff, retain_blocks)

                # 更新客户端模型的参数
                # -7
                self.update_client_model_parameters(client_id, diff)
                # +更新
                
                layer_params_dict = dict(
                    # 模型的所有参数名称列表             # 每个元素是一个客户端的模型参数列表
                    zip(self.all_params_name, self.client_model_params_list[client_id])
                )
                old_net_pool[client_id] = layer_params_dict
                
            if E % self.args.save_period == 0:
                torch.save(
                    # 包含各个客户端模型参数的列表
                    self.client_model_params_list, self.temp_dir / "clients_model.pt",
                )
                with open(self.temp_dir / "epoch.pkl", "wb") as f:
                    pickle.dump(E, f)
        # 打印轮次信息
        self.logger.log(self.all_clients_stats)

    def test(self) -> None:
        self.logger.log("=" * 30, "TESTING", "=" * 30, style="bold blue")
        all_loss = []
        all_acc = []
        for client_id in track(
            self.client_id_indices,
            "[bold blue]Testing...",
            console=self.logger,
            disable=self.args.log,
        ):
            client_local_params, retain_blocks = self.generate_client_model_parameters(
                client_id
            )
            dummy_diff, stats = self.trainer.test(
                client_id=client_id, model_params=client_local_params,
            )

            # NOTE: make sure that all client model params are on CPU, not CUDA
            # or self.generate_...() would raise the error of stacking tensors on different devices
            self.update_client_model_parameters(client_id, dummy_diff)
            self.logger.log(
                f"client [{client_id}] retain {retain_blocks}, [red]loss: {stats['loss']:.4f}    [magenta]accuracy: {stats['acc']:.2f}%"
            )
            all_loss.append(stats["loss"])
            all_acc.append(stats["acc"])

        self.logger.log("=" * 20, "RESULTS", "=" * 20, style="bold green")
        self.logger.log(
            "loss: {:.4f}    accuracy: {:.2f}%".format(
                sum(all_loss) / len(all_loss), sum(all_acc) / len(all_acc),
            )
        )

    # 更新客户端模型的参数
    @torch.no_grad()
    def update_client_model_parameters(
                              # 模型参数的更新量
        self, client_id: int, delta: OrderedDict[str, torch.Tensor],
    ) -> None:
        # 存储更新后的模型参数
        updated_params = []
        # 循环遍历客户端模型的参数列表，其中包括客户端模型的所有参数。对于每个参数和其对应的更新量（从 delta 中获得）：
        for param, diff in zip(
            self.client_model_params_list[client_id], delta.values()
        ):
            # 原始模型参数与更新量相加
            updated_params.append((param + diff).detach().cpu())
        # 将 updated_params 列表中的更新后的参数替代客户端模型参数列表中原始的模型参数。
        self.client_model_params_list[client_id] = updated_params

    # 生成特定客户端的本地模型参数以及应保留的块（模型层）
    def generate_client_model_parameters(
        self, client_id: int
    ) -> Tuple[OrderedDict[str, torch.Tensor], List[str]]: # 返回一个元组类型，包含了一个字典和一个字符串列表
        # 将每个模型参数的名称与来自所有客户端的对应参数值列表（以元组的形式）进行关联，形成一个字典。
        layer_params_dict = dict(
              # 模型的所有参数名称列表             # 每个元素是一个客户端的模型参数列表
            zip(self.all_params_name, list(zip(*self.client_model_params_list)))
        )
        alpha, retain_blocks = self.hypernet(client_id)
        # 存储聚合后的模型参数
        aggregated_parameters = {}
        # 设置每个客户端的权重
        # 创建一个与总客户端数量相同长度的张量，其中对应当前客户端的位置为 1，其余位置为 0。
        default_weight = torch.tensor(
            [i == client_id for i in range(self.client_num_in_total)],
            dtype=torch.float,
            # device=self.device,
        )
        # 循环迭代所有模型参数的名称
        for name in self.all_params_name:
            if name in self.trainable_params_name:
                a = alpha[name.split(".")[0]].to('cpu')
            else:
                a = default_weight
            if a.sum() == 0:
                self.logger.log(self.all_clients_stats)
                raise RuntimeError(
                    f"client [{client_id}]'s {name.split('.')[0]} alpha is a all 0 vector"
                )
            # self.logger.log(a.device)
            # 当前参数的每个客户端的参数按照权重加权平均并求和，得到聚合后的参数值。
            aggregated_parameters[name] = torch.sum(
                a
                / a.sum()
                * torch.stack(layer_params_dict[name], dim=-1),
                # * torch.stack(layer_params_dict[name], dim=-1).to(self.device),
                dim=-1,
            )

        # 更新客户端模型参数列表中当前客户端的模型参数。
        self.client_model_params_list[client_id] = list(aggregated_parameters.values())
        return aggregated_parameters, retain_blocks
    
    # 更新超网络的参数
    def update_hypernetwork(
        self,
        client_id: int,
        diff: OrderedDict[str, torch.Tensor],     # 模型参数的差异
        retain_blocks: List[str] = [],            # 应该保留的模型块（层）
    ) -> None:
        # 计算梯度
        # 计算一组输出相对于一组输入的梯度
        hn_grads = torch.autograd.grad(
            # 需要梯度的参数
            outputs=list(
                filter(
                    lambda param: param.requires_grad,
                    self.client_model_params_list[client_id],
                )
            ),

            # 输入列表
            inputs=self.hypernet.mlp_parameters()
            + self.hypernet.fc_layer_parameters()
            + self.hypernet.emd_parameters(),

            # 控制梯度传播的参数
            grad_outputs=list(
                map(
                    lambda tup: tup[1],
                    filter(
                        lambda tup: tup[1].requires_grad
                        and tup[0].split(".")[0] not in retain_blocks,
                        diff.items(),
                    ),
                )
            ),
            allow_unused=True,
        )
        # 根据计算得到的超网络参数的梯度，分别更新超网络中不同部分（全连接层、MLP、嵌入层）的参数，并将更新后的参数保存起来。
        mlp_grads = hn_grads[: len(self.hypernet.mlp_parameters())]
        fc_grads = hn_grads[
            len(self.hypernet.mlp_parameters()) : len(
                self.hypernet.mlp_parameters() + self.hypernet.fc_layer_parameters()
            )
        ]
        emd_grads = hn_grads[
            len(self.hypernet.mlp_parameters() + self.hypernet.fc_layer_parameters()) :
        ]

        for param, grad in zip(self.hypernet.fc_layer_parameters(), fc_grads):
            if grad is not None:
                param.data -= self.args.hn_lr * grad

        for param, grad in zip(self.hypernet.mlp_parameters(), mlp_grads):
            param.data -= self.args.hn_lr * grad

        for param, grad in zip(self.hypernet.emd_parameters(), emd_grads):
            param.data -= self.args.hn_lr * grad

        self.hypernet.save_hn()

    # + 移去cpu
    @torch.no_grad()
    def tocpu(
            # 模型参数的更新量
            self, client_id: int,
    ) -> None:
        # 存储更新后的模型参数
        updated_params = []
        # 循环遍历客户端模型的参数列表，其中包括客户端模型的所有参数。对于每个参数和其对应的更新量（从 delta 中获得）：
        for param in (self.client_model_params_list[client_id]):
            # 原始模型参数与更新量相加
            updated_params.append(param.detach().cpu())
        # 将 updated_params 列表中的更新后的参数替代客户端模型参数列表中原始的模型参数。
        self.client_model_params_list[client_id] = updated_params
    
    # +生成公共数据
    def get_shared_data(self):
        share_data = []
        for i in range(10):
            dataset_sub = get_dataloader(
                dataset=self.args.dataset,
                client_id=i,
                batch_size=self.args.batch_size,
                valset_ratio=self.args.valset_ratio,
                testset_ratio=self.args.testset_ratio,
            )
            share_data.extend(dataset_sub['val'])
        # 返回dataloader，迭代取值
        return share_data


    def run(self):
        super().run()
        # clean out all HNs
        self.hypernet.clean_models()


if __name__ == "__main__":
    server = HyPeFLServer()
    server.run()
