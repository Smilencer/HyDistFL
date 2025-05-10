from collections import OrderedDict
from typing import OrderedDict
import torch.nn as nn
import copy
import torch
from rich.console import Console
from utils.util import clone_parameters

from .base import ClientBase


class HyPeFLClient(ClientBase):
    def __init__(
        self,
        backbone: torch.nn.Module,
        dataset,
        testset,
        batch_size: int,
        valset_ratio: float,
        testset_ratio: float,
        local_epochs: int,
        local_lr: float,
        logger: Console,
        gpu: int,
    ):
        super(HyPeFLClient, self).__init__(
            backbone,
            dataset,
            testset,
            batch_size,
            valset_ratio,
            testset_ratio,
            local_epochs,
            local_lr,
            logger,
            gpu,
        )
        self.shared_data = None
        self.old_model = None
        self.old_net_pool = None
        self.average_feature = None

    def train(
        self,
        client_id: int,
        model_params: OrderedDict[str, torch.Tensor],
        shared_data: list,
        old_model,
        old_net_pool,
        average_feature,
        verbose=True,
    ):
        self.client_id = client_id

        self.shared_data = shared_data
        self.old_model = old_model
        self.old_net_pool = old_net_pool
        self.average_feature = average_feature.to(self.device)
        
        self.set_parameters(model_params)
        # self.get_client_local_dataset()
        self.model.to(self.device)
        res, stats = self._log_while_training(evaluate=True, verbose=verbose)()
        self.model.cpu()
        return res, stats  # 暂时不返回old_net_pool

    def _train(self):
        self.model.train()
        frz_model_params = clone_parameters(self.model)
        cos = torch.nn.CosineSimilarity(dim=-1)
        # self.logger.log(len(self.old_net_pool))
        # self.logger.log(self.old_net_pool(1))
        tr_set = self.dataset
        for ep in range(self.local_epochs):
            for x, y in tr_set[self.client_id]:
                x, y = x.to(self.device), y.to(self.device)
                _, out = self.model(x)
                loss1 = self.criterion(out, y)
                self.optimizer.zero_grad()
                loss1.backward()
                self.optimizer.step()
            # self.old_model.load_state_dict(self.old_net_pool[self.client_id])
            pre_model = copy.deepcopy(self.model)
            pre_model.load_state_dict(self.old_net_pool[self.client_id])
            # if isinstance(pre_model, nn.Module):
            #     print("pre_model 是一个 PyTorch 模型对象")
            # else:
            #     print("pre_model 不是一个 PyTorch 模型对象")

            for batch_idx, (x, y) in enumerate(self.shared_data):
                x, y = x.to(self.device), y.to(self.device)
                pre_features, _ = pre_model(x)
                cur_features, _ = self.model(x)
                # print(x.shape[0])
                # 使用公共数据集训练当前model，获得Loss1
                _, out = self.model(x)
                loss1 = self.criterion(out, y)
                # ！=batch_size
                if cur_features.shape[0] != 32:
                    temp_feature = self.average_feature[batch_idx][0:cur_features.shape[0]]
                    posi = cos(cur_features, temp_feature)
                else:   
                    posi = cos(cur_features, self.average_feature[batch_idx])
                logits = posi.reshape(-1, 1)

                # 负样本
                nega = cos(cur_features, pre_features)
                logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)

                # temperature
                logits /= 0.5
                labels = torch.zeros(x.shape[0]).cuda().long()

                loss2 = self.criterion(logits, labels)
                loss = loss1 + loss2

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                
                
        delta = OrderedDict(
            {
                k: p1 - p0
                for (k, p1), p0 in zip(
                    self.model.state_dict(keep_vars=True).items(),
                    frz_model_params.values(),
                )
            }
        )
        for key, value in delta.items():
            delta[key] = value.to("cpu")
        return delta

    def test(
        self, client_id: int, model_params: OrderedDict[str, torch.Tensor],
    ):
        self.client_id = client_id
        self.set_parameters(model_params)
        # self.get_client_local_dataset()
        self.model.to(self.device)
        loss, acc = self.evaluate()
        dummy_diff = OrderedDict(
            {
                name: torch.zeros_like(param)
                for name, param in self.model.state_dict().items()
            }
        )

        for key, value in dummy_diff.items():
            dummy_diff[key] = value.to("cpu")
        self.model.cpu()
        stats = {"loss": loss, "acc": acc}
        return dummy_diff, stats
