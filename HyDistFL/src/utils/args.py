from argparse import ArgumentParser, Namespace


def get_HyPeFL_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--global_epochs", type=int, default=10)     # 全局训练的轮数
    parser.add_argument("--local_epochs", type=int, default=10)      # 每个客户端在一轮全局训练中进行本地训练的轮数
    parser.add_argument("--local_lr", type=float, default=1e-2)      # 每个客户端在本地训练中使用的学习率
    parser.add_argument("--hn_lr", type=float, default=5e-3)         # 超网络的学习率
    parser.add_argument("--verbose_gap", type=int, default=1)        # 在训练过程中每隔多少轮输出一次详细信息
    parser.add_argument("--embedding_dim", type=int, default=100)    # 超网络的嵌入维度
    parser.add_argument("--hidden_dim", type=int, default=100)       # 超网络的隐藏维度
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cifar10", "cifar100", "emnist", "fmnist", "ad"],
        default="cifar10",
    )
    parser.add_argument("--batch_size", type=int, default=32)        # 每个训练批次的大小

    parser.add_argument("--valset_ratio", type=float, default=0.1)   # 验证集的数据比例，每个客户端累加后为公共数据集
    parser.add_argument("--testset_ratio", type=float, default=0.3)  # 测试集的数据比例

    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--log", type=int, default=1)                # 是否进行日志记录
    parser.add_argument("--seed", type=int, default=17)              # 设置随机数生成器的种子
    parser.add_argument("--client_num_per_round", type=int, default=10)       # 每轮全局训练中随机选择的参与客户端数量
    parser.add_argument("--save_period", type=int, default=20)       # 多少轮保存一次模型参数
    return parser.parse_args()


def get_FedAvg_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--global_epochs", type=int, default=10)
    parser.add_argument("--local_epochs", type=int, default=10)
    parser.add_argument("--local_lr", type=float, default=1e-2)
    parser.add_argument("--verbose_gap", type=int, default=20)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cifar10", "cifar100", "emnist", "fmnist"],
        default="cifar10",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--valset_ratio", type=float, default=0.0)
    parser.add_argument("--testset_ratio", type=float, default=0.3)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--log", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--client_num_per_round", type=int, default=10)
    parser.add_argument("--save_period", type=int, default=20)
    return parser.parse_args()