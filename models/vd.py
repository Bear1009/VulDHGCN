import math
import pandas as pd
from torch import nn
from omegaconf import DictConfig
import torch
from src.datas.samples import XFGBatch
from typing import Dict
from pytorch_lightning import LightningModule
from src.models.modules.gnns import GraphConvEncoder, GatedGraphConvEncoder, HGCN
from torch.optim import Adam, SGD, Adamax, RMSprop
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
import torch.nn.functional as F
from src.metrics import Statistic
from torch_geometric.data import Batch
from src.vocabulary import Vocabulary
from configs.config import parser


class DeepWuKong(LightningModule):
    r"""vulnerability detection model to detect vulnerability

    Args:
        config (DictConfig): configuration for the model
        vocabulary_size (int): the size of vacabulary
        pad_idx (int): the index of padding token
    """
    # 私有变量 字典类型；
    # 使用方法如，如果想要获取 "RMSprop" 优化器的类，可以使用 _optimizers["RMSprop"] 来获取 RMSprop 类。
    _optimizers = {
        "RMSprop": RMSprop,
        "Adam": Adam,
        "SGD": SGD,
        "Adamax": Adamax
    }

    _encoders = {
        "gcn": GraphConvEncoder,
        "ggnn": GatedGraphConvEncoder,
        "hgcn": HGCN
    }

    """
    # 融合策略
    def merge_tensors(self, tensor1, tensor2, merge_type='direct'):
        # 确认两个张量维度相同
        if tensor1.shape != tensor2.shape:
            raise ValueError("两个张量的维度不相同，无法进行融合。")

        # 直接合并
        if merge_type == 'direct':
            return tensor1 + tensor2
            # return torch.cat((tensor1, tensor2), dim=0)

        # 平均值合并
        elif merge_type == 'average':
            return (tensor1 + tensor2) / 2

        # 加权合并
        elif merge_type == 'weighted':
            weight1, weight2 = 1, 1  # 权重值，可以根据实际情况修改
            return weight1 * tensor1 + weight2 * tensor2

        # 选择最大值合并
        elif merge_type == 'max':
            return torch.max(tensor1, tensor2)

        # 选择最小值合并
        elif merge_type == 'min':
            return torch.min(tensor1, tensor2)

        else:
            raise ValueError("融合策略不正确，请选择正确的融合策略。")
    
    """


    # config：DictConfig，指定参数类型为DictConfig。
    def __init__(self, config: DictConfig, vocab: Vocabulary, vocabulary_size: int,
                 pad_idx: int):
        super().__init__()  # 调用父类的初始化构造函数
        self.save_hyperparameters()  # 主要功能是将模型的参数保存到 hparams 属性中。
        self.__config = config  # 属性名前面的双下划线 __ 表示这是一个私有属性，意味着在类的外部不应直接访问该属性。

        # 初始化权重参数，这里的值可以根据需要进行调整
        self.weight1 = nn.Parameter(torch.tensor(0.5))
        self.weight2 = nn.Parameter(torch.tensor(0.5))

        hidden_size = config.classifier.hidden_size  # hidden_size = 512
        '''
            这行代码创建了一个名为__graph_encoder的类成员变量，
            并将其赋值为self._encoders[config.gnn.name](config.gnn, vocab, vocabulary_size, pad_idx)。
            在这段代码中，config.gnn.name是一个配置对象中的属性，用于获取GNN（图神经网络）的名称。
            然后，通过self._encoders字典来选择正确的编码器类，
            并通过传递config.gnn、vocab、vocabulary_size和pad_idx参数来实例化该编码器类。
        '''
        #  self.__graph_encoder = gcn
        #  self.__graph_encoder = self._encoders[config.gnn.name](config.gnn, vocab, vocabulary_size, pad_idx)

        args = parser.parse_args()
        # args.feat_dim =  16 手动添加输入节点特征维度16
        args.feat_dim = 256
        print("args中的内容：", args)
        # args中的内容： Namespace(act='relu', alpha=0.2, bias=1, c=1.0, cuda=-1, dataset='cora', dim=128,
        # double_precision='0', dropout=0.0, epochs=5000, eval_freq=1, feat_dim=16, gamma=0.5, grad_clip=None,
        # local_agg=0, log_freq=1, lr=0.01, lr_reduce_freq=None, manifold='Euclidean', min_epochs=100, model='GCN',
        # momentum=0.999, n_heads=4, normalize_adj=1, normalize_feats=1, num_layers=2, optimizer='Adam',
        # patience=100, pos_weight=0, pretrained_embeddings=None, print_epoch=True, r=2.0, save=0, save_dir=None,
        # seed=1234, split_seed=1234, sweep_c=0, t=1.0, task='nc', test_prop=0.1, use_att=0, use_feats=1,
        # val_prop=0.05, weight_decay=0.0)
        # self.__graph_encoder_ggnn = self._encoders[config.gnn.name_ggnn](config.gnn, vocab, vocabulary_size, pad_idx)
        self.__graph_encoder_gcn = self._encoders[config.gnn.name_e](config.gnn, vocab, vocabulary_size, pad_idx)
        # self.__graph_encoder = hgcn
        self.__graph_encoder_hgcn = self._encoders[config.gnn.name_h](config.gnn, vocab, vocabulary_size, pad_idx, config.gnn.c, args)

        # hidden layers
        # 每个隐藏层包含一个线性层、一个激活函数层和一个 Dropout 层
        layers = [
            nn.Linear(config.gnn.hidden_size, hidden_size),  # [256,512] 线性层用于进行线性变换（输入特征维度256，输出特征维度512）
            nn.ReLU(),  # 激活函数ReLU()，将输入的负值置为零。
            nn.Dropout(config.classifier.drop_out)  # 用于在训练过程中随机地将一部分神经元的输出置为零，以减少模型的过拟合。它接收一个参数，即丢弃的比例 = 0.5
        ]
        # 这段代码的作用是检查隐藏层的数量是否小于1，并在条件不满足时引发异常，创建一个 ValueError 异常对象，并将一个描述无效层数的错误消息与之关联。
        if config.classifier.n_hidden_layers < 1:
            raise ValueError(
                f"Invalid layers number ({config.classifier.n_hidden_layers})")
        # 设置多层隐藏层，这里层数=1
        for _ in range(config.classifier.n_hidden_layers - 1):
            layers += [
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(config.classifier.drop_out)
            ]
        # 通过nn.Sequential(*layers)将隐藏层的序列化列表转换为一个nn.Sequential对象，并将其赋值给self.__hidden_layers属性 [256x512]
        self.__hidden_layers = nn.Sequential(*layers)
        # 构建输出层，使用nn.Linear(hidden_size, config.classifier.n_classes)创建一个全连接层，将隐藏层的输出映射到类别数量
        self.__classifier = nn.Linear(hidden_size, config.classifier.n_classes)  # (512, 2)

    # 融合策略
    def merge_tensors(self, tensor1, tensor2, merge_type='direct'):
        # 确认两个张量维度相同
        if tensor1.shape != tensor2.shape:
            raise ValueError("两个张量的维度不相同，无法进行融合。")

        # 直接合并
        if merge_type == 'direct':
            return tensor1 + tensor2
            # return torch.cat((tensor1, tensor2), dim=0)

        # 平均值合并
        elif merge_type == 'average':
            return (tensor1 + tensor2) / 2

        # 加权合并
        elif merge_type == 'weighted':
            # weight1, weight2 = 1, 1    权重值，可以根据实际情况修改
            return self.weight1 * tensor1 + self.weight2 * tensor2

        # 选择最大值合并
        elif merge_type == 'max':
            return torch.max(tensor1, tensor2)

        # 选择最小值合并
        elif merge_type == 'min':
            return torch.min(tensor1, tensor2)

        else:
            raise ValueError("融合策略不正确，请选择正确的融合策略。")

    # 定义了一个前向传播方法 forward，用于对输入的批次数据进行分类。
    def forward(self, batch: Batch) -> torch.Tensor:
        """

        Args:
            batch (Batch): [n_XFG (Data)]

        Returns: classifier results: [n_method; n_classes]
        """
        # print("batch在这里：" + str(batch))
        # batch在这里：Batch(x=[1282, 16], edge_index=[2, 9283], batch=[1282], ptr=[65])
        print("batch在这里：", batch)
        # [n_XFG, hidden size] 64x512
        # 将输入的批次数据传入图编码器模型中，得到图的隐藏表示 graph_hid，维度为 [n_XFG, hidden size]。
        # graph_hid = self.__graph_encoder(batch)  # GCN
        # graph_hid_ggnn = self.__graph_encoder_ggnn(batch)
        graph_hid_gcn = self.__graph_encoder_gcn(batch)
        graph_hid_hgcn = self.__graph_encoder_hgcn.encode(batch)  # HGCN [64, 256]
        # 五种融合策略：'direct' 'average' 'weighted' 'max' 'min'
        merge_type = 'direct'
        graph_hid = self.merge_tensors(graph_hid_gcn, graph_hid_hgcn, merge_type)
        # graph_hid = graph_hid_hgcn
        # graph_hid的形状： torch.Size([2, 128])  应为[1282, 128]
        print("graph_hid的内容：", graph_hid)
        print("graph_hid的形状：", graph_hid.shape)

        # 将 graph_hid 作为输入传递给 self.__hidden_layers，它是一个包含隐藏层的 nn.Sequential 对象，
        # 用于对隐藏表示进行一系列的线性变换、激活函数和 dropout 操作。这样可以通过隐藏层的非线性变换来提取更高级的特征表示。
        hiddens = self.__hidden_layers(graph_hid)  # [输入256,输出512]
        print("hideens的形状：", hiddens.shape)  # [2, 512]  应为[1282, 512]  GCN[64, 512]
        # [n_XFG; n_classes]
        # 将隐藏层的输出 hiddens 作为输入传递给 self.__classifier，它是一个全连接层，将隐藏表示映射到分类结果的维度。
        # 最终，该方法返回分类结果，维度为 [n_XFG, n_classes]。
        print("***********weight1的值**********:", self.weight1)
        print("***********weight2的值**********:", self.weight2)
        return self.__classifier(hiddens)  # [1282, 2]  GCN[64, 2]

    # 获取优化器
    def _get_optimizer(self, name: str) -> torch.nn.Module:
        if name in self._optimizers:
            return self._optimizers[name]
        raise KeyError(f"Optimizer {name} is not supported")

    '''    # 余弦退火学习率
    def my_get_cosine_schedule_with_warmup_lr_lambda(
            current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float
    ):
        num_warmup_steps = min(num_warmup_steps, int(0.1 * num_training_steps))
        # 如果当前步骤小于预热步数（num_warmup_steps），则学习率会线性增加，直到达到预热步数。
        if current_step < num_warmup_steps:
            # 一旦超过预热步数，学习率会根据余弦函数进行退火，这是一种常用的学习率调度策略，可以帮助模型在训练后期更加稳定地收敛。
            return float(current_step) / float(max(1, num_warmup_steps))
        # progress 表示训练的进度，它是当前步骤减去预热步数后的步骤数除以总训练步数减去预热步数。
        # 余弦函数部分会使学习率在训练后期缓慢下降，从而有助于模型更好地拟合数据。
        # num_cycles 控制余弦函数的周期数，它影响了学习率曲线的震荡情况。
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        # 最终返回的学习率因子是余弦函数的值，取在 0.1 到 1 之间。
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

        current_step = 1000
        num_warmup_steps = 100
        num_training_steps = 10000
        num_cycles = 0.5

        lr = my_get_cosine_schedule_with_warmup_lr_lambda(current_step,
                                                          num_warmup_steps=num_warmup_steps,
                                                          num_training_steps=num_training_steps,
                                                          num_cycles=num_cycles)
        '''

    # 配置优化器（optimizer）和学习率调度器（lr_scheduler）
    def configure_optimizers(self) -> Dict:
        # 创建参数列表
        parameters = [self.parameters()]
        # 通过 _get_optimizer 方法根据配置中指定的优化器名称创建了一个优化器对象。
        # 该方法根据传入的优化器名称选择相应的优化器类，并传入参数列表和学习率（从配置中获取）来初始化优化器对象。
        optimizer = self._get_optimizer(
            self.__config.hyper_parameters.optimizer)(
            [{
                "params": p
            } for p in parameters],
            self.__config.hyper_parameters.learning_rate)

        # 通过 torch.optim.lr_scheduler.LambdaLR 创建了一个学习率调度器对象。
        '''
        通过将学习率调整函数传递给 LambdaLR，调度器将根据每个训练轮次的衰减因子自动调整优化器的学习率。
        随着训练轮次的增加，学习率会以指数方式衰减，从而在训练过程中逐渐降低模型的学习率。
        这种学习率调度策略可以帮助模型在训练过程中逐步减小学习率，从而更好地适应数据的变化和优化模型的性能。
        '''
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            # 使用学习率调整函数le_lambda指数衰减函数来调整学习率
            lr_lambda=lambda epoch: self.__config.hyper_parameters.decay_gamma
                                    ** epoch)
        # 将优化器和学习率调度器以字典的形式返回，用于在训练过程中进行优化器的更新和学习率的调度。
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    # 记录训练步骤（training step）结果的方法，方便后续分析和监控模型的训练过程。
    def _log_training_step(self, results: Dict):
        # 通过调用 self.log_dict 方法，将结果字典中的数据记录到训练日志中。
        # 通过设置 on_step=True，表示将这些数据记录到当前训练步骤的日志中，而不是整个训练轮次的日志。
        # 而 on_epoch=False 表示这些数据不会被记录到整个训练轮次的日志中。
        self.log_dict(results, on_step=True, on_epoch=False)

    # 定义训练步骤的方法
    # 该方法接受两个参数，batch 是一个 XFGBatch 类型的对象，包含了训练所需的输入数据，batch_idx 表示当前训练批次的索引。
    def training_step(self, batch: XFGBatch,
                      batch_idx: int) -> torch.Tensor:  # type: ignore
        # [n_XFG; n_classes]
        # 获取模型的输出，即模型对输入数据的预测结果。
        logits = self(batch.graphs)
        # 使用交叉熵损失函数 F.cross_entropy 计算预测结果与真实标签之间的损失。
        loss = F.cross_entropy(logits, batch.labels)

        # 用于存储训练步骤的结果信息。
        result: Dict = {"train_loss": loss}
        # 使用 torch.no_grad() 上下文管理器对后续计算进行禁用梯度计算，以避免对模型参数进行更新。
        with torch.no_grad():
            # 计算预测结果的准确性
            _, preds = logits.max(dim=1)
            statistic = Statistic().calculate_statistic(
                batch.labels,
                preds,
                2,
            )
            #  计算预测结果其他统计指标，并将这些指标添加到 result 字典中。
            batch_metric = statistic.calculate_metrics(group="train")
            result.update(batch_metric)
            # 通过调用 self._log_training_step(result) 将训练步骤的结果记录到训练日志中。
            self._log_training_step(result)
            # 通过调用 self.log 方法将训练集的 F1 分数记录到训练日志中，并在进度条中显示。
            self.log("F1",
                     batch_metric["train_f1"],
                     prog_bar=True,
                     logger=False)
        # 最后，将损失值和统计信息作为字典返回，其中包括键 "loss" 和 "statistic"。这些返回值将在训练过程中用于优化器的反向传播和更新模型参数。
        return {"loss": loss, "statistic": statistic}

    # 这段代码主要完成了对验证集的一个批次数据的处理，包括了模型的预测、损失的计算以及统计信息的生成。这些结果将在训练过程中用于监控模型的性能。
    def validation_step(self, batch: XFGBatch,
                        batch_idx: int) -> torch.Tensor:  # type: ignore
        # [n_XFG; n_classes]
        # 获取模型的输出，即模型对输入数据的预测结果。这一行调用了模型(self)，将输入的图数据 batch.graphs 传递给模型，得到模型的输出 logits，它是对输入数据的预测结果。
        # print("batch.graphs的内容是：", batch.graphs)
        logits = self(batch.graphs)  # 应该是[64, 2]  GCN[64, 2]
        print("logits中的内容是：", logits)
        print("logits的形状是：", logits.shape)
        # 使用交叉熵损失函数 F.cross_entropy 计算预测结果与真实标签之间的损失。
        print("batch_lables的内容：", batch.labels)
        print("batch_lables的形状：", batch.labels.shape)
        loss = F.cross_entropy(logits, batch.labels)
        # 创建了一个字典 result，其中存储了验证步骤的结果信息，初始化了一个键值对 val_loss，对应的值是上一步计算得到的损失 loss。
        result: Dict = {"val_loss": loss}
        # 在验证阶段，不进行梯度计算和参数更新，因此使用了torch.no_grad()上下文管理器，以确保在计算指标时不会保存梯度信息。
        with torch.no_grad():
            # 在验证阶段，从预测结果 logits 中选择具有最大值的类别作为模型的最终预测。
            _, preds = logits.max(dim=1)
            # 使用 Statistic 类的 calculate_statistic 方法计算了一些统计信息，这些信息可能包括精确度、召回率等。
            statistic = Statistic().calculate_statistic(
                batch.labels,
                preds,
                2,
            )
            # 根据计算得到的统计信息，调用 calculate_metrics 方法生成一个用于批次的指标。
            batch_metric = statistic.calculate_metrics(group="val")
            # 将批次的指标更新到之前创建的 result 字典中。
            result.update(batch_metric)
        # 将损失值和统计信息作为字典返回，其中包括键 "loss" 和 "statistic"。这些返回值可以用于计算验证集的平均损失值和指标，并进行进一步的性能评估和比较。
        return {"loss": loss, "statistic": statistic}

    def test_step(self, batch: XFGBatch,
                  batch_idx: int) -> torch.Tensor:  # type: ignore
        # [n_XFG; n_classes]
        # 获取模型的输出，即模型对输入数据的预测结果。
        logits = self(batch.graphs)
        # 使用交叉熵损失函数 F.cross_entropy 计算预测结果与真实标签之间的损失。
        loss = F.cross_entropy(logits, batch.labels)
        # 用于存储测试步骤的结果信息。
        result: Dict = {"test_loss", loss}
        # 不进行梯度计算和参数更新
        with torch.no_grad():
            _, preds = logits.max(dim=1)
            statistic = Statistic().calculate_statistic(
                batch.labels,
                preds,
                2,
            )
            batch_metric = statistic.calculate_metrics(group="test")
            result.update(batch_metric)
        # result.to("cpu")
        # pd.DataFrame({"loss": loss, "statistic": statistic}).to_csv('results.csv')
        return {"loss": loss, "statistic": statistic}

    # ========== EPOCH END ==========
    # 该 _prepare_epoch_end_log 方法用于在每个 epoch 结束时准备用于日志记录的信息。接收一个包含每个步骤输出的列表 step_outputs。
    def _prepare_epoch_end_log(self, step_outputs: EPOCH_OUTPUT,
                               step: str) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            losses = [
                so if isinstance(so, torch.Tensor) else so["loss"]
                for so in step_outputs
            ]
            mean_loss = torch.stack(losses).mean()
        # 返回一个包含平均损失值的字典。
        return {f"{step}_loss": mean_loss}

    # 用于在每个 epoch 结束时对训练和验证步骤的输出进行处理并生成日志。
    '''
    方法接受两个参数：
        step_outputs：包含每个步骤输出的列表。
        group：表示输出所属的分组，可以是 "train" 或 "val"。
    '''

    def _shared_epoch_end(self, step_outputs: EPOCH_OUTPUT, group: str):
        # 方法首先调用 _prepare_epoch_end_log 方法，将步骤输出列表和分组作为参数传入，生成一个日志字典 log。
        # 该日志字典包含了平均损失值的键值对，键为 <group>_loss。
        log = self._prepare_epoch_end_log(step_outputs, group)
        # 方法从步骤输出中提取统计信息，并调用 Statistic 类的 calculate_metrics 方法计算指定分组的指标。将这些指标添加到日志字典 log 中。
        statistic = Statistic.union_statistics(
            [out["statistic"] for out in step_outputs])
        log.update(statistic.calculate_metrics(group))
        # 方法使用 self.log_dict 方法将日志字典 log 中的键值对记录到训练日志中，
        # 设置 on_step=False 表示在每个 epoch 结束时记录，on_epoch=True 表示记录到当前 epoch 的日志中。
        self.log_dict(log, on_step=False, on_epoch=True)

    # 方法接受一个参数 training_step_output，它是一个包含训练步骤输出的列表。该列表中的每个元素包含训练步骤的损失和统计信息。
    def training_epoch_end(self, training_step_output: EPOCH_OUTPUT):
        # 这个方法将处理训练步骤输出并生成相应的日志。
        self._shared_epoch_end(training_step_output, "train")

    def validation_epoch_end(self, validation_step_output: EPOCH_OUTPUT):
        self._shared_epoch_end(validation_step_output, "val")

    def test_epoch_end(self, test_step_output: EPOCH_OUTPUT):
        self._shared_epoch_end(test_step_output, "test")
