from pytorch_lightning import LightningDataModule
from omegaconf import DictConfig
from os import cpu_count
from os.path import join

import torch

from src.datas.samples import XFGBatch, XFGSample
from src.datas.datasets import XFGDataset
from typing import List, Optional
from torch.utils.data import DataLoader, Dataset
from src.vocabulary import Vocabulary


class XFGDataModule(LightningDataModule):
    def __init__(self, config: DictConfig, vocab: Vocabulary):
        super().__init__()
        self.__vocab = vocab
        self.__config = config
        self.__data_folder = join(config.data_folder, config.dataset.name)
        self.__n_workers = cpu_count(
        ) if self.__config.num_workers == -1 else self.__config.num_workers

    # 该方法接受一个批量的 XFGSample 对象列表作为输入，然后将其作为参数传递给 XFGBatch 类的构造函数，创建一个 XFGBatch 对象作为输出。
    @staticmethod
    def collate_wrapper(batch: List[XFGSample]) -> XFGBatch:
        return XFGBatch(batch)

    # __create_dataset方法是一个私有方法，负责创建一个数据集对象。
    # 它接受一个data_path参数，该参数是数据的路径，并返回一个XFGDataset类的实例。
    def __create_dataset(self, data_path: str) -> Dataset:
        return XFGDataset(data_path, self.__config, self.__vocab)

    # train_dataloader 方法返回一个用于训练的 DataLoader 对象。它加载训练数据集并返回以批次形式提供数据的迭代器。
    def train_dataloader(self) -> DataLoader:
        train_dataset_path = join(self.__data_folder, "train.json")
        train_dataset = self.__create_dataset(train_dataset_path)
        # 使用`DataLoader`类创建一个数据加载器对象。
        '''
             `train_dataset`：要加载的训练数据集。
             `batch_size`：批次大小，指定每个批次中样本的数量。
             `shuffle`：是否在每个训练周期之前对数据进行洗牌。
             `num_workers`：用于数据加载的并行工作进程数。
             `collate_fn`：用于批次数据的处理函数，这里使用了`collate_wrapper`方法来封装批次的创建。
             `pin_memory`：是否将数据存储在固定内存中，用于加速数据传输。
        '''
        return DataLoader(
            train_dataset,
            batch_size=self.__config.hyper_parameters.batch_size,
            shuffle=self.__config.hyper_parameters.shuffle_data,
            num_workers=self.__n_workers,
            collate_fn=self.collate_wrapper,
            pin_memory=True,
        )

    # `val_dataloader`方法返回一个用于验证的`DataLoader`对象。它加载验证数据集并返回以批次形式提供数据的迭代器。
    def val_dataloader(self) -> DataLoader:
        val_dataset_path = join(self.__data_folder, "val.json")
        val_dataset = self.__create_dataset(val_dataset_path)
        '''
            - `val_dataset`：要加载的验证数据集。
            - `batch_size`：批次大小，指定每个批次中样本的数量。
            - `shuffle`：是否在每个训练周期之前对数据进行洗牌。在验证过程中，通常不需要对数据进行洗牌。
            - `num_workers`：用于数据加载的并行工作进程数。
            - `collate_fn`：用于批次数据的处理函数，这里使用了`collate_wrapper`方法来封装批次的创建。
            - `pin_memory`：是否将数据存储在固定内存中，用于加速数据传输。
        '''
        return DataLoader(
            val_dataset,
            batch_size=self.__config.hyper_parameters.batch_size,
            shuffle=False,
            num_workers=self.__n_workers,
            collate_fn=self.collate_wrapper,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        # 构建了测试数据集的路径`test_dataset_path`
        test_dataset_path = join(self.__data_folder, "test.json")
        # 然后调用`__create_dataset`方法创建了测试数据集的实例`test_dataset`。
        test_dataset = self.__create_dataset(test_dataset_path)
        '''
            - `test_dataset`：要加载的测试数据集。
            - `batch_size`：批处理大小，指定每个批次中的样本数量。
            - `shuffle`：是否对数据进行洗牌。在测试阶段，通常不需要洗牌。
            - `num_workers`：用于数据加载的并行工作进程数。
            - `collate_fn`：用于处理批次数据的函数，这里使用`collate_wrapper`方法进行封装。
            - `pin_memory`：是否将数据存储在固定内存中，以加速数据传输。
        '''
        return DataLoader(
            test_dataset,
            batch_size=self.__config.hyper_parameters.batch_size,
            shuffle=False,
            num_workers=self.__n_workers,
            collate_fn=self.collate_wrapper,
            pin_memory=True,
        )

    def transfer_batch_to_device(self, batch: XFGBatch, device: Optional[torch.device] = None) -> XFGBatch:
        # 如果提供了目标设备，则调用move_to_device方法
        if device is not None:
            batch.move_to_device(device)
        return batch
