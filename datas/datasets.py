from torch.utils.data import Dataset
from omegaconf import DictConfig
from src.datas.graphs import XFG
from src.datas.samples import XFGSample
from os.path import exists
import json
from src.vocabulary import Vocabulary

# 这是一个名为XFGDataset的自定义数据集类，用于加载XFG（XML-based Function Graph）数据集。
'''
    XFGDataset类将XFG数据集加载到内存中，并提供了方便的访问方法。每个样本由一个XFG图和对应的标签组成。
    你可以使用该类创建一个XFGDataset对象，并使用索引访问其中的样本数据。
'''


class XFGDataset(Dataset):
    def __init__(self, XFG_paths_json: str, config: DictConfig, vocab: Vocabulary) -> None:
        """
        Args:
            XFG_root_path: json file of list of XFG paths
        """
        super().__init__()
        self.__config = config
        assert exists(XFG_paths_json), f"{XFG_paths_json} not exists!"
        with open(XFG_paths_json, "r") as f:
            __XFG_paths_all = list(json.load(f))
        self.__vocab = vocab
        self.__XFGs = list()
        for xfg_path in __XFG_paths_all:
            xfg = XFG(path=xfg_path)
            # if len(xfg.nodes) != 0:
            self.__XFGs.append(xfg)
        self.__n_samples = len(self.__XFGs)

    # 返回数据集的样本数量
    def __len__(self) -> int:
        return self.__n_samples

    # 根据给定的索引返回数据集中的一个样本。索引可以是一个整数，用于指定样本的位置。返回一个XFGSample对象，包含了图数据和对应的标签。
    def __getitem__(self, index) -> XFGSample:
        xfg: XFG = self.__XFGs[index]
        return XFGSample(graph=xfg.to_torch(self.__vocab,
                                            self.__config.dataset.token.max_parts),  # max_parts: 16
                         label=xfg.label)

    # 返回数据集中样本的总数。
    def get_n_samples(self):
        return self.__n_samples
