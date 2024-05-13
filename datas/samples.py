from dataclasses import dataclass
from typing import List
from torch_geometric.data import Data, Batch
import torch


@dataclass
class XFGSample:
    graph: Data
    label: int


class XFGBatch:
    def __init__(self, XFGs: List[XFGSample]):
        # 首先通过列表解析的方式从 XFGs 中提取每个 XFGSample 对象的 label 属性，
        # 并使用 torch.tensor 将它们转换为一个 dtype 为 torch.long 的张量，并将其赋值给类的实例变量 labels
        self.labels = torch.tensor([XFG.label for XFG in XFGs],
                                   dtype=torch.long)
        # 接下来，创建一个空列表 graphs，用于存储每个 XFGSample 对象的 graph 属性。
        # 通过循环遍历 XFGs，将每个 XFGSample 对象的 graph 属性添加到 graphs 列表中。
        self.graphs = []
        for XFG in XFGs:
            self.graphs.append(XFG.graph)
        # 然后，使用 Batch.from_data_list 方法将 graphs 列表转换为一个批次数据对象，并将其赋值给类的实例变量 graphs。
        self.graphs = Batch.from_data_list(self.graphs)
        # 最后，将 XFGs 的长度赋值给类的实例变量 sz，以记录列表中 XFGSample 对象的数量。
        self.sz = len(XFGs)

    def __len__(self):
        return self.sz

    # 该方法的作用是将 XFGBatch 类的实例中的数据存储在固定内存中，以提高数据读取的效率。
    # 这在数据加载过程中特别适用，尤其是在使用 GPU 进行训练时。
    def pin_memory(self) -> "XFGBatch":
        self.labels = self.labels.pin_memory()
        self.graphs = self.graphs.pin_memory()
        return self
    # 这个方法在深度学习中常用于将数据移动到 GPU 上进行并行计算。
    # 通过将数据移动到 GPU 上，可以利用 GPU 的并行计算能力加速模型的训练或推断过程。
    def move_to_device(self, device: torch.device):
        self.labels = self.labels.to(device)
        self.graphs = self.graphs.to(device)