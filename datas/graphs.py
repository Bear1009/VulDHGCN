from dataclasses import dataclass
import networkx as nx
from os.path import exists
from typing import List
import torch
from torch_geometric.data import Data
from src.vocabulary import Vocabulary


# 通过设置frozen=True，可以确保创建的实例是不可变的，以提高安全性和可靠性。
@dataclass(frozen=True)
class XFGNode:
    ln: int


@dataclass
class XFGEdge:
    from_node: XFGNode
    to_node: XFGNode


# 这个类的目的是将XFG数据结构转换为torch-geometric图数据结构，以便在深度学习模型中使用。
@dataclass
class XFG:
    def __init__(self, path: str = None, xfg: nx.DiGraph = None):
        if xfg is not None:
            xfg_nx: nx.DiGraph = xfg
        elif path is not None:
            assert exists(path), f"xfg {path} not exists!"
            xfg_nx: nx.DiGraph = nx.read_gpickle(path)
        else:
            raise ValueError("invalid inputs!")
        self.__init_graph(xfg_nx)

    # 用于初始化XFG的节点、边和标签。
    # 该方法的目的是将给定的XFG图的节点、边和标签信息存储到XFG类的实例变量中，以便后续处理和使用。
    def __init_graph(self, xfg_nx: nx.DiGraph):
        # 初始化节点、边和标记为空列表
        self.__nodes, self.__edges, self.__tokens_list = [], [], []
        # 初始化节点到索引的映射字典
        self.__node_to_idx = {}
        # 创建一个临时字典k_to_nodes，用于将节点名称映射到对应的XFGNode对象。
        k_to_nodes = {}
        # 遍历XFG中的节点，并为每个节点执行以下操作：
        for idx, n in enumerate(xfg_nx):
            # 获取节点的文本标记(tokens)
            tokens = xfg_nx.nodes[n]["code_sym_token"]
            # 创建一个XFGNode对象，并将其添加到节点列表和对应的索引映射字典中。
            xfg_node = XFGNode(ln=n)
            # 将节点的文本标记添加到self.__tokens_list中。
            self.__tokens_list.append(tokens)
            self.__nodes.append(xfg_node)
            k_to_nodes[n] = xfg_node
            self.__node_to_idx[xfg_node] = idx
        # 遍历XFG中的节点和边，并为每个边执行以下操作：
        for n in xfg_nx:
            for k in xfg_nx[n]:
                # 根据边的类型（"c"或"d"），创建一个XFGEdge对象，并将其添加到边列表中。
                if xfg_nx[n][k]["c/d"] == "c":
                    self.__edges.append(
                        XFGEdge(from_node=k_to_nodes[n],
                                to_node=k_to_nodes[k]))
                elif xfg_nx[n][k]["c/d"] == "d":
                    self.__edges.append(
                        XFGEdge(from_node=k_to_nodes[n],
                                to_node=k_to_nodes[k]))
        # 获取XFG的标签并将其保存到self.__label中。
        self.__label = xfg_nx.graph["label"]

    # 这是XFG类中的属性方法，用于获取XFG对象的节点、边和标签信息。
    @property
    def nodes(self) -> List[XFGNode]:
        return self.__nodes

    @property
    def edges(self) -> List[XFGEdge]:
        return self.__edges

    @property
    def label(self) -> int:
        return self.__label

    # 这是XFG类中的to_torch方法，用于将XFG对象转换为torch-geometric图形。
    # 这样，你可以使用torch-geometric库中的功能和模型来处理和训练这个转换后的图形数据。
    def to_torch(self, vocab: Vocabulary, max_len: int) -> Data:
        """Convert this graph into torch-geometric graph

        Args:
            # vocab是一个Vocabulary对象，用于将节点的标记转换为对应的ID。
            vocab:
            # max_len是节点内容的最大长度，用于设置节点的向量长度。
            max_len: vector max_len for node content
        Returns:
            :torch_geometric.data.Data
        """
        # 存储节点标记
        node_tokens = []
        # 首先遍历所有节点，将节点的标记存储在node_tokens列表中。
        # 然后，创建一个大小为[n_node, max_len]的张量node_ids，并使用填充ID初始化该张量。
        for idx, n in enumerate(self.nodes):
            node_tokens.append(self.__tokens_list[idx])
        # [n_node, max seq len]
        node_ids = torch.full((len(node_tokens), max_len),
                              vocab.get_pad_id(),
                              dtype=torch.long)
        # 接下来，对于每个节点的标记，将其转换为对应的ID序列，
        # 并将其填充到node_ids张量中的相应位置。如果节点的标记长度超过max_len，则仅保留前max_len个标记。
        for tokens_idx, tokens in enumerate(node_tokens):
            ids = vocab.convert_tokens_to_ids(tokens)
            less_len = min(max_len, len(ids))
            node_ids[tokens_idx, :less_len] = torch.tensor(ids[:less_len],
                                                           dtype=torch.long)
        # 然后，创建一个边索引张量edge_index，其中每一列表示一条边的起始节点和结束节点的索引。
        edge_index = torch.tensor(list(
            zip(*[[self.__node_to_idx[e.from_node],
                   self.__node_to_idx[e.to_node]] for e in self.edges])),
            dtype=torch.long)

        """   
        # 获取节点的总数、边的总数
        num_nodes = edge_index.max() + 1
        num_edges = edge_index.shape[1]
        # 构建稀疏矩阵的坐标和值
        indices = edge_index
        values = torch.ones(num_edges)
        # 使用 torch.sparse_coo_tensor 创建稀疏矩阵
        edge_index = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes))
        """


        # 最后，将节点标记存储在Data对象的属性x中，并将边索引存储在Data对象的属性edge_index中。返回转换后的Data对象。
        # save token to `x` so Data can calculate properties like `num_nodes`
        return Data(x=node_ids, edge_index=edge_index)