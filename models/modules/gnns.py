from omegaconf import DictConfig
import torch
from torch_geometric.data import Batch
from torch_geometric.nn import TopKPooling, GCNConv, GINEConv, GATv2Conv, GatedGraphConv, GlobalAttention
import torch.nn.functional as F


from src.vocabulary import Vocabulary
from src.models.modules.common_layers import STEncoder

'''----------------------------------------嵌入HGCN-------------------------------------------------------------------'''
import torch.nn as nn
from src.models import manifolds
from src.models.layers.att_layers import GraphAttentionLayer
import src.models.layers.hyp_layers as hyp_layers
from src.models.layers.layers import GraphConvolution, Linear, get_dim_act
# 这个类实现了图卷积编码器的前向传播逻辑，用于处理图数据并提取特征。
# 你可以根据需要调用该类的实例，并使用输入数据调用其forward方法以获得编码后的特征表示。
class GraphConvEncoder(torch.nn.Module):
    """

    Kipf and Welling: Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)
    (https://arxiv.org/pdf/1609.02907.pdf)

    """

    # 类的初始化方法。它接受一些参数，如配置信息、词汇表、词汇表大小和填充索引。在初始化方法中，它首先调用父类的初始化方法
    def __init__(self, config: DictConfig, vocab: Vocabulary,
                 vocabulary_size: int,
                 pad_idx: int):
        super(GraphConvEncoder, self).__init__()
        self.__config = config
        self.__pad_idx = pad_idx
        # 这是一个STEncoder对象，用于对输入的节点进行嵌入（embedding）。
        self.__st_embedding = STEncoder(config, vocab, vocabulary_size, pad_idx)
        # 这是一个GCNConv对象，用于对节点嵌入进行图卷积操作。
        self.input_GCL = GCNConv(config.rnn.hidden_size, config.hidden_size)  # [256, 256]
        # 这是一个TopKPooling对象，用于对图卷积结果进行池化操作。
        self.input_GPL = TopKPooling(config.hidden_size,
                                     ratio=config.pooling_ratio)

        # 动态创建多个GCN层、GPL层，用于后续的隐藏图卷积和图池化操作。2层
        for i in range(config.n_hidden_layers - 1):
            '''
            这段代码使用了 Python 的 setattr 函数，它用于在对象上设置指定属性的值。
            在这个例子中，setattr 函数被用于设置 self 对象的属性，
            属性名是动态生成的，使用了字符串格式化来包含变量 i 的值。
            属性值是一个 GCNConv 对象，其参数包括config.hidden_size，num_layers 的值为 config.n_gru。
            '''
            setattr(self, f"hidden_GCL{i}",
                    GCNConv(config.hidden_size, config.hidden_size))
            setattr(
                self, f"hidden_GPL{i}",
                TopKPooling(config.hidden_size,
                            ratio=config.pooling_ratio))
        # 全局注意力池化操作
        self.attpool = GlobalAttention(torch.nn.Linear(config.hidden_size, 1))

    # 这是前向传播方法，接收一个Batch对象作为输入。
    # 它首先对输入的节点进行嵌入操作，然后使用图卷积和池化操作对节点嵌入进行多次处理。最后，使用全局注意力池化操作得到最终的输出结果。
    # batch在这里： Batch(x=[1282, 16], edge_index=[2, 9283], batch=[1282], ptr=[65])
    # tensor([[0, 2, 2, ..., 1280, 1281, 1281],
    #        [1, 2, 4, ..., 1278, 1278, 1279]], device='cuda:0')
    def forward(self, batched_graph: Batch):
        # [n nodes; rnn hidden]
        print("batch_graph中的内容：", batched_graph)
        # batch_graph中的内容： Batch(x=[1282, 16], edge_index=[2, 9283], batch=[1282], ptr=[65])
        # 输入节点嵌入
        node_embedding = self.__st_embedding(batched_graph.x)
        print("节点嵌入后的内容：", node_embedding)
        print("节点嵌入后的形状：", node_embedding.shape)  #  torch.Size([1095, 256])
        # 边索引
        edge_index = batched_graph.edge_index
        print(edge_index)
        batch = batched_graph.batch
        print("batched_graph.batch的内容：", batch)  # tensor([ 0,  0,  0,  ..., 63, 63, 63], device='cuda:0')
        print("batched_graph.batch的形状：", batch.shape)  # torch.Size([1095])
        node_embedding = F.relu(self.input_GCL(node_embedding, edge_index))
        print("经过一层图卷积得到的node_embedding内容：", node_embedding)
        print("经过一层图卷积得到的node_embedding形状：", node_embedding.shape)  # torch.Size([1095, 256])
        node_embedding, edge_index, _, batch, _, _ = self.input_GPL(node_embedding, edge_index, None,
                                                                    batch)
        print("经过一层图池化层得到的node_embedding：", node_embedding)
        print("经过一层图池化层得到的node_embedding的形状：", node_embedding.shape)
        print("经过一层图池化层得到的edge_index：", edge_index)
        print("经过一层图池化层得到的edge_index的形状：", edge_index.shape)
        print("经过一层图池化层得到的batch：", batch)
        # [n_XFG; XFG hidden dim]
        out = self.attpool(node_embedding, batch)  # [1281, 256], [1282]
        print("经过attpool后的输出：", out)
        print("经过attpool后的输出形状：", out.shape)    # torch.Size([64, 256])
        for i in range(self.__config.n_hidden_layers - 1):
            node_embedding = F.relu(getattr(self, f"hidden_GCL{i}")(node_embedding, edge_index))
            node_embedding, edge_index, _, batch, _, _ = getattr(self, f"hidden_GPL{i}")(
                node_embedding, edge_index, None, batch)
            # 全局注意力池化
            out += self.attpool(node_embedding, batch)
        # [n_XFG; XFG hidden dim]
        print("经过GCN输出最终的形状：", out.shape)  # torch.Size([64, 256])
        return out  # [64, 256]


class GatedGraphConvEncoder(torch.nn.Module):
    """

    from Li et al.: Gated Graph Sequence Neural Networks (ICLR 2016)
    (https://arxiv.org/pdf/1511.05493.pdf)

    """

    def __init__(self, config: DictConfig, vocab: Vocabulary,
                 vocabulary_size: int,
                 pad_idx: int):
        super(GatedGraphConvEncoder, self).__init__()
        self.__config = config
        self.__pad_idx = pad_idx
        self.__st_embedding = STEncoder(config, vocab, vocabulary_size, pad_idx)

        self.input_GCL = GatedGraphConv(out_channels=config.hidden_size, num_layers=config.n_gru)

        self.input_GPL = TopKPooling(config.hidden_size,
                                     ratio=config.pooling_ratio)

        for i in range(config.n_hidden_layers - 1):
            setattr(self, f"hidden_GCL{i}",
                    GatedGraphConv(out_channels=config.hidden_size, num_layers=config.n_gru))
            setattr(
                self, f"hidden_GPL{i}",
                TopKPooling(config.hidden_size,
                            ratio=config.pooling_ratio))
        self.attpool = GlobalAttention(torch.nn.Linear(config.hidden_size, 1))

    def forward(self, batched_graph: Batch):
        # [n nodes; rnn hidden]
        node_embedding = self.__st_embedding(batched_graph.x)
        edge_index = batched_graph.edge_index
        batch = batched_graph.batch
        node_embedding = F.relu(self.input_GCL(node_embedding, edge_index))
        node_embedding, edge_index, _, batch, _, _ = self.input_GPL(node_embedding, edge_index, None,
                                                                    batch)
        # [n_XFG; XFG hidden dim]
        out = self.attpool(node_embedding, batch)
        for i in range(self.__config.n_hidden_layers - 1):
            node_embedding = F.relu(getattr(self, f"hidden_GCL{i}")(node_embedding, edge_index))
            node_embedding, edge_index, _, batch, _, _ = getattr(self, f"hidden_GPL{i}")(
                node_embedding, edge_index, None, batch)
            out += self.attpool(node_embedding, batch)
        # [n_XFG; XFG hidden dim]
        return out


'''----------------------------------------嵌入HGCN-------------------------------------------------------------------'''


class Encoder(nn.Module):
    """
    Encoder abstract class.
    """

    # 初始化对象。在这里，它接受一个参数 c，并将其赋值给 self.c，表示曲率
    def __init__(self, c, config: DictConfig):
        super(Encoder, self).__init__()
        self.c = c
        # 全局注意力池化操作
        self.attpool = GlobalAttention(torch.nn.Linear(config.hidden_size, 1))

    # batched_graph.x [1282x16] adj[2x9283]
    def encode(self, x, adj, batch):
        # 如果 encode_graph 为真，则调用 self.layers.forward(input) 对输入进行编码。这里假设 self.layers 是一个可以进行前向传播的模型层。
        if self.encode_graph:
            input = (x, adj)
            output, _ = self.layers.forward(input)  # [1282, 256]
        # 如果 encode_graph 为假，则调用 self.layers.forward(x) 对输入 x 进行编码。
        else:
            output = self.layers.forward(x)

        output = self.attpool(output, batch)  # [1281, 256], [1282]

        return output


'''class HGCN(Encoder):
    """
    Hyperbolic-GCN.
    """

    def __init__(self, c, args):
        super(HGCN, self).__init__(c)
        # 初始化流形，根据参数中的 `manifold` 值选择相应的流形模型
        self.manifold = getattr(manifolds, args.manifold)()
        # 确保网络层数大于1
        assert args.num_layers > 1
        # 调用 hyp_layers 模块中的函数，获取维度、激活函数和曲率信息
        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(args)
        # 将当前流形的曲率加入到曲率列表中
        self.curvatures.append(self.c)
        hgc_layers = []
        # 循环构建HGCN层
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]

            # 将构建的超边卷积层加入列表中
            hgc_layers.append(
                hyp_layers.HyperbolicGraphConvolution(
                    self.manifold, in_dim, out_dim, c_in, c_out, args.dropout, act, args.bias, args.use_att,
                    args.local_agg
                )
            )
        # 将所有的双曲图卷积层组成一个层序列
        self.layers = nn.Sequential(*hgc_layers)
        # 是否进行图编码
        self.encode_graph = True

    def encode(self, x, adj):
        # 这里调用了 manifold 对象的 proj_tan0 方法，传入了参数 x 和 self.curvatures[0]。它的作用是将输入的张量 x 投影到切空间中。
        x_tan = self.manifold.proj_tan0(x, self.curvatures[0])
        # 接着，调用了 manifold 对象的 expmap0 方法，传入了参数 x_tan 和 self.curvatures[0]。它的作用是在切空间中执行指数映射，得到一个在超几何空间中的张量 x_hyp。
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        # 然后，又调用了 manifold 对象的 proj 方法，传入了参数 x_hyp 和 self.curvatures[0]。这个方法将 x_hyp 投影回超几何空间。
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])
        # 最后，调用了父类 HGCN 的 encode 方法，传入了 x_hyp 和 adj 作为参数。这里使用 super() 方法是为了调用父类的方法，
        # 它会将处理后的 x_hyp 和 adj 传递给父类的 encode 方法进行进一步处理。
        return super(HGCN, self).encode(x_hyp, adj)'''


class HGCN(Encoder):
    """
    Hyperbolic-GCN.
    """

    def __init__(self, config: DictConfig, vocab, vocabulary_size, pad_idx, c, args):
        super(HGCN, self).__init__(c, config)

        self.__config = config
        self.__pad_idx = pad_idx
        # 这是一个STEncoder对象，用于对输入的节点进行嵌入（embedding）。
        self.__st_embedding = STEncoder(config, vocab, vocabulary_size, pad_idx)

        # 初始化流形，根据参数中的 `manifold` 值选择相应的流形模型
        self.manifold = getattr(manifolds, args.manifold)()
        # 确保网络层数大于1
        assert args.num_layers > 1
        # 调用 hyp_layers 模块中的函数，获取维度、激活函数和曲率信息
        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(args)
        # 获取到的维度、激活函数和曲率信息： [16, 128] [<function relu at 0x7fe59cdbb310>] [tensor([1.])]
        print("获取到的维度、激活函数和曲率信息：", dims, acts, self.curvatures)
        # 将当前流形的曲率加入到曲率列表中
        self.curvatures.append(self.c)
        hgc_layers = []
        # 循环构建HGCN层
        # batched_graph.x [1282x16] adj[2x9283]
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            # in_dim = 16, out_dim = 128
            in_dim, out_dim = dims[i], dims[i + 1]
            # act = relu
            act = acts[i]

            # 将构建的双曲卷积层加入列表中
            hgc_layers.append(
                hyp_layers.HyperbolicGraphConvolution(
                    self.manifold, in_dim, out_dim, c_in, c_out, args.dropout, act, args.bias, args.use_att,
                    args.local_agg
                )
            )
        # 将所有的双曲图卷积层组成一个层序列
        self.layers = nn.Sequential(*hgc_layers)
        # 是否进行图编码
        self.encode_graph = True

    def encode(self, batched_graph: Batch):
        # batch在这里：Batch(x=[1282, 16], edge_index=[2, 9283], batch=[1282], ptr=[65])
        print("传到encode中的batch内容：", batched_graph)
        # 输入节点嵌入
        node_embedding = self.__st_embedding(batched_graph.x)
        x = node_embedding
        print("HGCN中节点嵌入后x的内容：", x)
        print("HGCN中节点嵌入后x的形状：", x.shape)
        # 边索引 edge_index=[2, 9283]
        adj = batched_graph.edge_index
        print("adj中的内容：", adj)
        batch = batched_graph.batch
        print("HGCN中的batched_graph.batch", batch)
        # batched_graph.x [1282x16] adj[2x9283]
        # 这里调用了 manifold 对象的 proj_tan0 方法，传入了参数 x 和 self.curvatures[0]。它的作用是将输入的张量 x 投影到切空间中。
        x_tan = self.manifold.proj_tan0(x, self.curvatures[0])
        # 接着，调用了 manifold 对象的 expmap0 方法，传入了参数 x_tan 和 self.curvatures[0]。它的作用是在切空间中执行指数映射，得到一个在超几何空间中的张量 x_hyp。
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        # 然后，又调用了 manifold 对象的 proj 方法，传入了参数 x_hyp 和 self.curvatures[0]。这个方法将 x_hyp 投影回超几何空间。
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])
        print("encode中x_hyp的形状为:", x_hyp.shape) # [1282, 16]
        # 最后，调用了父类 HGCN 的 encode 方法，传入了 x_hyp 和 adj 作为参数。这里使用 super() 方法是为了调用父类的方法，
        # 它会将处理后的 x_hyp 和 adj 传递给父类的 encode 方法进行进一步处理。
        return super(HGCN, self).encode(x_hyp, adj, batch)
        # return super(HGCN, self).encode(x_hyp, edge_index)