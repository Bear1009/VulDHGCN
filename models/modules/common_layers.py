from torch import nn
from omegaconf import DictConfig
import torch
import numpy
from gensim.models import KeyedVectors
from src.vocabulary import Vocabulary
from os.path import exists


def linear_after_attn(in_dim: int, out_dim: int, activation: str) -> nn.Module:
    """Linear layers after attention

        Args:
            in_dim (int): input dimension
            out_dim (int): output dimension
            activation (str): the name of activation function
        """
    # add drop out?
    # 通过 torch.nn.Sequential 将这些层按照顺序组合成一个序列模型，并将其作为结果返回。
    return torch.nn.Sequential(
        # 一个全连接层，输入大小为 2 * in_dim，输出大小也为 2 * in_dim。这个层将输入的特征进行线性变换。
        torch.nn.Linear(2 * in_dim, 2 * in_dim),
        # 一个批归一化层，用于对上一层的输出进行归一化处理，有助于加速训练和提高模型的泛化能力。
        torch.nn.BatchNorm1d(2 * in_dim),
        # 一个激活函数层，通过调用 get_activation 函数获取指定的激活函数，并将其应用于上一层的输出。
        get_activation(activation),
        # 另一个全连接层，输入大小为 2 * in_dim，输出大小为 out_dim。这个层将前面层的输出映射到最终的输出大小。
        torch.nn.Linear(2 * in_dim, out_dim),
    )


# 激活函数
activations = {
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "lkrelu": nn.LeakyReLU(0.3)
}


# 获取激活函数
def get_activation(activation_name: str) -> torch.nn.Module:
    if activation_name in activations:
        return activations[activation_name]
    raise KeyError(f"Activation {activation_name} is not supported")

    # 这个 RNNLayer 类可以作为模型的一部分，用于处理序列数据的特征提取和表示学习。
    '''
    该代码段定义了一个 RNNLayer 类，通过 LSTM 层实现了对输入序列的处理，得到节点的嵌入表示。
    通过排序和逆排序操作，保证了嵌入结果与原始输入的对应关系。
    这个 RNNLayer 类可以作为模型的一部分，用于处理序列数据的特征提取和表示学习。
    '''


class RNNLayer(torch.nn.Module):
    """

    """
    __negative_value = -numpy.inf

    def __init__(self, config: DictConfig, pad_idx: int):
        super(RNNLayer, self).__init__()
        self.__pad_idx = pad_idx
        self.__config = config

        # 创建了一个双向LSTM的RNN对象,实现了对输入序列的处理，得到节点嵌入表示
        self.__rnn = nn.LSTM(
            input_size=config.embed_size,
            hidden_size=config.rnn.hidden_size,
            num_layers=config.rnn.num_layers,
            bidirectional=config.rnn.use_bi,
            # 该代码段根据条件判断，如果模型中的 RNN 层数大于 1，则将 dropout 设置为 config.rnn.drop_out 的值，否则将其设置为 0。
            dropout=config.rnn.drop_out if config.rnn.num_layers > 1 else 0,
            batch_first=True)
        # 创建了一个Dropout对象
        self.__dropout_rnn = nn.Dropout(config.rnn.drop_out)

    '''
    前向传播方法，接收输入张量 subtokens_embed 和 node_ids。
    在方法中，首先通过对 node_ids 进行处理，找到填充索引的位置，并对其进行排序。然后根据排序后的索引对 subtokens_embed 进行排序，并将其转换为 PackedSequence 格式。
    接着将 PackedSequence 作为输入传递给 LSTM 层进行计算，得到输出张量 node_embedding。最后通过 Dropout 对 node_embedding 进行处理，
    并根据之前的排序进行逆操作，得到最终的节点嵌入张量 node_embedding。
    '''

    def forward(self, subtokens_embed: torch.Tensor, node_ids: torch.Tensor):
        """

        Args:
            subtokens_embed: [n nodes; max parts; embed dim]
            node_ids: [n nodes; max parts]

        Returns:

        """
        with torch.no_grad():
            is_contain_pad_id, first_pad_pos = torch.max(
                node_ids == self.__pad_idx, dim=1)
            first_pad_pos[~is_contain_pad_id] = node_ids.shape[
                1]  # if no pad token use len+1 position
            sorted_path_lengths, sort_indices = torch.sort(first_pad_pos,
                                                           descending=True)
            _, reverse_sort_indices = torch.sort(sort_indices)
            sorted_path_lengths = sorted_path_lengths.to(torch.device("cpu"))
        subtokens_embed = subtokens_embed[sort_indices]
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(
            subtokens_embed, sorted_path_lengths, batch_first=True)
        # [2; N; rnn hidden]
        _, (node_embedding, _) = self.__rnn(packed_embeddings)
        # [N; rnn hidden]
        node_embedding = node_embedding.sum(dim=0)

        # [n nodes; max parts; rnn hidden]
        node_embedding = self.__dropout_rnn(
            node_embedding)[reverse_sort_indices]

        return node_embedding


# 这个STEncoder模型类用于对输入的语句进行编码，其中包括对输入序列进行词嵌入和RNN编码操作。
class STEncoder(torch.nn.Module):
    """

    encoder for statement

    """

    def __init__(self, config: DictConfig, vocab: Vocabulary,
                 vocabulary_size: int,
                 pad_idx: int):
        # 初始化父类
        super(STEncoder, self).__init__()
        self.__config = config
        self.__pad_idx = pad_idx
        # 根据传入的参数初始化嵌入层，使用nn.Embedding类来创建一个词嵌入层
        self.__wd_embedding = nn.Embedding(vocabulary_size,
                                           config.embed_size,
                                           padding_idx=pad_idx)  # config.embed_size 指定词向量维度
        # Additional embedding value for masked token
        torch.nn.init.xavier_uniform_(self.__wd_embedding.weight.data)
        # 如果存在预训练的词向量文件（通过config.w2v_path指定路径），则调用__add_w2v_weights方法将预训练的词向量添加到词嵌入层中。
        if exists(config.w2v_path):
            self.__add_w2v_weights(config.w2v_path, vocab)
        # 初始化一个RNNLayer对象作为self.__rnn_attn，用于对序列进行编码。
        self.__rnn_attn = RNNLayer(config, pad_idx)

    # 该方法用于将预训练的词向量添加到词嵌入层中
    def __add_w2v_weights(self, w2v_path: str, vocab: Vocabulary):
        # 接受预训练词向量的路径w2v_path和词汇表vocab作为参数
        """
        add pretrained word embedding to embedding layer

        Args:
            w2v_path: path to the word2vec model

        Returns:

        """
        # 加载预训练的词向量模型
        model = KeyedVectors.load(w2v_path, mmap="r")
        w2v_weights = self.__wd_embedding.weight.data
        # 遍历词汇表中的词，并将对应词的词向量赋值给词嵌入层的权重。
        # 通过索引和词汇表进行匹配，将对应词的词向量赋值给权重。
        for wd in model.index2word:
            w2v_weights[vocab.convert_token_to_id(wd)] = torch.from_numpy(model[wd])
        self.__wd_embedding.weight.data.copy_(w2v_weights)

    def forward(self, seq: torch.Tensor):
        """

        Args:
            seq: [n nodes (seqs); max parts (seq len); embed dim]

        Returns:

        """
        # [n nodes; max parts; embed dim]
        # 首先通过词嵌入层self.__wd_embedding对输入序列进行词嵌入，得到词嵌入后的张量wd_embedding。
        wd_embedding = self.__wd_embedding(seq)
        # [n nodes; rnn hidden]
        # 然后，将词嵌入后的张量传入self.__rnn_attn中，进行RNN层的编码操作，得到节点编码的张量node_embedding。
        node_embedding = self.__rnn_attn(wd_embedding, seq)
        # 将节点编码的张量作为输出返回。
        return node_embedding
