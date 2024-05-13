from argparse import ArgumentParser
from typing import cast, List
from omegaconf import OmegaConf, DictConfig
import json
import networkx as nx
from gensim.models import Word2Vec, KeyedVectors
from os import cpu_count
from src.utils import PAD, MASK, UNK
from tqdm import tqdm
from multiprocessing import cpu_count, Manager, Pool
import functools

SPECIAL_TOKENS = [PAD, UNK, MASK]
USE_CPU = cpu_count()

# 从指定的path读取XFG图中所有节点的代码标记序列。
def process_parallel(path: str, split_token: bool):
    """

    Args:
        path：文件路径。
        split_token：是否对代码进行分词的标志。

    Returns:

    """
    # 使用 nx.read_gpickle() 从指定的路径读取图形表示数据，并将其存储在变量 xfg 中。
    xfg = nx.read_gpickle(path)
    # 创建一个空的列表 tokens_list，用于存储处理后的代码标记序列。
    tokens_list = list()
    # 遍历 xfg 中的每个节点 ln：
    for ln in xfg:
        # 获取节点 ln 的代码符号标记序列 code_tokens。
        code_tokens = xfg.nodes[ln]["code_sym_token"]
        # 如果 code_tokens 的长度不为零，则将其添加到 tokens_list 中。
        if len(code_tokens) != 0:
            tokens_list.append(code_tokens)
    # 返回 tokens_list，其中包含了所有节点的代码标记序列。
    return tokens_list

# 训练word2vec词嵌入模型
def train_word_embedding(config_path: str):
    """
    train word embedding using word2vec

    Args:
        config_path:

    Returns:

    """
    # 加载指定的配置文件，将其存储在变量 config 中。
    config = cast(DictConfig, OmegaConf.load(config_path))
    # 根据配置获取训练数据的路径。
    cweid = config.dataset.name
    root = config.data_folder
    train_json = f"{root}/{cweid}/train.json"
    with open(train_json, "r") as f:
        paths = json.load(f)
    tokens_list = list()
    with Manager():
        pool = Pool(USE_CPU)
        # 使用并行处理的方式，处理数据路径列表中的每个路径，调用 process_parallel 函数处理，并将返回的代码标记序列存储在 tokens 列表中。
        process_func = functools.partial(process_parallel,
                                         split_token=config.split_token)
        tokens: List = [
            res
            for res in tqdm(
                pool.imap_unordered(process_func, paths),
                desc=f"xfg paths: ",
                total=len(paths),
            )
        ]
        pool.close()
        pool.join()
    # 将 tokens 列表展开为 tokens_list，其中包含了所有路径对应的代码标记序列。
    for token_l in tokens:
        tokens_list.extend(token_l)

    print("training w2v...")
    num_workers = cpu_count(
    ) if config.num_workers == -1 else config.num_workers
    # 创建 Word2Vec 模型对象，并使用 tokens_list 进行训练。模型的参数由配置文件中的相关配置指定。
    # size = config.gnn.embed_size = 256，vocabulary_size: 190000，num_workers = 8
    model = Word2Vec(sentences=tokens_list, min_count=3, size=config.gnn.embed_size,
                     max_vocab_size=config.dataset.token.vocabulary_size, workers=num_workers, sg=1)
    # 将训练好的词嵌入保存在指定的路径下。
    model.wv.save(f"{root}/{cweid}/w2v.wv")


# 该函数用于加载已训练好的词嵌入模型。
def load_wv(config_path: str):
    """

    Args:
        config_path:

    Returns:

    """
    # 加载指定的配置文件，将其存储在变量 config 中。
    config = cast(DictConfig, OmegaConf.load(config_path))
    # 根据配置获取词嵌入模型文件的路径。
    cweid = config.dataset.name
    # 使用 KeyedVectors 类加载词嵌入模型文件，并将其存储在 model 变量中。
    model = KeyedVectors.load(f"{config.data_folder}/{cweid}/w2v.wv", mmap="r")
    # 最后，该函数会打印一个空行。请注意，该函数的返回值为空（Returns 部分为空），它主要完成了加载词嵌入模型的过程，并将模型存储在 model 变量中。
    print()


if __name__ == '__main__':
    # 创建了一个参数解析器 ArgumentParser 并添加了一个名为 config 的命令行参数。
    # 该参数用于指定 YAML 配置文件的路径，默认为 "configs/dwk.yaml"。
    __arg_parser = ArgumentParser()
    __arg_parser.add_argument("-c",
                              "--config",
                              help="Path to YAML configuration file",
                              default="configs/dwk.yaml",
                              type=str)
    # 然后，你解析了命令行参数，并将解析结果存储在 __args 变量中。
    __args = __arg_parser.parse_args()
    # 调用了 train_word_embedding 函数，并传递了 __args.config 作为参数，以训练词嵌入模型。
    train_word_embedding(__args.config)
    '''
        如果你想要加载词嵌入模型，你可以取消注释 load_wv(__args.config) 行，
        并传递相同的配置文件路径作为参数。这将加载已训练好的词嵌入模型并将其存储在 model 变量中。
    '''
    # load_wv(__args.config)
