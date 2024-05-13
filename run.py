import math
import os
import time
from argparse import ArgumentParser
from typing import cast
import torch
from commode_utils.common import print_config
from omegaconf import OmegaConf, DictConfig  # OmegaConf是一个用于处理配置文件的Python库，它提供了一种方便加载、访问和修改配置数据的方式。
from pytorch_lightning import seed_everything
from src.datas.datamodules import XFGDataModule
from src.models.vd import DeepWuKong
from src.train import train
from src.utils import filter_warnings, PAD
from src.vocabulary import Vocabulary


# 函数功能：配置和设置命令行参数解析器
def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    # 使用 add_argument 方法添加不同的参数和选项，并定义它们的属性，例如名称、类型、帮助信息等等。
    arg_parser.add_argument("-c",
                            "--config",
                            help="Path to YAML configuration file",
                            default="configs/dwk.yaml",
                            type=str)
    # 返回的 ArgumentParser 对象可以进一步用于解析命令行参数，并将其应用于程序的其他部分，以实现不同的功能或配置。
    return arg_parser


def vul_detect(config_path: str):  # 漏洞检测功能，接受一个配置文件路径作为参数
    # 过滤警告信息
    filter_warnings()
    # 使用OmegaConf.load()函数加载配置文件，然后使用cast()函数将加载的配置文件强制转换为DictConfig类型
    config = cast(DictConfig, OmegaConf.load(config_path))
    # 打印配置信息，包括 gnn、classifier和hyper_parameters
    print_config(config, ["gnn", "classifier", "hyper_parameters"])

    # 设置随机种子，可以控制随机数生成的起始状态，从而实现结果的可重现性。
    seed_everything(config.seed, workers=True)  # workers则是一个布尔值，用于指示是否在多线程或多进程环境中设置随机种子。

    # 根据配置文件中的路径构建词汇表（Vocabulary）并获取词汇表的大小和填充索引。
    vocab = Vocabulary.build_from_w2v(config.gnn.w2v_path)
    print("vocab,vocab_size,pad_idx的内容：", vocab)
    vocab_size = vocab.get_vocab_size()
    print(vocab_size)  # 3214
    pad_idx = vocab.get_pad_id()  # 0
    print(pad_idx)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 初始化数据模块（XFGDataModule）
    data_module = XFGDataModule(config, vocab)
    print(data_module.train_dataloader())
    # 初始化模型（DeepWuKong），并将配置、词汇表、词汇表大小和填充索引作为参数传递。
    model = DeepWuKong(config, vocab, vocab_size, pad_idx)

    # 并行使用cuda训练模型
    if torch.cuda.device_count() > 1:  # 检查电脑是否有多块GPU
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)  # 将模型对象转变为多GPU并行运算的模型
    model.to(device)  # 把并行的模型移动到GPU上

    # 调用train函数，传递模型、数据模块和配置进行训练。
    train(model, data_module, config)


if __name__ == "__main__":
    print(os.getcwd())
    # 调用configure_arg_parser()函数来配置命令行参数解析器，并将其赋值给__arg_parser变量。
    __arg_parser = configure_arg_parser()
    # 调用__arg_parser.parse_args()方法来解析命令行参数，并将结果赋值给__args变量。
    __args = __arg_parser.parse_args()
    print("__args中的内容：", __args)
    # 调用vul_detect()函数，并传递__args.config作为参数。
    start_time = time.time()
    vul_detect(__args.config)
    end_time = time.time()
    run_time = end_time - start_time
    print("代码运行时间为：{:.2f}秒".format(run_time))
