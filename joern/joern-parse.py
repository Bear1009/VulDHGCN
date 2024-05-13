from argparse import ArgumentParser
from os import system
from os.path import join
from typing import cast
from omegaconf import OmegaConf, DictConfig


# 用于配置命令行参数解析器（ArgumentParser）并返回该解析器。
def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    # 使用arg_parser.add_argument()方法添加一个命令行参数。
    # 在这个例子中，参数的选项为-c和--config，表示配置文件的路径。help参default参type参数指定了参数的类型（字符串类型）。
    arg_parser.add_argument("-c",
                            "--config",
                            help="Path to YAML configuration file",
                            default="configs/dwk.yaml",
                            type=str)
    return arg_parser


if __name__ == "__main__":
    # 创建命令行参数解析器，并将其赋值给`__arg_parser`变量。
    __arg_parser = configure_arg_parser()
    # 使用命令行参数解析器解析命令行参数，并将结果存储在`__args`变量中。
    __args = __arg_parser.parse_args()
    # 使用`OmegaConf`库加载指定路径的YAML配置文件，并将加载的配置文件转换为`DictConfig`类型，存储在`config`变量中。
    config = cast(DictConfig, OmegaConf.load(__args.config))
    # 从配置文件中获取`joern_path`字段的值，并将其赋值给`joern`变量。
    joern = config.joern_path
    # 从配置文件中获取`data_folder`和`dataset.name`字段的值，并将它们拼接起来作为根路径，并将其赋值给`root`变量。
    root = join(config.data_folder, config.dataset.name)
    # 将根路径和字符串`"source-code"`拼接起来，作为源代码文件夹的路径，并将其赋值给`source_path`变量。
    source_path = join(root, "source-code")
    # 将根路径和字符串`"csv"`拼接起来，作为输出文件夹的路径，并将其赋值给`out_path`变量。
    out_path = join(root, "csv")
    # 使用`system`函数执行一个命令，该命令将`joern`变量的值、`out_path`变量的值和`source_path`变量的值作为参数传递给外部系统。
    # 具体执行的命令会根据`joern`的值以及`out_path`和`source_path`的路径生成。
    system(f"{joern} {out_path} {source_path}")
