from typing import List, cast
from os.path import join
from argparse import ArgumentParser
import os
from src.utils import unique_xfg_sym, split_list
import networkx as nx
from src.preprocess.symbolizer import clean_gadget
from tqdm import tqdm
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf, DictConfig
from multiprocessing import cpu_count, Manager, Pool, Queue
import functools
import dataclasses
from src.preprocess.symbolizer import tokenize_code_line

# 代码功能：符号化和分割数据集

# 获取当前计算机的CPU核心数。
USE_CPU = cpu_count()


# 配置命令行参数解析器，返回一个ArgumentParser对象，该对象用于解析命令行参数。
def configure_arg_parser() -> ArgumentParser:
    # 创建了一个ArgumentParser对象，并将其赋值给arg_parser变量。
    arg_parser = ArgumentParser()
    # 使用arg_parser.add_argument()方法添加了一个命令行参数选项。
    '''
        -c或--config：选项的短名称和长名称，用于在命令行中指定该选项。
        help：选项的帮助文本，用于描述该选项的作用和用法。
        default：选项的默认值。如果在命令行中没有指定该选项，将使用默认值。
        type：选项的类型。在这种情况下，选项的类型为字符串（str）。
    '''
    arg_parser.add_argument("-c",
                            "--config",
                            help="Path to YAML configuration file",
                            default="configs/dwk.yaml",
                            type=str)
    # 添加完所有的命令行参数选项后，函数返回配置好的ArgumentParser对象。这样，用户可以使用该对象解析命令行参数，并获取相应的配置信息。
    return arg_parser


# 用于从原始数据中提取唯一的数据样本（不包含符号化）。
# 返回唯一的xfg数据路径列表。
def unique_data(cweid: str, root: str):
    """
    unique raw data without symbolization
    Args：
        cweid：CWE ID，用于指定要处理的特定CWE（Common Weakness Enumeration）。
        root：数据根目录，用于指定原始数据的根目录。
    Returns:

    """
    # 首先构建了XFG数据的根路径XFG_root_path，该路径由root、cweid和"XFG"组成。
    XFG_root_path = join(root, cweid, "XFG")
    # 然后，获取指定CWE下的测试用例ID列表testcaseids。
    testcaseids = os.listdir(XFG_root_path)  # /home/czx/codes/DeepWukong/data/CWE119/XFG
    xfg_paths = list()
    # 接下来，遍历每个测试用例ID，并构建相应的测试用例根路径testcase_root_path。
    for testcase in testcaseids:
        testcase_root_path = join(XFG_root_path, testcase)  # /home/czx/codes/DeepWukong/data/CWE119/XFG/6
        # 在测试用例根路径下，根据四种类型的XFG数据（"arith"、"array"、"call"、"ptr"），分别构建对应的类型根路径k_root_path。
        for k in ["arith", "array", "call", "ptr"]:
            k_root_path = join(testcase_root_path, k) # /home/czx/codes/DeepWukong/data/CWE119/XFG/6/arith
            # 将所有XFG数据路径存储在xfg_paths列表中。
            xfg_ps = os.listdir(k_root_path)
            for xfg_p in xfg_ps:
                xfg_path = join(k_root_path, xfg_p)  # /home/czx/codes/DeepWukong/data/CWE119/XFG/6/arith/15.xfg.pkl
                xfg_paths.append(xfg_path)
    # remove duplicates and conflicts
    # 接下来，调用unique_xfg_sym函数，将xfg_paths作为参数，获取唯一的XFG数据字典xfg_dict。
    # XFG数据字典中的键是XFG数据的MD5值，值是一个字典，包含XFG数据的路径和标签信息。
    xfg_dict = unique_xfg_sym(xfg_paths)
    xfg_unique_paths = list()
    # 然后，遍历XFG数据字典中的每个MD5值，根据标签信息筛选出不包含冲突的唯一XFG数据路径，并将其存储在xfg_unique_paths列表中。
    for md5 in xfg_dict:
        # remove conflicts
        if xfg_dict[md5]["label"] != -1:
            xfg_unique_paths.append(xfg_dict[md5]["xfg"])
    # 最后，函数返回xfg_unique_paths，即唯一的XFG数据路径列表。
    return xfg_unique_paths

'''
    XFG：类型为nx.DiGraph，表示XFG有向图数据。
    xfg_path：类型为str，表示XFG图数据的路径。
    to_remove：类型为bool，表示是否需要移除该XFG图数据。
    is_finished：类型为bool，表示是否已完成处理该XFG图数据。
    这个数据类的作用是用来传递消息，使用 @dataclasses.dataclass 装饰器可以使得这个类拥有一些默认的行为，比如自动生成 __init__ 方法、__repr__ 方法等，使得定义和使用类更加方便。
'''
@dataclasses.dataclass
class QueueMessage:
    XFG: nx.DiGraph
    xfg_path: str
    to_remove: bool = False
    is_finished: bool = False


# 这段代码定义了一个名为handle_queue_message的函数，用于处理队列中的消息。函数接受一个queue参数，表示队列对象。
def handle_queue_message(queue: Queue):
    """

    Args:
        queue:

    函数的主要逻辑如下：
        使用一个循环不断从队列中获取消息。
        如果接收到的消息的is_finished字段为True，则表示处理完成，退出循环。
        如果接收到的消息的to_remove字段为True，则执行删除操作，即通过os.system调用rm命令删除对应的XFG图数据文件。
        如果接收到的消息的XFG字段不为空，表示有新的XFG图数据需要保存。
        调用nx.write_gpickle将XFG保存到指定的路径xfg_path中，并计数保存的XFG图数据数量。
    """
    xfg_ct = 0
    while True:
        message: QueueMessage = queue.get()
        if message.is_finished:
            break
        if message.to_remove:
            os.system(f"rm {message.xfg_path}")
        else:
            if message.XFG is not None:
                nx.write_gpickle(message.XFG, message.xfg_path)
                xfg_ct += 1
    return xfg_ct


# 这段代码定义了一个名为process_parallel的函数，用于并行处理测试用例中的XFG图数据。
def process_parallel(testcaseid: str, queue: Queue, XFG_root_path: str, split_token: bool):
    """

    Args:
        testcaseid：测试用例ID。
        queue：消息队列，用于将处理结果传递给主进程。
        XFG_root_path：XFG图数据的根目录路径。
        split_token：一个布尔值，表示是否对代码进行分词。

    Returns:

    """
    # 根据测试用例ID构建对应的路径。
    testcase_root_path = join(XFG_root_path, testcaseid)
    # 遍历四种类型（arith、array、call、ptr）的XFG图数据文件。
    for k in ["arith", "array", "call", "ptr"]:
        k_root_path = join(testcase_root_path, k)
        xfg_ps = os.listdir(k_root_path)
        for xfg_p in xfg_ps:
            xfg_path = join(k_root_path, xfg_p)
            # 读取XFG图数据文件，并检查每个节点是否包含code_sym_token字段，如果存在则表示已经处理过，直接返回测试用例ID。
            xfg: nx.DiGraph = nx.read_gpickle(xfg_path)
            for idx, n in enumerate(xfg):
                if "code_sym_token" in xfg.nodes[n]:
                    return testcaseid
            # 读取对应的源代码文件，并逐行读取文件内容。
            file_path = xfg.graph["file_paths"][0]
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                file_contents = f.readlines()
            code_lines = list()
            # 对每个节点的代码行进行清洗操作，得到符号化的代码行。
            for n in xfg:
                code_lines.append(file_contents[n - 1])
            sym_code_lines = clean_gadget(code_lines)

            to_remove = list()
            # 遍历每个节点，将符号化的代码行进行分词，得到code_sym_token字段，并将长度为零的节点标记为需要移除。
            for idx, n in enumerate(xfg):
                xfg.nodes[n]["code_sym_token"] = tokenize_code_line(sym_code_lines[idx], split_token)
                if len(xfg.nodes[n]["code_sym_token"]) == 0:
                    to_remove.append(n)
            xfg.remove_nodes_from(to_remove)

            # 如果经过移除操作后，XFG图节点数不为零，则将处理结果通过消息队列传递给主进程，包含XFG图数据和文件路径。
            # 否则，将处理结果通过消息队列传递给主进程，包含XFG图数据、文件路径以及移除标记。
            if len(xfg.nodes) != 0:
                queue.put(QueueMessage(xfg, xfg_path))
            else:
                queue.put(QueueMessage(xfg, xfg_path, to_remove=True))
    return testcaseid


# 这段代码定义了一个名为add_symlines的函数，用于为给定的CWE ID下的测试用例添加符号化的代码行。
def add_symlines(cweid: str, root: str, split_token: bool):
    """

    Args:
        cweid：CWE ID。
        root：根目录路径。
        split_token：一个布尔值，表示是否对代码进行分词。
    Returns:

    """
    # 根据CWE ID构建XFG图数据的根目录路径。
    XFG_root_path = join(root, cweid, "XFG")
    # 获取测试用例ID列表，并计算测试用例的数量。
    testcaseids = os.listdir(XFG_root_path)
    testcase_len = len(testcaseids)

    # 使用Manager创建一个消息队列（message_queue）。
    with Manager() as m:
        message_queue = m.Queue()  # type: ignore
        # 使用Pool创建一个进程池，进程数为USE_CPU。
        pool = Pool(USE_CPU)
        # 使用apply_async方法异步调用handle_queue_message函数，将消息队列作为参数传递给它，并返回一个结果对象（xfg_ct）。
        xfg_ct = pool.apply_async(handle_queue_message, (message_queue,))
        # 使用functools.partial创建一个部分函数process_func，将消息队列、XFG图数据根目录路径和split_token作为默认参数。
        process_func = functools.partial(process_parallel, queue=message_queue, XFG_root_path=XFG_root_path,
                                         split_token=split_token)
        testcaseids_done: List = [
            testcaseid
            for testcaseid in tqdm(
                # 使用pool.imap_unordered方法并行处理测试用例，
                # 每个测试用例调用process_func函数进行处理，将处理结果添加到testcaseids_done列表中。
                pool.imap_unordered(process_func, testcaseids),
                desc=f"testcases: ",
                total=testcase_len,
            )
        ]
        # 向消息队列中放入一个表示处理结束的消息。
        message_queue.put(QueueMessage(None, None, False, True))
        # 关闭进程池并等待所有子进程执行完毕。
        pool.close()
        pool.join()
    # 输出总共处理的XFG图数量。最后，函数打印处理的XFG图数量，并返回结果。
    print(f"total {xfg_ct.get()} XFGs!")


# 这段代码的作用是根据配置文件的设置执行符号化代码的处理，并将处理后的XFG图进行唯一化和划分。
if __name__ == '__main__':
    # 解析命令行参数，包括配置文件路径。
    __arg_parser = configure_arg_parser()
    __args = __arg_parser.parse_args()
    # 使用OmegaConf.load加载配置文件，并将其转换为DictConfig对象。
    config = cast(DictConfig, OmegaConf.load(__args.config))
    # 调用seed_everything函数设置随机种子。
    seed_everything(config.seed, workers=True)
    # 调用add_symlines函数，传递配置中的数据集名称、数据文件夹路径和分词选项。
    add_symlines(config.dataset.name, config.data_folder, config.split_token)
    # 调用unique_data函数，传递配置中的数据集名称和数据文件夹路径，以获取唯一的XFG图路径列表。
    xfg_unique_paths = unique_data(config.dataset.name, config.data_folder)
    # 调用split_list函数，传递XFG图路径列表和数据文件夹路径下的数据集名称，
    # 将XFG图路径列表按照训练集、测试集、验证集的比例划分并保存为JSON文件。
    split_list(xfg_unique_paths, join(config.data_folder, config.dataset.name))
