import hashlib
from warnings import filterwarnings
import subprocess

from sklearn.model_selection import train_test_split
from typing import List, Union, Dict, Tuple
import numpy
import torch
import json
import os
import networkx as nx
from os.path import exists
from tqdm import tqdm

PAD = "<PAD>"
UNK = "<UNK>"
MASK = "<MASK>"
BOS = "<BOS>"
EOS = "<EOS>"


def getMD5(s):
    '''
    得到字符串s的md5加密后的值

    :param s:
    :return:
    '''
    hl = hashlib.md5()
    hl.update(s.encode("utf-8"))
    return hl.hexdigest()


# filter_warnings函数用于过滤特定的警告消息，以避免在训练过程中产生过多的警告信息。
def filter_warnings():
    # "The dataloader does not have many workers which may be a bottleneck."
    # 使用filterwarnings函数来设置要过滤的警告类别和相关的模块及行号信息。
    filterwarnings("ignore",
                   category=UserWarning,
                   module="pytorch_lightning.trainer.data_loading",
                   lineno=102)
    filterwarnings("ignore",
                   category=UserWarning,
                   module="pytorch_lightning.utilities.data",
                   lineno=41)
    # "Please also save or load the state of the optimizer when saving or loading the scheduler."
    # 第216行和第234行产生的UserWarning警告会被忽略。
    filterwarnings("ignore",
                   category=UserWarning,
                   module="torch.optim.lr_scheduler",
                   lineno=216)  # save
    filterwarnings("ignore",
                   category=UserWarning,
                   module="torch.optim.lr_scheduler",
                   lineno=234)  # load
    filterwarnings("ignore",
                   category=DeprecationWarning,
                   module="pytorch_lightning.metrics.__init__",
                   lineno=43)
    filterwarnings("ignore",
                   category=UserWarning,
                   module="torch._tensor",
                   lineno=575)
    filterwarnings("ignore",
                   category=UserWarning,
                   module="src.models.modules.common_layers",
                   lineno=0)

# 计算文件中的行数。
def count_lines_in_file(file_path: str) -> int:
    # 使用subprocess.run函数执行Shell命令wc -l <file_path>，并将输出捕获为字符串。
    command_result = subprocess.run(["wc", "-l", file_path],
                                    capture_output=True,
                                    encoding="utf-8")
    # 检查命令执行的返回代码returncode，如果不为0，则表示命令执行失败，抛出RuntimeError异常，并将错误信息包含在异常消息中。
    if command_result.returncode != 0:
        raise RuntimeError(
            f"Counting lines in {file_path} failed with error\n{command_result.stderr}"
        )
    # 将命令输出的字符串按空格进行分割，并取第一个元素（行数）转换为整数。
    return int(command_result.stdout.split()[0])

'''
    函数的实现步骤如下：
    创建一个空字典md5_dict，用于存储xfg的唯一标识符和相关信息。
    初始化计数器mul_ct和conflict_ct，用于统计多重和冲突的xfg。
    对于xfg路径列表中的每个xfg文件：
    使用nx.read_gpickle函数读取xfg文件内容为一个NetworkX图对象。
    获取xfg的标签和文件路径。
    使用open函数打开文件，并逐行读取文件内容，存储在file_contents中。
    遍历xfg的每个节点，计算相应行的内容的MD5哈希值，并将该值作为节点的一个属性存储。
    初始化一个空列表edges_md5，用于存储xfg的边的MD5哈希值。
    遍历xfg的每条边，将连接的两个节点的MD5哈希值拼接为一个字符串，并添加到edges_md5列表中。
    将edges_md5列表按字典顺序排序，并将其转换为字符串，计算该字符串的MD5哈希值作为xfg的唯一标识符xfg_md5。
    如果xfg_md5不在md5_dict中，将其添加到字典中，并关联其标签和xfg文件路径。
    否则，如果xfg_md5在md5_dict中，获取已存在的xfg的标签，并与当前xfg的标签进行比较：
    如果两个标签不相等且不为-1，则表示存在冲突，将冲突计数器conflict_ct加1，并将该xfg的标签设置为-1。
    否则，如果两个标签相等或其中一个标签为-1，则表示存在多个相同的xfg，将多重计数器mul_ct加1。
    打印冲突计数器和多重计数器的值。
    返回存储了xfg唯一标识符、标签和文件路径的md5_dict字典。
    通过调用unique_xfg_raw函数，可以处理xfg列表，并提取出唯一的xfg，同时统计冲突和多重的xfg情况。
    返回的md5_dict字典可以用于进一步处理唯一的xfg。
'''
# unique_xfg_raw函数用于从xfg列表中提取唯一的xfg。该函数遍历xfg文件的路径列表，并对每个xfg文件进行处理，
# 提取xfg的内容并计算其唯一标识符（使用MD5哈希算法）。然后，将xfg的唯一标识符与其标签进行关联，并将其存储在一个字典中。
def unique_xfg_raw(xfg_path_list):
    """f
    unique xfg from xfg list
    Args:
        xfg_path_list:

    Returns:
        md5_dict: {xfg md5:{"xfg": xfg_path, "label": 0/1/-1}}, -1 stands for conflict
    """
    md5_dict = dict()
    mul_ct = 0
    conflict_ct = 0

    for xfg_path in xfg_path_list:
        xfg = nx.read_gpickle(xfg_path)
        label = xfg.graph["label"]
        file_path = xfg.graph["file_paths"][0]
        assert exists(file_path), f"{file_path} not exists!"
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            file_contents = f.readlines()
        for ln in xfg:
            ln_md5 = getMD5(file_contents[ln - 1])
            xfg.nodes[ln]["md5"] = ln_md5
        edges_md5 = list()
        for edge in xfg.edges:
            edges_md5.append(xfg.nodes[edge[0]]["md5"] + "_" + xfg.nodes[edge[1]]["md5"])
        xfg_md5 = getMD5(str(sorted(edges_md5)))
        if xfg_md5 not in md5_dict:
            md5_dict[xfg_md5] = dict()
            md5_dict[xfg_md5]["label"] = label
            md5_dict[xfg_md5]["xfg"] = xfg_path
        else:
            md5_label = md5_dict[xfg_md5]["label"]
            if md5_label != -1 and md5_label != label:
                conflict_ct += 1
                md5_dict[xfg_md5]["label"] = -1
            else:
                mul_ct += 1
    print(f"total conflit: {conflict_ct}")
    print(f"total multiple: {mul_ct}")
    return md5_dict

'''
    unique_xfg_sym函数的实现步骤如下：
    创建一个空字典md5_dict，用于存储xfg的唯一标识符和相关信息。
    初始化计数器mul_ct和conflict_ct，用于统计多重和冲突的xfg。
    对于xfg路径列表中的每个xfg文件：
    使用nx.read_gpickle函数读取xfg文件内容为一个NetworkX图对象。
    获取xfg的标签和文件路径。
    使用循环遍历xfg的每个节点：
    获取节点的code_sym_token属性，并将其转换为字符串。
    计算该字符串的MD5哈希值，并将其作为节点的一个属性md5存储。
    初始化一个空列表edges_md5，用于存储xfg的边的MD5哈希值。
    遍历xfg的每条边，将连接的两个节点的MD5哈希值拼接为一个字符串，并添加到edges_md5列表中。
    将edges_md5列表按字典顺序排序，并将其转换为字符串，计算该字符串的MD5哈希值作为xfg的唯一标识符xfg_md5。
    如果xfg_md5不在md5_dict中，将其添加到字典中，并关联其标签和xfg文件路径。
    否则，如果xfg_md5在md5_dict中，获取已存在的xfg的标签，并与当前xfg的标签进行比较：
    如果两个标签不相等且不为-1，则表示存在冲突，将冲突计数器conflict_ct加1，并将该xfg的标签设置为-1。
    否则，如果两个标签相等或其中一个标签为-1，则表示存在多个相同的xfg，将多重计数器mul_ct加1。
    打印冲突计数器和多重计数器的值。
    返回存储了xfg唯一标识符、标签和文件路径的md5_dict字典。
    unique_xfg_sym函数在处理xfg时，额外对节点的code_sym_token属性进行了处理，并使用其内容计算MD5哈希值。
    这种处理方式可能与unique_xfg_raw函数在计算节点的唯一标识符时的方式不同。通过调用unique_xfg_sym函数，可以处理xfg列表，
    并提取出唯一的xfg，同时统计冲突和多重的xfg情况。返回的md5_dict字典可以用于进一步处理唯一的xfg。
'''
# unique_xfg_sym函数与之前介绍的unique_xfg_raw函数类似，用于从xfg列表中提取唯一的xfg。两者的主要区别在于处理xfg的方式不同
def unique_xfg_sym(xfg_path_list):
    """f
    unique xfg from xfg list
    Args:
        xfg_path_list:

    Returns:
        md5_dict: {xfg md5:{"xfg": xfg_path, "label": 0/1/-1}}, -1 stands for conflict
    """
    md5_dict = dict()
    mul_ct = 0
    conflict_ct = 0

    for xfg_path in tqdm(xfg_path_list, total=len(xfg_path_list), desc="xfgs: "):
        xfg = nx.read_gpickle(xfg_path)
        label = xfg.graph["label"]
        file_path = xfg.graph["file_paths"][0]
        assert exists(file_path), f"{file_path} not exists!"
        for ln in xfg:
            ln_md5 = getMD5(str(xfg.nodes[ln]["code_sym_token"]))
            xfg.nodes[ln]["md5"] = ln_md5
        edges_md5 = list()
        for edge in xfg.edges:
            edges_md5.append(xfg.nodes[edge[0]]["md5"] + "_" + xfg.nodes[edge[1]]["md5"])
        xfg_md5 = getMD5(str(sorted(edges_md5)))
        if xfg_md5 not in md5_dict:
            md5_dict[xfg_md5] = dict()
            md5_dict[xfg_md5]["label"] = label
            md5_dict[xfg_md5]["xfg"] = xfg_path
        else:
            md5_label = md5_dict[xfg_md5]["label"]
            if md5_label != -1 and md5_label != label:
                conflict_ct += 1
                md5_dict[xfg_md5]["label"] = -1
            else:
                mul_ct += 1
    print(f"total conflit: {conflict_ct}")
    print(f"total multiple: {mul_ct}")
    return md5_dict

# split_list函数用于将文件列表按照一定比例划分为训练集、测试集和验证集，并将划分结果保存为JSON文件。
def split_list(files: List[str], out_root_path: str):
    """

    Args:
        files:
        out_root_path:

    Returns:

    """
    #  使用train_test_split函数将文件列表files划分为训练集X_train和剩余部分。
    X_train, X_test = train_test_split(files, test_size=0.2)
    # 使用train_test_split函数将剩余部分划分为测试集X_test和验证集X_val，其中测试集占剩余部分的一半。
    X_test, X_val = train_test_split(X_test, test_size=0.5)
    # 如果输出根路径out_root_path不存在，则创建该路径。
    if not exists(f"{out_root_path}"):
        os.makedirs(f"{out_root_path}")
    # 使用json.dump函数将训练集X_train保存到out_root_path/train.json文件中。
    with open(f"{out_root_path}/train.json", "w") as f:
        json.dump(X_train, f)
    # 使用json.dump函数将测试集X_test保存到out_root_path/test.json文件中。
    with open(f"{out_root_path}/test.json", "w") as f:
        json.dump(X_test, f)
    # 使用json.dump函数将验证集X_val保存到out_root_path/val.json文件中。
    with open(f"{out_root_path}/val.json", "w") as f:
        json.dump(X_val, f)
