from torch_geometric.data.data import Data
import torch

'''i = [[0, 1, 1, 2],
     [2, 0, 2, 1]]

v =  [1, 1, 1, 1]
s = torch.sparse_coo_tensor(i, v)
print(s.to_dense())'''

# 假设 edge_index 是你的边索引张量
edge_index = torch.tensor([[0, 0, 1, 1, 1],
                           [2, 3, 0, 1, 2]])

# 创建对应的边权重值，这里统一设置为1
edge_values = torch.ones(edge_index.shape[1])

# 创建稀疏张量
sparse_tensor = torch.sparse_coo_tensor(edge_index, edge_values, torch.Size([2, 4]))

# 输出稀疏张量的信息
print(sparse_tensor)
print(sparse_tensor.shape)

#---------------------------------
# pytorch中稀疏矩阵edge_index转稠密矩阵
adj = torch.tensor([[0, 1, 2], [1, 2, 3]])

num_nodes = adj.max() + 1
num_edges = adj.shape[1]

# 构建稀疏矩阵的坐标和值
indices = adj
values = torch.ones(num_edges)

# 使用 torch.sparse_coo_tensor 创建稀疏矩阵
sparse_tensor = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes))

# 将稀疏矩阵转换为稠密矩阵
adj = sparse_tensor.to_dense()
adj = adj.float()
print(sparse_tensor)
print(adj)