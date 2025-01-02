import torch


def norm_adj_bu_dui_cheng(adj):
    # 计算入度
    in_degrees = torch.sum(adj, dim=0)  # 沿着第0维求和
    # D_in = torch.diag(in_degrees)  # 创建对角矩阵

    # 计算出度
    out_degrees = torch.sum(adj, dim=1)  # 沿着第1维求和
    # D_out = torch.diag(out_degrees)  # 创建对角矩阵

    # 处理零值，避免 0^(-0.5) 导致 NaN
    in_degrees = in_degrees + (in_degrees == 0).float() * 1e-10
    out_degrees = out_degrees + (out_degrees == 0).float() * 1e-10

    # 计算 -0.5 次方的矩阵
    D_in_norm = torch.diag(torch.pow(in_degrees, -0.5))
    D_out_norm = torch.diag(torch.pow(out_degrees, -0.5))

    # 左乘 D_out_neg_half 和右乘 D_in_neg_half
    result = torch.mm(D_out_norm, torch.mm(adj, D_in_norm))

    return result

def norm_adj(adj):
    # 计算入度
    degrees = torch.sum(adj, dim=0)  # 沿着第0维求和
    # D = torch.diag(degrees)  # 创建对角矩阵

    # 处理零值，避免 0^(-0.5) 导致 NaN
    in_degrees = degrees + (degrees == 0).float() * 1e-10

    # 计算 -0.5 次方的矩阵
    D_norm = torch.diag(torch.pow(in_degrees, -1))

    # 左乘 D_out_neg_half 和右乘 D_in_neg_half
    result = torch.mm(D_norm, adj)

    return result
