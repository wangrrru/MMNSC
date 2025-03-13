import torch
import numpy as np
import torch.nn as nn
import torch_geometric
import torch_geometric.nn as geo_nn
import torch.nn.functional as F
from gnn import GIN, GAT , GCN


class BasicCountNet(nn.Module):
    def __init__(self, input_feat_dim, query_hidden_dim, data_hidden_dim, out_dim, pooling_method='sumpool', share_net=False):
        super(BasicCountNet, self).__init__()
        self.pool_method = pooling_method
        if not share_net:
            self.query_GNN = GIN(input_feat_dim, query_hidden_dim, out_dim)
            self.data_GNN = GIN(input_feat_dim, data_hidden_dim, out_dim)
        else:
            self.query_GNN = self.data_GNN = GIN(input_feat_dim, data_hidden_dim, out_dim)
        self.linear_layers = nn.Sequential(
            nn.Linear(2* out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim//2),
            nn.ReLU(),
            nn.Linear(out_dim//2, 8),
            nn.ReLU()
        )



    def pool_operation(self, x):
        # build batch
        num_nodes = x.size(0)
        batch = torch.from_numpy(np.zeros(num_nodes)).type(torch.LongTensor)
        if self.pool_method == 'sumpool':
            return geo_nn.global_add_pool(x, batch)
        elif self.pool_method == 'meanpool':
            return geo_nn.global_mean_pool(x, batch)
        elif self.pool_method == 'maxpool':
            return geo_nn.global_max_pool(x, batch)
        else:
            raise NotImplementedError

    def forward(self, query_in_feat, data_in_feat, query_edge_list, data_edge_list,query2data_edge_list=None):


        query_x = self.query_GNN(query_in_feat, query_edge_list)
        data_x = self.data_GNN(data_in_feat, data_edge_list)
        #cos_sim = torch.cosine_similarity(query_x[:, None, :], data_x[None, :, :], dim=-1)
        #cos_sim = torch.where(cos_sim < 0, torch.zeros_like(cos_sim), cos_sim)
        query_x = self.pool_operation(query_x)
        data_x = self.pool_operation(data_x)
        out_feat = torch.cat((query_x, data_x), dim=1)
        pred = self.linear_layers(out_feat)
        pred = pred.view(1, 8)
        #result = cos_sim * pred

        return pred



class AttentiveCountNet(nn.Module):
    def __init__(self, input_feat_dim, query_hidden_dim, data_hidden_dim, out_dim, pooling_method='sumpool', share_net=False):
        super(AttentiveCountNet, self).__init__()
        self.pool_method = pooling_method
        if not share_net:
            self.query_GNN = GIN(input_feat_dim, query_hidden_dim, out_dim)
            self.data_GNN = GIN(input_feat_dim, data_hidden_dim, out_dim)
        else:
            self.query_GNN = self.data_GNN = GIN(input_feat_dim, data_hidden_dim, out_dim)
        self.attention_layer = GAT(input_feat_dim, out_dim)
        self.linear_layers = nn.Sequential(
            nn.Linear(4* out_dim, 2*out_dim),
            nn.Linear(2* out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim//2),
            nn.ReLU(),
            nn.Linear(out_dim//2, 8*4674),
            nn.ReLU()
        )

    def pool_operation(self, x):
        # build batch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_nodes = x.size(0)
        batch = torch.from_numpy(np.zeros(num_nodes)).type(torch.LongTensor).to(device)
        if self.pool_method == 'sumpool':
            return geo_nn.global_add_pool(x, batch)
        elif self.pool_method == 'meanpool':
            return geo_nn.global_mean_pool(x, batch)
        elif self.pool_method == 'maxpool':
            return geo_nn.global_max_pool(x, batch)
        else:
            raise NotImplementedError

    def forward(self, query_in_feat, data_in_feat, query_edge_list, data_edge_list, query2data_edge_list):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        query_x = self.query_GNN(query_in_feat, query_edge_list).to(device)
        data_x = self.data_GNN(data_in_feat, data_edge_list).to(device)
        query_x_1 = query_x.to(device)
        data_x_2 = data_x.to(device)
        num_query_vertices = query_x.shape[0]
        query2data_in_feat = torch.cat((query_in_feat, data_in_feat), dim=0).to(device)
        query2data_x = self.attention_layer(query2data_in_feat, query2data_edge_list).to(device)
        query_x_with_data = query2data_x[:num_query_vertices, :].to(device)
        data_x_with_query = query2data_x[num_query_vertices:, :].to(device)
        out_query_x = torch.cat((query_x, query_x_with_data), dim=1).to(device)
        out_data_x = torch.cat((data_x, data_x_with_query),dim=1).to(device)
        query_x = self.pool_operation(out_query_x).to(device)
        data_x = self.pool_operation(out_data_x).to(device)
        out_feat = torch.cat((query_x, data_x), dim=1)
        # note that we can change the scale function.
        pred = self.linear_layers(out_feat).to(device)
        pred = pred.view(8, 4674).to(device)
        return pred, query_x_1, data_x_2

class AttentiveCountNet2(nn.Module):
    def __init__(self, input_feat_dim, query_hidden_dim, data_hidden_dim, out_dim, pooling_method='sumpool', share_net=False):
        super(AttentiveCountNet2, self).__init__()
        self.pool_method = pooling_method
        if not share_net:
            self.query_GNN = GIN(input_feat_dim, query_hidden_dim, out_dim)
            self.data_GNN = GIN(input_feat_dim, data_hidden_dim, out_dim)
        else:
            self.query_GNN = self.data_GNN = GIN(input_feat_dim, data_hidden_dim, out_dim)
        self.attention_layer = GAT(input_feat_dim, out_dim)
        self.linear_layers = nn.Sequential(
            nn.Linear(4* out_dim, 2*out_dim),
            nn.Linear(2* out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim//2),
            nn.ReLU(),
            nn.Linear(out_dim//2, 1),
            nn.ReLU()
        )

    def pool_operation(self, x):
        # build batch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_nodes = x.size(0)
        batch = torch.from_numpy(np.zeros(num_nodes)).type(torch.LongTensor).to(device)
        if self.pool_method == 'sumpool':
            return geo_nn.global_add_pool(x, batch)
        elif self.pool_method == 'meanpool':
            return geo_nn.global_mean_pool(x, batch)
        elif self.pool_method == 'maxpool':
            return geo_nn.global_max_pool(x, batch)
        else:
            raise NotImplementedError

    def forward(self, query_in_feat, data_in_feat, query_edge_list, data_edge_list, query2data_edge_list):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        query_x = self.query_GNN(query_in_feat, query_edge_list).to(device)
        data_x = self.data_GNN(data_in_feat, data_edge_list).to(device)
        query_x_1 = query_x.to(device)
        data_x_2 = data_x.to(device)
        num_query_vertices = query_x.shape[0]
        data_vertices = data_x.shape[0]
        query2data_in_feat = torch.cat((query_in_feat, data_in_feat), dim=0).to(device)
        query2data_x = self.attention_layer(query2data_in_feat, query2data_edge_list).to(device)
        query_x_with_data = query2data_x[:num_query_vertices, :].to(device)
        data_x_with_query = query2data_x[num_query_vertices:, :].to(device)
        out_query_x = torch.cat((query_x, query_x_with_data), dim=1).to(device)
        out_data_x = torch.cat((data_x, data_x_with_query),dim=1).to(device)
        #padding = torch.zeros(862664-data_vertices, 128).to(device)
        # Concatenating the original tensor with the padding
        #out_data_x_padded = torch.cat((out_data_x, padding), dim=0)
        query_expanded = out_query_x.unsqueeze(1).repeat(1, out_data_x.size(0), 1)  # [4, 100, 64]
        data_expanded = out_data_x.unsqueeze(0).repeat(out_query_x.size(0), 1, 1)  # [4, 100, 64]
        # 拼接后一次性通过网络预测
        x = torch.cat((query_expanded, data_expanded), dim=-1)  # [4, 100, 128]
        pred = self.linear_layers(x).squeeze(-1).to(device)

        return pred, query_x_1, data_x_2

class CrossGraphMatchingModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(CrossGraphMatchingModel, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_layers = num_layers
        self.linear_layers = nn.Sequential(
            nn.Linear(4 * output_dim, 2 * output_dim),
            nn.Linear(2 * output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim // 2),
            nn.ReLU(),
            nn.Linear(output_dim // 2, 1),
            nn.ReLU()
        )

        # 定义图内卷积层并移动到指定设备
        self.query_gnn_layers = torch.nn.ModuleList(
            [GCN(input_dim if i == 0 else hidden_dim, input_dim).to(device) for i in range(num_layers)]
        )
        self.data_gnn_layers = torch.nn.ModuleList(
            [GCN(input_dim if i == 0 else hidden_dim, input_dim).to(device) for i in range(num_layers)]
        )
    def forward(self, query_features, data_features, query_edge_index, data_edge_index, query2data_edge_list):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for layer in range(self.num_layers):
            query_features = F.relu(self.query_gnn_layers[layer](query_features, query_edge_index).to(device)).to(device)
            data_features = F.relu(self.data_gnn_layers[layer](data_features, data_edge_index).to(device)).to(device)
            # 获取查询节点和数据节点的索引列表
            query_nodes = query2data_edge_list[0].to(device)  # 查询节点索引
            data_nodes = query2data_edge_list[1] .to(device) # 数据节点索引

            # 查询-数据节点对的特征提取
            query_selected = query_features[query_nodes].to(device) # 获取与数据节点相连的查询节点特征
            data_selected = data_features[data_nodes].to(device)  # 获取与查询节点相连的数据节点特征

            # 计算注意力权重（例如使用余弦相似度）
            attention_scores =F.cosine_similarity(query_selected, data_selected, dim=1).to(device)  # 计算相似度
            attention_weights = F.softmax(attention_scores, dim=0).to(device)  # 转换为权重

            # 1. 跨图卷积：将数据节点的信息加权聚合到查询节点
            aggregated_to_query = torch.zeros_like(query_features).to(query_features.device)
            aggregated_to_query.scatter_add_(
                0, query_nodes.unsqueeze(-1).expand(-1, data_features.size(1)),
                attention_weights.unsqueeze(1) * data_selected
            )

            # 2. 反向跨图卷积：将查询节点的信息加权聚合到数据节点
            aggregated_to_data = torch.zeros_like(data_features).to(data_features.device)
            aggregated_to_data.scatter_add_(
                0, (data_nodes).unsqueeze(-1).expand(-1, query_features.size(1)),
                attention_weights.unsqueeze(1) * query_selected
            )

            # 将图内卷积的特征与跨图卷积结果结合
            query_features = torch.cat((query_features, aggregated_to_query), dim=1).to(device)
            data_features = torch.cat((data_features, aggregated_to_data), dim=1).to(device)
        # 返回最后的查询图和数据图嵌入
        # Concatenating the original tensor with the padding
        query_expanded = query_features.unsqueeze(1).repeat(1, data_features.size(0), 1)  # [4, 100, 64]
        data_expanded = data_features.unsqueeze(0).repeat(query_features.size(0), 1, 1)  # [4, 100, 64]
        # 拼接后一次性通过网络预测
        x = torch.cat((query_expanded, data_expanded), dim=-1)  # [4, 100, 128]
        # note that we can change the scale function.
        pred = self.linear_layers(x).squeeze(-1).to(device)

        return pred,query_features, data_features




class WasserstainDiscriminator(nn.Module):
    def __init__(self, hidden_dim, hidden_dim2=512):
        super(WasserstainDiscriminator, self).__init__()
        self.linear_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(hidden_dim2, hidden_dim2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(hidden_dim2, 1)
        )
    
    def forward(self, input_x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_x = input_x.to(device)
        wq = self.linear_layers(input_x).to(device)
        return wq

class CustomLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weight = self.weight.to(device)
        if self.bias is not None:
            self.bias = self.bias.to(device)
        return nn.functional.linear(input, self.weight, self.bias)

class QErrorLoss:
    def __init__(self):
        pass

    def __call__(self, input_card, true_card):
        q_error = torch.max(torch.cat(((torch.max(torch.cat((input_card.unsqueeze(0), torch.tensor([1]))))/torch.max(torch.cat((true_card.unsqueeze(0), torch.tensor([1]))))).unsqueeze(0), (torch.max(torch.cat((true_card.unsqueeze(0), torch.tensor([1]))))/torch.max(torch.cat((input_card.unsqueeze(0), torch.tensor([1]))))).unsqueeze(0))))
        return q_error


class QErrorLikeLoss:
    def __init__(self, epsilon=1e-9):
        self.epsilon = epsilon
        self.mse_loss = nn.MSELoss()

    def __call__(self, input_card, true_card, pred, ground_truth, diff):
        q_error = torch.max(torch.cat(((true_card/(input_card+self.epsilon)).unsqueeze(0), (input_card/(true_card+self.epsilon)).unsqueeze(0))))
        mse_loss = self.mse_loss(pred, ground_truth)
        diff_loss = diff.mean()

        return q_error,mse_loss, diff_loss


class CoarsenNet(nn.Module):
    def __init__(self, input_feat_dim, query_hidden_dim, data_hidden_dim, out_dim, pooling_method='sumpool', share_net=False):
        super(CoarsenNet, self).__init__()
        self.pool_method = pooling_method
        if not share_net:
            self.query_GNN = GIN(input_feat_dim, query_hidden_dim, out_dim)
            self.data_GNN = GIN(input_feat_dim, data_hidden_dim, out_dim)
        else:
            self.query_GNN = self.data_GNN = GIN(input_feat_dim, data_hidden_dim, out_dim)
        self.linear_layers = nn.Sequential(
            nn.Linear(2* out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim//2),
            nn.ReLU(),
            nn.Linear(out_dim//2, 1),
            nn.ReLU()
        )
