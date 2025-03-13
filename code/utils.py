import torch
import os
import numpy as np
import signal
import time
import subprocess
import torch_geometric
import torch_geometric.nn as geo_nn
from copy import deepcopy
from tqdm import tqdm
from functools import wraps
from collections import defaultdict


def load_g_graph(g_file):
    nid = list()
    nlabel = list()
    nindeg = list()
    elabel = list()
    e_u = list()
    e_v = list()
    with open(g_file) as f2:
        ch,num_nodes,num_edges =f2.readline().rstrip().split()
        num_nodes = int(num_nodes)
        num_edges = int(num_edges)
        v_neigh = list()
        for i in range(num_nodes):
            temp_list = list()
            v_neigh.append(temp_list)
            node_info = f2.readline()
            type ,node_id, node_label , ndeg = node_info.rstrip().split()
            nid.append(int(node_id))
            nlabel.append(int(node_label))
            nindeg.append(int(ndeg))
            # read until the end of the file.
        for i in range( num_edges):
                #type,eu,ev,label = f2.readline().rstrip().split()  #eu2005 :4 yeast:3 human:3 #dblp:4 youtube:4
                type, eu, ev = f2.readline().rstrip().split()
                elabel.append(1)
                e_u.append(int(eu))
                e_v.append(int(ev))
                v_neigh[int(eu)].append(int(ev))
                v_neigh[int(ev)].append(int(eu))
    g_nid = deepcopy(nid)
    g_nlabel = deepcopy(nlabel)
    g_indeg = deepcopy(nindeg)
    g_edges = [deepcopy(e_u), deepcopy(e_v)]
    g_e_label = deepcopy(elabel)
    g_v_neigh = deepcopy(v_neigh)
    g_label_dict = defaultdict(list)
    for i in range(len(g_nlabel)):
        g_label_dict[g_nlabel[i]].append(i)
    graph_info = [
        g_nid,
        g_nlabel,
        g_indeg,
        g_edges,
        g_e_label,
        g_v_neigh,
        g_label_dict
    ]
    print(graph_info)
    return graph_info


def load_p_data(p_file):
    nid = list()
    nlabel = list()
    nindeg = list()
    e_u = list()
    elabel = list()
    e_v = list()

    with open(p_file) as f1:
        ch,num_nodes,num_edges =f1.readline().rstrip().split()
        num_nodes = int(num_nodes)
        num_edges = int(num_edges)
        v_neigh = list()
        for i in range(num_nodes):
            temp_list = list()
            v_neigh.append(temp_list)
            node_info = f1.readline()
            type, node_id, node_label, ndeg = node_info.rstrip().split()
            nid.append(int(node_id))
            nlabel.append(int(node_label))
            nindeg.append(int(ndeg))
        for i in range(num_edges):
                type,eu,ev,label = f1.readline().rstrip().split()
                #type, eu, ev = f1.readline().rstrip().split()
                elabel.append(1)
                e_u.append(int(eu))
                e_v.append(int(ev))
                v_neigh[int(eu)].append(int(ev))
                v_neigh[int(ev)].append(int(eu))
    p_nid = deepcopy(nid)
    p_nlabel = deepcopy(nlabel)
    p_indeg = deepcopy(nindeg)
    p_edges = [deepcopy(e_u), deepcopy(e_v)]
    p_elabel = deepcopy(elabel)
    p_v_neigh = deepcopy(v_neigh)
    p_label_dict = defaultdict(list)
    for i in range(len(p_nlabel)):
        p_label_dict[p_nlabel[i]].append(i)
    pattern_info = [
        p_nid,
        p_nlabel,
        p_indeg,
        p_edges,
        p_elabel,
        p_v_neigh,
        p_label_dict
    ]
    return pattern_info


def load_baseline(b_file):
    baseline_dict = dict()
    with open(b_file) as f:
        for line in f:
            file, true_count = line.rstrip().split()
            baseline_dict[file] = int(true_count)
    return baseline_dict

def load_ground_truth(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        ground_truth = [list(map(int, line.strip().split())) for line in lines[0:]]
        ground_truth = torch.tensor(ground_truth, dtype=torch.float)
    return ground_truth

def load_iso_baseline(baseline_path, prefix):
    baseline_dict = dict()
    file_paths = os.listdir(baseline_path)
    for file_path in file_paths:
        current_path = os.path.join(baseline_path, file_path)
        file_list = os.listdir(current_path)
        for file_name in file_list:
            current_file = os.path.join(current_path, file_name)
            with open(current_file) as f:
                line_0 = f.readline().rstrip()
                baseline_dict[prefix+file_name.replace('.txt', '.graph')] = int(line_0)
    return baseline_dict


def int_to_multihot(input_int, dim):
    init = np.zeros(dim)
    binary_string = '{0:b}'.format(input_int)
    diff = dim - len(binary_string)
    for i in reversed(range(len(binary_string))):
        init[i+diff] = int(binary_string[i])
    return init


def generate_features(graph_info, vec_dim):
    # data information contains:
    # 0: id 1: label 2: degree 3: edge_info 4:e_label 5: vertex neighbor 6: label_dict
    vertices_id = graph_info[0]
    label_info = graph_info[1]
    degree_info = graph_info[2]
    neighbor_info = graph_info[5]
    feature_vec = np.array([])
    for i in range(len(label_info)):    # num of graph vertices
        label_vec = int_to_multihot(label_info[vertices_id[i]], vec_dim)
        degree_vec = int_to_multihot(degree_info[vertices_id[i]], vec_dim)
        neigh_label_vec = np.expand_dims(np.zeros(vec_dim), axis=0)
        neigh_degree_vec = np.expand_dims(np.zeros(vec_dim), axis=0)
        for j in range(len(neighbor_info[vertices_id[i]])):
            if j == 0:
                neigh_label_vec = np.expand_dims(int_to_multihot(label_info[neighbor_info[vertices_id[i]][j]], vec_dim), axis=0)
                neigh_degree_vec = np.expand_dims(int_to_multihot(degree_info[neighbor_info[vertices_id[i]][j]], vec_dim), axis=0)
            else:
                neigh_label_vec = np.append(neigh_label_vec, np.expand_dims(int_to_multihot(label_info[neighbor_info[vertices_id[i]][j]], vec_dim), axis=0),axis=0)
                neigh_degree_vec = np.append(neigh_degree_vec, np.expand_dims(int_to_multihot(degree_info[neighbor_info[vertices_id[i]][j]], vec_dim), axis=0),axis=0)
        # print(neigh_label_vec)
        neigh_label_vec = np.mean(neigh_label_vec, axis=0)
        neigh_degree_vec = np.mean(neigh_degree_vec, axis=0)
        # print(neigh_label_vec)
        current_feat = np.concatenate((label_vec, degree_vec, neigh_label_vec, neigh_degree_vec), axis=0)
        # print(current_feat)
        if i == 0:
            feature_vec = np.expand_dims(current_feat, axis=0)
        else:
            feature_vec = np.append(feature_vec, np.expand_dims(current_feat, axis=0), axis=0)
    return torch.from_numpy(feature_vec).type(torch.FloatTensor)


def preprocess_data_edge(vertices, origin_edge_list):
    vertices_dict = dict()
    new_e_u = list()
    new_e_v = list()
    for i in range(len(vertices)):
        vertices_dict[vertices[i]] = i
    for i in range(len(origin_edge_list[0])):
        new_e_u.append(vertices_dict[origin_edge_list[0][i]])
        new_e_v.append(vertices_dict[origin_edge_list[1][i]])
    return [new_e_u, new_e_v]

def preprocess_query2data(sub_vertices, candidate_info):
    total_len = len(candidate_info)
    num_query_vertex = total_len/2
    vertices_dict = dict()
    new_e_u = list()
    new_e_v = list()
    for i in range(len(sub_vertices)):
        vertices_dict[sub_vertices[i]] = i
    for i in range(total_len):
        if i%2 != 0:
            candidate_list = candidate_info[i]
            query_vertex = i//2
            for data_vertex in candidate_list:
                data_vertex = int(data_vertex)
                if data_vertex in sub_vertices:
                    new_e_u.append(query_vertex)
                    new_e_v.append(vertices_dict[data_vertex])
            # try:
            #     candidate_list = candidate_info[i+2].split()
            #     data_vertex = int(candidate_list[0])
            #     new_e_u.append(query_vertex)
            #     new_e_v.append(vertices_dict[data_vertex])
            # except IndexError:
            #     continue
            #
    return [new_e_u, new_e_v]

def preprocess_data2query(neighbors, sub_vertices, candidate_info):
    #vertices_dict = {sub_vertices[i]: i for i in range(len(sub_vertices))}
    total_len = len(candidate_info)
    num_query_vertex = total_len / 2
    vertices_dict = dict()
    new_e_u = list()
    new_e_v = list()
    for i in range(len(sub_vertices)):
        vertices_dict[sub_vertices[i]] = i + num_query_vertex

    for query_vertex in range(len(neighbors)):
        for neighbor in neighbors[query_vertex]:
            candidate_list = candidate_info[2 * neighbor + 1]

            for data_vertex in candidate_list:
                data_vertex = int(data_vertex)
                if data_vertex in sub_vertices:
                    new_e_u.append(query_vertex)
                    new_e_v.append(vertices_dict[data_vertex])

    return [new_e_u, new_e_v]

#
# def preprocess_data2query(order, sub_vertices, candidate_info):
#     order = list(map(int, order))[::-1]  # 将order转换为整数列表并反转
#     vertices_dict = dict()
#     for i in range(len(sub_vertices)):
#         # vertices_dict[sub_vertices[i]] = i + num_query_vertex
#         vertices_dict[sub_vertices[i]] = sub_vertices[i]
#     new_e_u = []
#     new_e_v = []
#
#     for i in range(len(order) - 1):
#         query_vertex = order[i]
#         next_vertex = order[i + 1]
#         candidate_list = candidate_info[2 * next_vertex + 1]
#
#         for data_vertex in candidate_list:
#             data_vertex = int(data_vertex)
#             if data_vertex in sub_vertices:
#                 new_e_u.append(query_vertex)
#                 new_e_v.append(vertices_dict[data_vertex])
#
#     return [new_e_u, new_e_v]

def create_vertices_degree_dict(new_vertices, new_v_neigh_degree):
    # 使用字典推导式来创建一个新的字典
    vertices_degree_dict = {vertex: degree for vertex, degree in zip(new_vertices, new_v_neigh_degree)}
    return vertices_degree_dict

def create_list_from_dicts(dicts, length=3112):
    # 创建一个长度为length，初始值为0的列表
    result_list = [0] * length

    # 遍历字典中的键和值
    for key, value in dicts.items():
        # 如果键是有效的索引，则更新列表
        if 0 <= key < length:
            result_list[key] = value

    return result_list

def calculate_normalized_degrees(candidate, vertices_degree_dict):
    # 初始化度数总和为0
    total_degree = 0
    # 初始化一个新的字典来存储存在于vertices_degree_dict中的候选者的度数
    existing_degrees = {}

    # 遍历 candidate[0] 中的每个候选者
    for cand in candidate:
        # 如果候选者的ID在vertices_degree_dict中存在
        if cand in vertices_degree_dict:
            # 累加度数总和
            total_degree += vertices_degree_dict[cand]
            # 将存在的候选者的度数添加到字典中
            existing_degrees[cand] = vertices_degree_dict[cand]

    # 创建一个新的字典，其中键是候选者ID，值是归一化的度
    normalized_degrees = {cand: degree / total_degree for cand, degree in existing_degrees.items()}

    return normalized_degrees




def calculate_neigh_degrees(subgraph_info):
    # 初始化 new_v_neigh_degree 列表
    new_v_neigh_degree = []

    # 遍历 subgraph_info 中的每个顶点
    for i in range(len(subgraph_info[0])):
        # 获取顶点的度
        j = subgraph_info[0][i]
        v_degree = subgraph_info[2][j]
        # 获取顶点的所有邻居
        v_neigh = subgraph_info[5][j]

        # 计算顶点的度和所有邻居的度的总和
        total_degree = v_degree + sum(subgraph_info[2][neigh] for neigh in v_neigh)


            # 将计算结果添加到列表中
        new_v_neigh_degree.append(total_degree)



    return new_v_neigh_degree


import os

def save_params(file_position, args):
    os.makedirs(os.path.dirname(file_position), exist_ok=True)

    with open(file_position, 'w') as f:
        f.write('input feat dim:' + str(args.in_feat) +'\n')
        f.write('hidden dim: '+str(args.hidden_dim)+ '\n')
        f.write('output dim: '+str(args.out_dim) +'\n')
        f.write('learning rate: '+str(args.learning_rate) +'\n')
        f.write('epochs: '+str(args.num_epoch) +'\n')
        f.write('model: '+str(args.model_name) +'\n')
        f.write('training ratio: ' + str(args.train_percent) +'\n')
        f.write('train method: '+ str(args.train_method) + '\n')
        f.write('sample method: '+str(args.sample_method)+'\n')

