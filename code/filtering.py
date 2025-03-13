# from graph_operation import match_bfs
import subprocess
from collections import defaultdict
from copy import deepcopy
import os

encoding = 'utf-8'


class Filtering:
    def __init__(self, pattern, data_graph):
        # data information contains:
        # 0: id 1: label 2: degree 3: edge_info 4: edge_label 5: vertex neighbor 6: label_dict
        self.pattern = pattern
        self.data_graph = data_graph

    def GQL_filter(self):
        # generate the candidate set using GraphQL.

        # get candidates by NLF, as initialization.
        local_candidates, candidate_count = self.generate_general_candidates()
        invalid_vertex_id = -1

        # basic information
        query_vertex_num = len(self.pattern[0])
        data_vertex_num = len(self.data_graph[0])
        query_max_degree = max(self.pattern[2])
        data_max_degree = max(self.data_graph[2])

        # generate valid candidate list
        valid_candidate = list()
        for i in range(query_vertex_num):
            temp_list = [False] * data_vertex_num
            for v in local_candidates[i]:
                temp_list[v] = True
            valid_candidate.append(deepcopy(temp_list))

        # global refinement
        for l in range(2):
            for i in range(query_vertex_num):
                query_vertex = i
                for j in range(candidate_count[i]):
                    data_vertex = local_candidates[i][j] #查询顶点i的第j个候选人
		    #如果这个点被标记为无效候选者，直接跳过
                    if data_vertex == invalid_vertex_id:
                        continue
                    if not self.verify_exact_twig_iso(query_vertex, data_vertex, query_vertex_num, data_vertex_num,
                                                      query_max_degree, data_max_degree, valid_candidate):
                        local_candidates[query_vertex][j] = invalid_vertex_id
                        valid_candidate[query_vertex][data_vertex] = False

        candidates, candidate_count = self.compact_candidate(local_candidates, query_vertex_num)

        return candidates, candidate_count

    def generate_general_candidates(self):
        # 0: id 1: label 2: degree 3: edge_info 4: edge_label 5: vertex neighbor 6: label_dict
        # generate the candidate using NLF.
        p_label = self.pattern[1] #查询图的标签
        p_degree = self.pattern[2]#查询图的度
        g_degree = self.data_graph[2]#数据图的度
        candidates = list() #空的候选列表
        candidate_count = [0] * len(p_label) #创建了一个名为 candidate_count 的列表，其中包含了与 p_label 中标签数量相同的元素。每个元素的初始值都是 0
        for i in range(len(p_label)):    # num of candidate vertices.
            selected_label_vertices = self.get_vertices_by_label(p_label[i])   # select nodes with the same label
            temp_list = []
            for v in selected_label_vertices:
                if g_degree[v] >= p_degree[i]:                                 # check the degree.
                    # check NLF
                    if self.check_NLF(i, v):
                        temp_list.append(v)
                        candidate_count[i] += 1
            candidates.append(deepcopy(temp_list))

        return candidates, candidate_count

    def get_vertices_by_label(self, label):
        selected_vertices = self.data_graph[6][label]
        return selected_vertices

    def verify_exact_twig_iso(self, query_vertex, data_vertex, query_vertex_num, data_vertex_num,
                                                      query_max_degree, data_max_degree, valid_candidates):
        # construct the bipartite graph and determine whether it is valid.
        # 0: id 1: label 2: degree 3: edge_info 4: edge_label 5: vertex neighbor 6: label_dict
	#这个查询点的邻居
        q_neighbors = self.pattern[5][query_vertex] 
        # print(q_neighbors)
	#这个候选顶点的邻居
        d_neighbors = self.data_graph[5][data_vertex]
        # print(d_neighbors)
        left_partition_size = len(q_neighbors)
        right_partition_size = len(d_neighbors)

        # note that the initial might be wrong.
        left_to_right_offset = [0] * (query_max_degree+1)  # query_max_degree + 1 #用于存储左侧顶点到右侧顶点的偏移量
        left_to_right_edges = [None] * (query_max_degree * data_max_degree)   # query_max_degree * data_max_degree #用于存储从左侧顶点到右侧顶点的边
        left_to_right_match = [None] * query_max_degree   # query_max_degree #用于存储左侧顶点到右侧顶点的匹配结果

        # right_to_left_match = list()   # data_max_degree
        # match_visited = list()         # data_max_degree + 1
        # match_queue = list()           # query_vertex_num
        # match_previous = list()        # data_max_degree + 1

        # print(query_max_degree)
        # print(data_max_degree)
        # print(left_partition_size)
        # print(right_partition_size)
        # build the bipartite graph
        edge_count = 0
        for i in range(left_partition_size):
            query_vertex_neighbor = q_neighbors[i] #查询点的第i个邻居
            left_to_right_offset[i] = edge_count
            for j in range(right_partition_size):
                data_vertex_neighbor = d_neighbors[j]
                if valid_candidates[query_vertex_neighbor][data_vertex_neighbor]:
                    edge_count+=1
                    # print(edge_count)
                    left_to_right_edges[edge_count] = j
        left_to_right_offset[left_partition_size] = edge_count

        # check if it is a semi-perfect match, process the left_to_right_match, find the ones that are not matched.
        for i in range(left_partition_size):
            if left_to_right_match[i] is None and left_to_right_offset[i] != left_to_right_offset[i+1]:
                for j in range(left_to_right_offset[i], left_to_right_offset[i+1]):
                    if left_to_right_edges[j] is not None:
                        left_to_right_match[i] = left_to_right_edges[j]
                        break
        for i in range(left_partition_size):
            if left_to_right_match[i] is None:
                return False
	


        return True

    # def is_valid_candidate(self, ):
    def compact_candidate(self, local_candidates, query_vertex_num):
        new_candidates = list()
        new_candidate_count = [0] * query_vertex_num
        for i in range(query_vertex_num):
            query_vertx = i
            temp_list = []
            for j in range(len(local_candidates[query_vertx])):
                if local_candidates[query_vertx][j] != -1:
                    temp_list.append(local_candidates[query_vertx][j])
                    new_candidate_count[query_vertx] += 1
            new_candidates.append(deepcopy(temp_list))

        return new_candidates, new_candidate_count

    def check_NLF(self, query_vertex, data_vertex):
        query_neighbors = self.pattern[5][query_vertex]
        data_neighbors = self.data_graph[5][data_vertex]
        q_neigh_labels = [self.pattern[1][u] for u in query_neighbors]
        d_neigh_labels = [self.data_graph[1][v] for v in data_neighbors]
        q_label_frequency = defaultdict(lambda:0)
        d_label_frequency = defaultdict(lambda:0)
        for l in q_neigh_labels:
            q_label_frequency[l] += 1
        for l in d_neigh_labels:
            d_label_frequency[l] += 1
        for l in q_neigh_labels:
            # if the query label grequency is greater than data label frequency, return false
            # if there is no label in data (l not in the d_dict) the number is 0 (default dict)
            if q_label_frequency[l] > d_label_frequency[l]:
                return False
        return True

    def update_query(self, pattern_info):
        self.pattern = pattern_info

    def cpp_GQL(self, query_graph_file, data_graph_file):
        num_query_vertices = len(self.pattern[0])#查询顶点的数目
        # base_command = ['/data/hancwang/Scalable Neural Subgraph Counting/Related_work/SubgraphMatching-master/build/filter/SubgraphMatching.out', '-d', data_graph_file, '-q', query_graph_file, '-filter', 'GQL']
        # base_command = ['/data/hancwang/Scalable Neural Subgraph Counting/Related_work/SubgraphMatching-master/build_with_subgraph/filter/SubgraphMatching.out', '-d', data_graph_file, '-q', query_graph_file, '-filter', 'GQL']
        # output = subprocess.run(base_command, capture_output=True)
        # baseline_visit = output.stdout.decode(encoding).split('\n')
        # print(baseline_visit)
        #base_command = ['./SubgraphMatching-master/SubgraphMatching-master/build/matching/SubgraphMatching.out','-d', data_graph_file, '-q', query_graph_file, '-filter', 'GQL',"-order", "GQL","-engine", "GQL", "-num", "1"]
        #result = subprocess.run(base_command)

        print(os.getcwd())  # 打印当前工作目录
        #
        # if result.returncode == 0:
        #     print("命令执行成功")
        # else:
        #     print("命令执行失败，返回码为：", result.returncode)
        candidates = list()
        candidate_count = list()
        induced_subgraph_list = list()
        neighbor_offset = list()
        candidate_info = []
        number = query_graph_file.split('_')[-1].split('.')[0]

        # 构造相应的文件名
        candidates_file = f"/home/dell/wangru/NeurSC_not_cleaned/yeast_8_candidates_y/candidates_{number}.txt"


        with open(candidates_file, 'r') as file:
            baseline_visit = file.readlines()

        # with open('./candidates.txt', 'r') as file:
        #     # 读取每一行，存入列表
        #     baseline_visit = file.readlines()

        # 去除每一行末尾的换行符
        baseline_visit = [line.strip() for line in baseline_visit]

        for i in range(len(baseline_visit)):
            if 'Candidate set is:' in baseline_visit[i]:
                print(baseline_visit[i])
            elif  i>=1 and i < 2*num_query_vertices + 1:
                if i % 2 != 0:
                    candidate_info.append(int(baseline_visit[i]))
                    # 如果索引 i 是奇数，添加候选顶点列表
                else:
                    # 将字符串 '1 2' 转换为列表 [1, 2]
                    candidates = list(map(int, baseline_visit[i].split()))
                    candidate_info.append(candidates)
            elif 'Candidate set version:' in baseline_visit[i]:
                candidates = baseline_visit[i+1].split()
                for j in range(len(candidates)):
                    candidates[j] = int(candidates[j])
            elif 'Subgraph List is :' in baseline_visit[i]:
                induced_subgraph_list = baseline_visit[i+1].split()
                for j in range(len(induced_subgraph_list)):
                    induced_subgraph_list[j] = int(induced_subgraph_list[j])
            elif 'Offset is :' in baseline_visit[i]:
                neighbor_offset = baseline_visit[i+1].split()
                for j in range(len(neighbor_offset)):
                    neighbor_offset[j] = int(neighbor_offset[j])
            elif 'Filter vertices' in baseline_visit[i]:
                print(baseline_visit[i])
        # print(what_we_need)
        for i in range(len(candidate_info)):
            if i%2 != 0:
                candidate_count.append(len(candidate_info[i]))
        
        return candidates, candidate_count, induced_subgraph_list, neighbor_offset, candidate_info
