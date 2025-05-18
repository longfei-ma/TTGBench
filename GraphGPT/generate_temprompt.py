import pandas as pd
import numpy as np
import random
import argparse
import torch
import json
import copy
import os
import datetime
from collections import defaultdict, deque
from torch_geometric.utils.map import map_index
from torch_geometric.utils import to_undirected
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def set_random_seed(seed: int = 0):
    """
    set random seed
    :param seed: int, random seed
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Data:

    def __init__(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray, edge_ids: np.ndarray, labels: np.ndarray, seed: int = None):
        """
        Data object to store the nodes interaction information.
        :param src_node_ids: ndarray
        :param dst_node_ids: ndarray
        :param node_interact_times: ndarray
        :param edge_ids: ndarray
        :param labels: ndarray
        """
        self.src_node_ids = src_node_ids
        self.dst_node_ids = dst_node_ids
        self.node_interact_times = node_interact_times
        self.edge_ids = edge_ids
        self.labels = labels
        self.num_interactions = len(src_node_ids)
        self.unique_node_ids = set(src_node_ids) | set(dst_node_ids)
        self.num_unique_nodes = len(self.unique_node_ids)
        self.random_state = np.random.RandomState(seed)

    def __getitem__(self, index):
        # 通过索引访问元素
        return (self.src_node_ids[index].item(), self.dst_node_ids[index].item(), self.node_interact_times[index].item(), self.edge_ids[index].item())
    
    def random_edge(self):
        """
        随机返回一条动态边的信息
        :return: 元组 (src_node, dst_node, time)
        """
        idx = self.random_state.randint(0, len(self.src_node_ids))  # 随机选择一个索引
        return (self.src_node_ids[idx].item(), self.dst_node_ids[idx].item(), self.node_interact_times[idx].item(), self.edge_ids[idx].item())


class NegativeEdgeSampler(object):

    def __init__(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, interact_times: np.ndarray = None, last_observed_time: float = None,
                 negative_sample_strategy: str = 'random', seed: int = None):
        """
        Negative Edge Sampler, which supports three strategies: "random", "historical", "inductive".
        :param src_node_ids: ndarray, (num_src_nodes, ), source node ids, num_src_nodes == num_dst_nodes
        :param dst_node_ids: ndarray, (num_dst_nodes, ), destination node ids
        :param interact_times: ndarray, (num_src_nodes, ), interaction timestamps
        :param last_observed_time: float, time of the last observation (for inductive negative sampling strategy)
        :param negative_sample_strategy: str, negative sampling strategy, can be "random", "historical", "inductive"
        :param seed: int, random seed
        """
        self.seed = seed
        self.negative_sample_strategy = negative_sample_strategy
        self.src_node_ids = src_node_ids
        self.dst_node_ids = dst_node_ids
        self.interact_times = interact_times
        self.unique_src_node_ids = np.unique(src_node_ids)
        self.unique_dst_node_ids = np.unique(dst_node_ids)
        self.unique_interact_times = np.unique(interact_times)
        self.earliest_time = min(self.unique_interact_times)
        self.last_observed_time = last_observed_time

        if self.negative_sample_strategy != 'random':
            # all the possible edges that connect source nodes in self.unique_src_node_ids with destination nodes in self.unique_dst_node_ids
            self.possible_edges = set((src_node_id, dst_node_id) for src_node_id in self.unique_src_node_ids for dst_node_id in self.unique_dst_node_ids)

        if self.negative_sample_strategy == 'inductive':
            # set of observed edges
            self.observed_edges = self.get_unique_edges_between_start_end_time(self.earliest_time, self.last_observed_time)

        if self.seed is not None:
            self.random_state = np.random.RandomState(self.seed)

    def get_unique_edges_between_start_end_time(self, start_time: float, end_time: float):
        """
        get unique edges happened between start and end time
        :param start_time: float, start timestamp
        :param end_time: float, end timestamp
        :return: a set of edges, where each edge is a tuple of (src_node_id, dst_node_id)
        """
        selected_time_interval = np.logical_and(self.interact_times >= start_time, self.interact_times <= end_time)
        # return the unique select source and destination nodes in the selected time interval
        return set((src_node_id, dst_node_id) for src_node_id, dst_node_id in zip(self.src_node_ids[selected_time_interval], self.dst_node_ids[selected_time_interval]))

    
    def sample(self, size: int, batch_src_node_ids: np.ndarray = None, batch_dst_node_ids: np.ndarray = None,
               current_batch_start_time: float = 0.0, current_batch_end_time: float = 0.0):
        """
        sample negative edges, support random, historical and inductive sampling strategy
        :param size: int, number of sampled negative edges
        :param batch_src_node_ids: ndarray, shape (batch_size, ), source node ids in the current batch
        :param batch_dst_node_ids: ndarray, shape (batch_size, ), destination node ids in the current batch
        :param current_batch_start_time: float, start time in the current batch
        :param current_batch_end_time: float, end time in the current batch
        :return:
        """
        if self.negative_sample_strategy == 'random':
            negative_src_node_ids, negative_dst_node_ids = self.random_sample(size=size)
        elif self.negative_sample_strategy == 'historical':
            negative_src_node_ids, negative_dst_node_ids = self.historical_sample(size=size, batch_src_node_ids=batch_src_node_ids,
                batch_dst_node_ids=batch_dst_node_ids,
                current_batch_start_time=current_batch_start_time,
                current_batch_end_time=current_batch_end_time)
        elif self.negative_sample_strategy == 'inductive':
            negative_src_node_ids, negative_dst_node_ids = self.inductive_sample(size=size, batch_src_node_ids=batch_src_node_ids,
                batch_dst_node_ids=batch_dst_node_ids,
                current_batch_start_time=current_batch_start_time,
                current_batch_end_time=current_batch_end_time)
        else:
            raise ValueError(f'Not implemented error for negative_sample_strategy {self.negative_sample_strategy}!')
        return negative_src_node_ids, negative_dst_node_ids

    def random_sample(self, size: int):
        """
        random sampling strategy, which is used by previous works
        :param size: int, number of sampled negative edges
        :return:
        """
        if self.seed is None:
            random_sample_edge_src_node_indices = np.random.randint(0, len(self.unique_src_node_ids), size)
            random_sample_edge_dst_node_indices = np.random.randint(0, len(self.unique_dst_node_ids), size)
        else:
            random_sample_edge_src_node_indices = self.random_state.randint(0, len(self.unique_src_node_ids), size)
            random_sample_edge_dst_node_indices = self.random_state.randint(0, len(self.unique_dst_node_ids), size)
        return self.unique_src_node_ids[random_sample_edge_src_node_indices], self.unique_dst_node_ids[random_sample_edge_dst_node_indices]

    def random_sample_with_collision_check(self, size: int, batch_src_node_ids: np.ndarray, batch_dst_node_ids: np.ndarray):
        """
        random sampling strategy with collision check, which guarantees that the sampled edges do not appear in the current batch,
        used for historical and inductive sampling strategy
        :param size: int, number of sampled negative edges
        :param batch_src_node_ids: ndarray, shape (batch_size, ), source node ids in the current batch
        :param batch_dst_node_ids: ndarray, shape (batch_size, ), destination node ids in the current batch
        :return:
        """
        assert batch_src_node_ids is not None and batch_dst_node_ids is not None
        batch_edges = set((batch_src_node_id, batch_dst_node_id) for batch_src_node_id, batch_dst_node_id in zip(batch_src_node_ids, batch_dst_node_ids))
        possible_random_edges = list(self.possible_edges - batch_edges)
        assert len(possible_random_edges) > 0
        # if replace is True, then a value in the list can be selected multiple times, otherwise, a value can be selected only once at most
        random_edge_indices = self.random_state.choice(len(possible_random_edges), size=size, replace=len(possible_random_edges) < size)
        return np.array([possible_random_edges[random_edge_idx][0] for random_edge_idx in random_edge_indices]), \
               np.array([possible_random_edges[random_edge_idx][1] for random_edge_idx in random_edge_indices])

    def historical_sample(self, size: int, batch_src_node_ids: np.ndarray, batch_dst_node_ids: np.ndarray,
                          current_batch_start_time: float, current_batch_end_time: float):
        """
        historical sampling strategy, first randomly samples among historical edges that are not in the current batch,
        if number of historical edges is smaller than size, then fill in remaining edges with randomly sampled edges
        :param size: int, number of sampled negative edges
        :param batch_src_node_ids: ndarray, shape (batch_size, ), source node ids in the current batch
        :param batch_dst_node_ids: ndarray, shape (batch_size, ), destination node ids in the current batch
        :param current_batch_start_time: float, start time in the current batch
        :param current_batch_end_time: float, end time in the current batch
        :return:
        """
        assert self.seed is not None
        # get historical edges up to current_batch_start_time
        historical_edges = self.get_unique_edges_between_start_end_time(start_time=self.earliest_time, end_time=current_batch_start_time)
        # get edges in the current batch
        current_batch_edges = self.get_unique_edges_between_start_end_time(start_time=current_batch_start_time, end_time=current_batch_end_time)
        # get source and destination node ids of unique historical edges
        unique_historical_edges = historical_edges - current_batch_edges
        unique_historical_edges_src_node_ids = np.array([edge[0] for edge in unique_historical_edges])
        unique_historical_edges_dst_node_ids = np.array([edge[1] for edge in unique_historical_edges])

        # if sample size is larger than number of unique historical edges, then fill in remaining edges with randomly sampled edges with collision check
        if size > len(unique_historical_edges):
            num_random_sample_edges = size - len(unique_historical_edges)
            random_sample_src_node_ids, random_sample_dst_node_ids = self.random_sample_with_collision_check(size=num_random_sample_edges,
                    batch_src_node_ids=batch_src_node_ids,
                    batch_dst_node_ids=batch_dst_node_ids)

            negative_src_node_ids = np.concatenate([random_sample_src_node_ids, unique_historical_edges_src_node_ids])
            negative_dst_node_ids = np.concatenate([random_sample_dst_node_ids, unique_historical_edges_dst_node_ids])
        else:
            historical_sample_edge_node_indices = self.random_state.choice(len(unique_historical_edges), size=size, replace=False)
            negative_src_node_ids = unique_historical_edges_src_node_ids[historical_sample_edge_node_indices]
            negative_dst_node_ids = unique_historical_edges_dst_node_ids[historical_sample_edge_node_indices]

        # Note that if one of the input of np.concatenate is empty, the output will be composed of floats.
        # Hence, convert the type to long to guarantee valid index
        return negative_src_node_ids.astype(np.longlong), negative_dst_node_ids.astype(np.longlong)

    def inductive_sample(self, size: int, batch_src_node_ids: np.ndarray, batch_dst_node_ids: np.ndarray,
                         current_batch_start_time: float, current_batch_end_time: float):
        """
        inductive sampling strategy, first randomly samples among inductive edges that are not in self.observed_edges and the current batch,
        if number of inductive edges is smaller than size, then fill in remaining edges with randomly sampled edges
        :param size: int, number of sampled negative edges
        :param batch_src_node_ids: ndarray, shape (batch_size, ), source node ids in the current batch
        :param batch_dst_node_ids: ndarray, shape (batch_size, ), destination node ids in the current batch
        :param current_batch_start_time: float, start time in the current batch
        :param current_batch_end_time: float, end time in the current batch
        :return:
        """
        assert self.seed is not None
        # get historical edges up to current_batch_start_time
        historical_edges = self.get_unique_edges_between_start_end_time(start_time=self.earliest_time, end_time=current_batch_start_time)
        # get edges in the current batch
        current_batch_edges = self.get_unique_edges_between_start_end_time(start_time=current_batch_start_time, end_time=current_batch_end_time)
        # get source and destination node ids of historical edges but 1) not in self.observed_edges; 2) not in the current batch
        unique_inductive_edges = historical_edges - self.observed_edges - current_batch_edges
        unique_inductive_edges_src_node_ids = np.array([edge[0] for edge in unique_inductive_edges])
        unique_inductive_edges_dst_node_ids = np.array([edge[1] for edge in unique_inductive_edges])

        # if sample size is larger than number of unique inductive edges, then fill in remaining edges with randomly sampled edges
        if size > len(unique_inductive_edges):
            num_random_sample_edges = size - len(unique_inductive_edges)
            random_sample_src_node_ids, random_sample_dst_node_ids = self.random_sample_with_collision_check(size=num_random_sample_edges,
                batch_src_node_ids=batch_src_node_ids,
                batch_dst_node_ids=batch_dst_node_ids)

            negative_src_node_ids = np.concatenate([random_sample_src_node_ids, unique_inductive_edges_src_node_ids])
            negative_dst_node_ids = np.concatenate([random_sample_dst_node_ids, unique_inductive_edges_dst_node_ids])
        else:
            inductive_sample_edge_node_indices = self.random_state.choice(len(unique_inductive_edges), size=size, replace=False)
            negative_src_node_ids = unique_inductive_edges_src_node_ids[inductive_sample_edge_node_indices]
            negative_dst_node_ids = unique_inductive_edges_dst_node_ids[inductive_sample_edge_node_indices]

        # Note that if one of the input of np.concatenate is empty, the output will be composed of floats.
        # Hence, convert the type to long to guarantee valid index
        return negative_src_node_ids.astype(np.longlong), negative_dst_node_ids.astype(np.longlong)

    def reset_random_state(self):
        """
        reset the random state by self.seed
        :return:
        """
        self.random_state = np.random.RandomState(self.seed)


def get_fix_shape_subgraph_sequence_fast(node_idx, recent_neighbors, recent_times, recent_edges, k_hop, sample_size, avoid_idx=None):
    """
    动态图版本的固定形状子图序列生成方法。
    :param edge_list: 动态图的邻接表，每个节点的邻居按时间顺序存储。
    :param node_idx: 目标节点索引。
    :param k_hop: 采样的跳数。
    :param sample_size: 每层采样的邻居数量。
    :param avoid_idx: 需要避免的节点索引（可选）。
    :return: 固定长度的节点序列。
    """
    assert k_hop > 0 and sample_size > 0

    neighbors = [[node_idx]]  # 初始化邻居列表，包含目标节点
    times = [[0]]  # 初始化邻居列表，包含目标节点
    edges = [[0]]  # 初始化邻居列表，包含目标节点
    edge_index = []
    mappings = [{}]

    for t in range(k_hop): # for循环结束后neighbors包含0/1和2阶邻居，每一阶邻居在对应的列表中,0阶是节点本身位于位置0处的子列表，1阶是在位置1处的子列表，2阶是位置2处的子列表
        last_hop = neighbors[-1]  # 获取上一跳的邻居
        current_hop_neighbor = []  # 初始化当前跳的邻居
        current_hop_time = []  
        current_hop_edge = []  
        mapping = {}

        for i in last_hop:
            # 获取节点 i 的邻居，邻居已按时间顺序排序
            node_neighbor = list(recent_neighbors[i]) # 数量已限制了
            node_time = list(recent_times[i]) 
            node_edge = list(recent_edges[i]) 
            for x in node_neighbor:
                edge_index.append([i, x])
                mapping[x] = i # 记录后一级邻居是从前一级哪个邻居来的

            current_hop_neighbor.extend(node_neighbor) # 已经是最近的 sample_size 个邻居了
            current_hop_time.extend(node_time) 
            current_hop_edge.extend(node_edge) 

        neighbors.append(current_hop_neighbor)
        times.append(current_hop_time)
        edges.append(current_hop_edge)
        mappings.append(mapping)

    # 展平邻居列表
    node_sequence = list(set([n for hop in neighbors for n in hop])) #去除重复节点，因为需要的信息已在edge_index中体现了
    #下面将edge_index转换为需要的格式
    if not edge_index: # edge_index为空，该节点之前没有邻居，此时让其自连接
        edge_index = [[node_sequence[0], node_sequence[0]]]
    edge_index = torch.as_tensor(edge_index).T
    edge_index, _ = map_index(edge_index, torch.tensor(node_sequence), inclusive=True)
    edge_index = to_undirected(edge_index)
    return node_sequence, edge_index.tolist(), neighbors, times, edges, mappings

def get_temporal_subgraph_sequence(node_idx, ts, full_neighbor_sampler, k_hop=2, sample_size=2, avoid_idx=None):
    """
    寻找动态图中节点u在时刻t之前的多跳邻居。
    :param edge_list: 动态图的邻接表，每个节点的邻居按时间顺序存储。
    :param node_idx: 目标节点索引。
    :param k_hop: 采样的跳数。
    :param sample_size: 每层采样的邻居数量。
    :param avoid_idx: 需要避免的节点索引（可选）。
    :return: 固定长度的节点序列。
    """
    assert k_hop > 0 and sample_size > 0

    neighbors = [[node_idx]]  # 初始化邻居列表，包含目标节点
    times = [[0]]  # 初始化邻居列表，包含目标节点
    edges = [[0]]  # 初始化邻居列表，包含目标节点
    edge_index = []
    mappings = [{}]

    for t in range(k_hop):
        last_hop = neighbors[-1]  # 获取上一跳的邻居
        current_hop_neighbor = []  # 初始化当前跳的邻居
        current_hop_time = []  
        current_hop_edge = []  
        mapping = {}

        for i in last_hop:
            # 获取节点 i 的邻居，邻居已按时间顺序排序
            node_neighbor,node_edge,node_time = full_neighbor_sampler.get_historical_neighbors(node_ids=[i],
                                                               node_interact_times=[ts],#都是寻找当前时刻之前的各阶邻居的
                                                               num_neighbors=sample_size)
            node_neighbor,node_edge,node_time = node_neighbor[0].tolist(),node_edge[0].tolist(),node_time[0].tolist()
            
            # # 如果是第一跳且需要避免某个节点，则从邻居中移除
            # if t == 0 and avoid_idx is not None and avoid_idx in node_neighbor:
            #     node_neighbor.remove(avoid_idx)
            for x in node_neighbor:
                edge_index.append([i, x])
                mapping[x] = i # 记录后一级邻居是从前一级哪个邻居来的

            current_hop_neighbor.extend(node_neighbor) # 已经是最近的 sample_size 个邻居了
            current_hop_time.extend(node_time) 
            current_hop_edge.extend(node_edge) 

        neighbors.append(current_hop_neighbor)
        times.append(current_hop_time)
        edges.append(current_hop_edge)
        mappings.append(mapping)

    # 展平邻居列表
    node_sequence = list(set([n for hop in neighbors for n in hop])) #去除重复节点，因为需要的信息已在edge_index中体现了
    #下面将edge_index转换为需要的格式
    if not edge_index: # edge_index为空，该节点之前没有邻居，此时让其自连接
        edge_index = [[node_sequence[0], node_sequence[0]]]
    edge_index = torch.as_tensor(edge_index).T
    edge_index, _ = map_index(edge_index, torch.tensor(node_sequence), inclusive=True)
    edge_index = to_undirected(edge_index)
    return node_sequence, edge_index.tolist(), neighbors, times, edges, mappings

def generate_prompt1_with_timestamps(neighbors, times, edges, node_texts, edge_texts,mappings, item_word='Product', args=None):
    prompt = f"Given a sequence of graph tokens: \n<graph>\n that constitute a user-{item_word.lower()} review subgraph, where the first token represents the central node (the user), and the remaining nodes represent the central node's first- and second-order neighbors. The first-order neighbors are the {item_word.lower()}s that the central node has reviewed and the second-order neighbors are other users who have reviewed the same {item_word.lower()}s.\n"
    if args.empty:
        return prompt
    
    product_details = list(zip(neighbors[1],times[1],edges[1])) # 一阶邻居，即商品
    if product_details:
        prompt += "For each first-order neighbor reviewed by the center node, the following information is provided:\n"
    for detail in product_details:
        vid, ts, eid = detail
        ts = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        prompt += f"Node {vid}:\n"
        description = node_texts[vid]
        if description: # 描述不为空
            description = " ".join(description.split(" ")[:128])
            prompt += f"{item_word} description: {description}.\n"
        review = edge_texts[eid]
        if review:
            review = " ".join(str(review).split(" ")[:128])
            prompt += f"Central node's review: {review}\nReview timestamp: {ts}\n"

    # product_details = list(zip(neighbors[2],times[2],edges[2])) # 二阶邻居，即用户
    # if product_details:
    #     prompt += "For each second-order neighbor (user who also reviews the products from the first-order neighbors), the following information is provided:\n"
    # for detail in product_details:
    #     vid, ts, eid = detail
    #     ts = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    #     review = edge_texts[eid]
    #     if review:
    #         prompt += f"Review by node {vid} (user) on node {mappings[2][vid]} ({item_word.lower()}):\n"
    #         prompt += f"Review text: {review}\nReview timestamp: {ts}\n"

    return prompt

def generate_prompt2_with_timestamps(neighbors, times, edges, node_texts, edge_texts,mappings, item_word='Product', args=None):
    prompt = f"and the other sequence of graph tokens: \n<graph>\n , where the first token corresponds to the center node (the {item_word.lower()}), and the remaining tokens represent the {item_word.lower()}'s first- and second-order neighbors. The first-order neighbors are the users who have reviewed the {item_word.lower()} and the second-order neighbors are the {item_word.lower()}s that those users also have reviewed."
    if args.empty:
        return prompt
    
    description = node_texts[neighbors[0][0]] 
    if description: # 描述不为空
        prompt += f"For the center node (node representing the {item_word.lower()}), its {item_word.lower()} information is:\n{description}.\n"
    product_details = list(zip(neighbors[1],times[1],edges[1])) # 一阶邻居，即用户
    if product_details:
        prompt += f"For each first-order neighbor (user who reviews the center node), the following information is provided:\n"
    for detail in product_details:
        vid, ts, eid = detail
        ts = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        review = edge_texts[eid]
        if review:
            review = " ".join(str(review).split(" ")[:128])
            prompt += f"Review by node {vid}: {review}\nReview timestamp: {ts}\n"

    # product_details = list(zip(neighbors[2],times[2],edges[2])) # 二阶邻居，即用户
    # if product_details:
    #     prompt += f"For each second-order neighbor (the other {item_word.lower()}s also reviewed by users from the first-order neighbors), the following information is provided:\n"
    # for detail in product_details:
    #     vid, ts, eid = detail
    #     ts = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    #     prompt += f"Node {vid}:\n"
    #     description = node_texts[vid]
    #     if description: # 描述不为空
    #         prompt += f"{item_word} description: {description}.\n"
    #     review = edge_texts[eid]
    #     if review:
    #         prompt += f"Review by node {mappings[2][vid]}: {review}\nReview timestamp: {ts}\n"
    return prompt

def sampler_prompt1_with_timestamps(neighbors, times, edges, node_texts, edge_texts,item_word='Product'):
    prompt = f"Given a sequence of graph tokens: \n<graph>\n that constitute a user-{item_word.lower()} review subgraph, where the first token represents the central node (the user), and the remaining nodes represent the central node's first- and second-order neighbors. The first-order neighbors are the {item_word.lower()}s that the central node has reviewed and the second-order neighbors are other users who have reviewed the same {item_word.lower()}s.\n"
    
    product_details = zip(neighbors[1],times[1],edges[1]) # 一阶邻居，即商品
    if list(product_details):
        prompt += "For each first-order neighbor reviewed by the center node, the following information is provided:\n"
    for detail in product_details:
        vid, ts, eid = detail
        ts = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        prompt += f"Node {vid}:\n"
        description = node_texts[vid]
        if description: # 描述不为空
            prompt += f"{item_word} description: {description}.\n"
        review = edge_texts[eid]
        if review:
            prompt += f"Central node's review: {review}\nReview timestamp: {ts}\n"

    product_details = zip(neighbors[2],times[2],edges[2]) # 二阶邻居，即用户
    if list(product_details):
        prompt += "For each second-order neighbor (user who also reviews the products from the first-order neighbors), the following information is provided:\n"
    for detail in product_details:
        uid, ts, eid = detail
        ts = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        review = edge_texts[eid]
        if review:
            prompt += f"Review by node {uid} (user) on node {mapping[vid]} ({item_word.lower()}):\n"
            prompt += f"Review text: {review}\nReview timestamp: {ts}\n"

    return prompt

def sampler_prompt2_with_timestamps(neighbors, times, edges, node_texts, edge_texts,mappings, item_word='Product'):
    prompt = f"and the other sequence of graph tokens: \n<graph>\n , where the first token corresponds to the center node (the {item_word.lower()}), and the remaining tokens represent the {item_word.lower()}'s first- and second-order neighbors. The first-order neighbors are the users who have reviewed the {item_word.lower()} and the second-order neighbors are the {item_word.lower()}s that those users also have reviewed."
    description = node_texts[neighbors[0][0]] 
    if description: # 描述不为空
        prompt += f"For the center node (node representing the {item_word.lower()}), its {item_word.lower()} information is:\n{description}.\n"
    product_details = zip(neighbors[1],times[1],edges[1]) # 一阶邻居，即用户
    if list(product_details):
        prompt += f"For each first-order neighbor (user who reviews the center node), the following information is provided:\n"
    for detail in product_details:
        vid, ts, eid = detail
        ts = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        review = edge_texts[eid]
        if review:
            prompt += f"Review by node {vid}: {review}\nReview timestamp: {ts}\n"

    product_details = zip(neighbors[2],times[2],edges[2],mappings[2]) # 二阶邻居，即用户
    if list(product_details):
        prompt += f"For each second-order neighbor (the other {item_word.lower()}s also reviewed by users from the first-order neighbors), the following information is provided:\n"
    for detail in product_details:
        vid, ts, eid, mapping = detail
        ts = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        prompt += f"Node {vid}:\n"
        description = node_texts[vid]
        if description: # 描述不为空
            prompt += f"{item_word} description: {description}.\n"
        review = edge_texts[eid]
        if review:
            prompt += f"Review by node {mapping[vid]}: {review}\nReview timestamp: {ts}\n"
    return prompt



class NeighborSampler:

    def __init__(self, adj_list: list, sample_neighbor_strategy: str = 'recent', time_scaling_factor: float = 0.0, seed: int = None):
        """
        Neighbor sampler.
        :param adj_list: list, list of list, where each element is a list of triple tuple (node_id, edge_id, timestamp)
        :param sample_neighbor_strategy: str, how to sample historical neighbors, 'uniform', 'recent', or 'time_interval_aware'
        :param time_scaling_factor: float, a hyper-parameter that controls the sampling preference with time interval,
        a large time_scaling_factor tends to sample more on recent links, this parameter works when sample_neighbor_strategy == 'time_interval_aware'
        :param seed: int, random seed
        """
        self.sample_neighbor_strategy = sample_neighbor_strategy
        self.seed = seed

        # list of each node's neighbor ids, edge ids and interaction times, which are sorted by interaction times
        self.nodes_neighbor_ids = []
        self.nodes_edge_ids = []
        self.nodes_neighbor_times = []

        if self.sample_neighbor_strategy == 'time_interval_aware':
            self.nodes_neighbor_sampled_probabilities = []
            self.time_scaling_factor = time_scaling_factor

        # the list at the first position in adj_list is empty, hence, sorted() will return an empty list for the first position
        # its corresponding value in self.nodes_neighbor_ids, self.nodes_edge_ids, self.nodes_neighbor_times will also be empty with length 0
        for node_idx, per_node_neighbors in enumerate(adj_list):
            # per_node_neighbors is a list of tuples (neighbor_id, edge_id, timestamp)
            # sort the list based on timestamps, sorted() function is stable
            # Note that sort the list based on edge id is also correct, as the original data file ensures the interactions are chronological
            sorted_per_node_neighbors = sorted(per_node_neighbors, key=lambda x: x[2])
            self.nodes_neighbor_ids.append(np.array([x[0] for x in sorted_per_node_neighbors]))
            self.nodes_edge_ids.append(np.array([x[1] for x in sorted_per_node_neighbors]))
            self.nodes_neighbor_times.append(np.array([x[2] for x in sorted_per_node_neighbors]))

            # additional for time interval aware sampling strategy (proposed in CAWN paper)
            if self.sample_neighbor_strategy == 'time_interval_aware':
                self.nodes_neighbor_sampled_probabilities.append(self.compute_sampled_probabilities(np.array([x[2] for x in sorted_per_node_neighbors])))

        if self.seed is not None:
            self.random_state = np.random.RandomState(self.seed)

    def compute_sampled_probabilities(self, node_neighbor_times: np.ndarray):
        """
        compute the sampled probabilities of historical neighbors based on their interaction times
        :param node_neighbor_times: ndarray, shape (num_historical_neighbors, )
        :return:
        """
        if len(node_neighbor_times) == 0:
            return np.array([])
        # compute the time delta with regard to the last time in node_neighbor_times
        node_neighbor_times = node_neighbor_times - np.max(node_neighbor_times)
        # compute the normalized sampled probabilities of historical neighbors
        exp_node_neighbor_times = np.exp(self.time_scaling_factor * node_neighbor_times)
        sampled_probabilities = exp_node_neighbor_times / np.cumsum(exp_node_neighbor_times)
        # note that the first few values in exp_node_neighbor_times may be all zero, which make the corresponding values in sampled_probabilities
        # become nan (divided by zero), so we replace the nan by a very large negative number -1e10 to denote the sampled probabilities
        sampled_probabilities[np.isnan(sampled_probabilities)] = -1e10
        return sampled_probabilities

    def find_neighbors_before(self, node_id: int, interact_time: float, return_sampled_probabilities: bool = False):
        """
        extracts all the interactions happening before interact_time (less than interact_time) for node_id in the overall interaction graph
        the returned interactions are sorted by time.
        :param node_id: int, node id
        :param interact_time: float, interaction time
        :param return_sampled_probabilities: boolean, whether return the sampled probabilities of neighbors
        :return: neighbors, edge_ids, timestamps and sampled_probabilities (if return_sampled_probabilities is True) with shape (historical_nodes_num, )
        """
        # return index i, which satisfies list[i - 1] < v <= list[i]
        # return 0 for the first position in self.nodes_neighbor_times since the value at the first position is empty
        i = np.searchsorted(self.nodes_neighbor_times[node_id], interact_time)

        if return_sampled_probabilities:
            return self.nodes_neighbor_ids[node_id][:i], self.nodes_edge_ids[node_id][:i], self.nodes_neighbor_times[node_id][:i], \
                   self.nodes_neighbor_sampled_probabilities[node_id][:i]
        else:
            return self.nodes_neighbor_ids[node_id][:i], self.nodes_edge_ids[node_id][:i], self.nodes_neighbor_times[node_id][:i], None

    def get_historical_neighbors(self, node_ids: np.ndarray, node_interact_times: np.ndarray, num_neighbors: int = 20):
        """
        get historical neighbors of nodes in node_ids with interactions before the corresponding time in node_interact_times
        :param node_ids: ndarray, shape (batch_size, ) or (*, ), node ids
        :param node_interact_times: ndarray, shape (batch_size, ) or (*, ), node interaction times
        :param num_neighbors: int, number of neighbors to sample for each node
        :return:
        """
        assert num_neighbors > 0, 'Number of sampled neighbors for each node should be greater than 0!'
        # All interactions described in the following three matrices are sorted in each row by time
        # each entry in position (i,j) represents the id of the j-th dst node of src node node_ids[i] with an interaction before node_interact_times[i]
        # ndarray, shape (batch_size, num_neighbors)
        nodes_neighbor_ids = np.zeros((len(node_ids), num_neighbors)).astype(np.longlong)
        # each entry in position (i,j) represents the id of the edge with src node node_ids[i] and dst node nodes_neighbor_ids[i][j] with an interaction before node_interact_times[i]
        # ndarray, shape (batch_size, num_neighbors)
        nodes_edge_ids = np.zeros((len(node_ids), num_neighbors)).astype(np.longlong)
        # each entry in position (i,j) represents the interaction time between src node node_ids[i] and dst node nodes_neighbor_ids[i][j], before node_interact_times[i]
        # ndarray, shape (batch_size, num_neighbors)
        nodes_neighbor_times = np.zeros((len(node_ids), num_neighbors)).astype(np.float32)

        # extracts all neighbors ids, edge ids and interaction times of nodes in node_ids, which happened before the corresponding time in node_interact_times
        for idx, (node_id, node_interact_time) in enumerate(zip(node_ids, node_interact_times)):
            # find neighbors that interacted with node_id before time node_interact_time
            node_neighbor_ids, node_edge_ids, node_neighbor_times, node_neighbor_sampled_probabilities = \
                self.find_neighbors_before(node_id=node_id, interact_time=node_interact_time, return_sampled_probabilities=self.sample_neighbor_strategy == 'time_interval_aware')

            if len(node_neighbor_ids) > 0:
                if self.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
                    # when self.sample_neighbor_strategy == 'uniform', we shuffle the data before sampling with node_neighbor_sampled_probabilities as None
                    # when self.sample_neighbor_strategy == 'time_interval_aware', we sample neighbors based on node_neighbor_sampled_probabilities
                    # for time_interval_aware sampling strategy, we additionally use softmax to make the sum of sampled probabilities be 1
                    if node_neighbor_sampled_probabilities is not None:
                        # for extreme case that node_neighbor_sampled_probabilities only contains -1e10, which will make the denominator of softmax be zero,
                        # torch.softmax() function can tackle this case
                        node_neighbor_sampled_probabilities = torch.softmax(torch.from_numpy(node_neighbor_sampled_probabilities).float(), dim=0).numpy()
                    if self.seed is None:
                        sampled_indices = np.random.choice(a=len(node_neighbor_ids), size=num_neighbors, p=node_neighbor_sampled_probabilities)
                    else:
                        sampled_indices = self.random_state.choice(a=len(node_neighbor_ids), size=num_neighbors, p=node_neighbor_sampled_probabilities)

                    nodes_neighbor_ids[idx, :] = node_neighbor_ids[sampled_indices]
                    nodes_edge_ids[idx, :] = node_edge_ids[sampled_indices]
                    nodes_neighbor_times[idx, :] = node_neighbor_times[sampled_indices]

                    # resort based on timestamps, return the ids in sorted increasing order, note this maybe unstable when multiple edges happen at the same time
                    # (we still do this though this is unnecessary for TGAT or CAWN to guarantee the order of nodes,
                    # since TGAT computes in an order-agnostic manner with relative time encoding, and CAWN computes for each walk while the sampled nodes are in different walks)
                    sorted_position = nodes_neighbor_times[idx, :].argsort()
                    nodes_neighbor_ids[idx, :] = nodes_neighbor_ids[idx, :][sorted_position]
                    nodes_edge_ids[idx, :] = nodes_edge_ids[idx, :][sorted_position]
                    nodes_neighbor_times[idx, :] = nodes_neighbor_times[idx, :][sorted_position]
                elif self.sample_neighbor_strategy == 'recent':
                    # Take most recent interactions with number num_neighbors
                    node_neighbor_ids = node_neighbor_ids[-num_neighbors:]
                    node_edge_ids = node_edge_ids[-num_neighbors:]
                    node_neighbor_times = node_neighbor_times[-num_neighbors:]

                    # put the neighbors' information at the back positions
                    nodes_neighbor_ids[idx, num_neighbors - len(node_neighbor_ids):] = node_neighbor_ids
                    nodes_edge_ids[idx, num_neighbors - len(node_edge_ids):] = node_edge_ids
                    nodes_neighbor_times[idx, num_neighbors - len(node_neighbor_times):] = node_neighbor_times
                else:
                    raise ValueError(f'Not implemented error for sample_neighbor_strategy {self.sample_neighbor_strategy}!')

        # three ndarrays, with shape (batch_size, num_neighbors)
        return nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times

    def get_multi_hop_neighbors(self, num_hops: int, node_ids: np.ndarray, node_interact_times: np.ndarray, num_neighbors: int = 20):
        """
        get historical neighbors of nodes in node_ids within num_hops hops
        :param num_hops: int, number of sampled hops
        :param node_ids: ndarray, shape (batch_size, ), node ids
        :param node_interact_times: ndarray, shape (batch_size, ), node interaction times
        :param num_neighbors: int, number of neighbors to sample for each node
        :return:
        """
        assert num_hops > 0, 'Number of sampled hops should be greater than 0!'

        # get the temporal neighbors at the first hop
        # nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times -> ndarray, shape (batch_size, num_neighbors)
        nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times = self.get_historical_neighbors(node_ids=node_ids,
                                                                                                 node_interact_times=node_interact_times,
                                                                                                 num_neighbors=num_neighbors)
        # three lists to store the neighbor ids, edge ids and interaction timestamp information
        nodes_neighbor_ids_list = [nodes_neighbor_ids]
        nodes_edge_ids_list = [nodes_edge_ids]
        nodes_neighbor_times_list = [nodes_neighbor_times]
        for hop in range(1, num_hops):
            # get information of neighbors sampled at the current hop
            # three ndarrays, with shape (batch_size * num_neighbors ** hop, num_neighbors)
            nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times = self.get_historical_neighbors(node_ids=nodes_neighbor_ids_list[-1].flatten(),
                                                                                                     node_interact_times=nodes_neighbor_times_list[-1].flatten(),
                                                                                                     num_neighbors=num_neighbors)
            # three ndarrays with shape (batch_size, num_neighbors ** (hop + 1))
            nodes_neighbor_ids = nodes_neighbor_ids.reshape(len(node_ids), -1)
            nodes_edge_ids = nodes_edge_ids.reshape(len(node_ids), -1)
            nodes_neighbor_times = nodes_neighbor_times.reshape(len(node_ids), -1)

            nodes_neighbor_ids_list.append(nodes_neighbor_ids)
            nodes_edge_ids_list.append(nodes_edge_ids)
            nodes_neighbor_times_list.append(nodes_neighbor_times)

        # tuple, each element in the tuple is a list of num_hops ndarrays, each with shape (batch_size, num_neighbors ** current_hop)
        return nodes_neighbor_ids_list, nodes_edge_ids_list, nodes_neighbor_times_list

    def get_all_first_hop_neighbors(self, node_ids: np.ndarray, node_interact_times: np.ndarray):
        """
        get historical neighbors of nodes in node_ids at the first hop with max_num_neighbors as the maximal number of neighbors (make the computation feasible)
        :param node_ids: ndarray, shape (batch_size, ), node ids
        :param node_interact_times: ndarray, shape (batch_size, ), node interaction times
        :return:
        """
        # three lists to store the first-hop neighbor ids, edge ids and interaction timestamp information, with batch_size as the list length
        nodes_neighbor_ids_list, nodes_edge_ids_list, nodes_neighbor_times_list = [], [], []
        # get the temporal neighbors at the first hop
        for idx, (node_id, node_interact_time) in enumerate(zip(node_ids, node_interact_times)):
            # find neighbors that interacted with node_id before time node_interact_time
            node_neighbor_ids, node_edge_ids, node_neighbor_times, _ = self.find_neighbors_before(node_id=node_id,
                                                                                                  interact_time=node_interact_time,
                                                                                                  return_sampled_probabilities=False)
            nodes_neighbor_ids_list.append(node_neighbor_ids)
            nodes_edge_ids_list.append(node_edge_ids)
            nodes_neighbor_times_list.append(node_neighbor_times)

        return nodes_neighbor_ids_list, nodes_edge_ids_list, nodes_neighbor_times_list

    def reset_random_state(self):
        """
        reset the random state by self.seed
        :return:
        """
        self.random_state = np.random.RandomState(self.seed)

def get_neighbor_sampler(data: Data, sample_neighbor_strategy: str = 'uniform', time_scaling_factor: float = 0.0, seed: int = None):
    """
    get neighbor sampler
    :param data: Data
    :param sample_neighbor_strategy: str, how to sample historical neighbors, 'uniform', 'recent', or 'time_interval_aware''
    :param time_scaling_factor: float, a hyper-parameter that controls the sampling preference with time interval,
    a large time_scaling_factor tends to sample more on recent links, this parameter works when sample_neighbor_strategy == 'time_interval_aware'
    :param seed: int, random seed
    :return:
    """
    max_node_id = max(data.src_node_ids.max(), data.dst_node_ids.max())
    # the adjacency vector stores edges for each node (source or destination), undirected
    # adj_list, list of list, where each element is a list of triple tuple (node_id, edge_id, timestamp)
    # the list at the first position in adj_list is empty
    adj_list = [[] for _ in range(max_node_id + 1)]
    for src_node_id, dst_node_id, edge_id, node_interact_time in zip(data.src_node_ids, data.dst_node_ids, data.edge_ids, data.node_interact_times):
        adj_list[src_node_id].append((dst_node_id, edge_id, node_interact_time))
        adj_list[dst_node_id].append((src_node_id, edge_id, node_interact_time))

    return NeighborSampler(adj_list=adj_list, sample_neighbor_strategy=sample_neighbor_strategy, time_scaling_factor=time_scaling_factor, seed=seed)
    

class CustomizedDataset(Dataset):
    def __init__(self, indices_list: list):
        """
        Customized dataset.
        :param indices_list: list, list of indices
        """
        super(CustomizedDataset, self).__init__()

        self.indices_list = indices_list

    def __getitem__(self, idx: int):
        """
        get item at the index in self.indices_list
        :param idx: int, the index
        :return:
        """
        return self.indices_list[idx]

    def __len__(self):
        return len(self.indices_list)

def get_idx_data_loader(indices_list: list, batch_size: int, shuffle: bool):
    """
    get data loader that iterates over indices
    :param indices_list: list, list of indices
    :param batch_size: int, batch size
    :param shuffle: boolean, whether to shuffle the data
    :return: data_loader, DataLoader
    """
    dataset = CustomizedDataset(indices_list=indices_list)

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             drop_last=False)
    return data_loader

def get_neighbors(neighbor_sampler, target_node, ts, sample_size=2, is_user=True):
    """
    获取目标节点（用户或 item）的二阶邻居列表
    :param neighbor_sampler: 邻居抽样器
    :param target_node: 目标节点（用户或 item）
    :param is_user: 如果 True，则 target_node 是用户；否则是 item
    :return: 字典，包含一阶和二阶邻居及其连接信息
    """
    
    # 一阶邻居
    first_order = graph[target_node]
    node_neighbor,node_edge,node_time = neighbor_sampler.get_historical_neighbors(node_ids=[target_node],
                                                               node_interact_times=[ts],#都是寻找当前时刻之前的各阶邻居的
                                                               num_neighbors=sample_size)
    
    # 二阶邻居
    second_order = []
    for neighbor, t, item_desc, comment in first_order:
        for second_neighbor, t2, item_desc2, comment2 in graph[neighbor]:
            if second_neighbor != target_node and (second_neighbor, t2, item_desc2, comment2) not in first_order:
                second_order.append((second_neighbor, t2, item_desc2, comment2))
    
    if is_user:
        # 用户视角：一阶邻居是 item，二阶邻居是其他用户
        return {
            "1st_order": [(n, t, d, c) for n, t, d, c in first_order],  # item 列表
            "2nd_order": [(n, t, d, c) for n, t, d, c in second_order if n.startswith("user")]  # 用户列表
        }
    else:
        # Item 视角：一阶邻居是用户，二阶邻居是其他 item
        return {
            "1st_order": [(n, t, d, c) for n, t, d, c in first_order],  # 用户列表
            "2nd_order": [(n, t, d, c) for n, t, d, c in second_order if not n.startswith("user")]  # item 列表
        }

def get_train_samples(train_data: Data, train_sampler: NeighborSampler,node_texts, edge_texts,samples=2, args=None):
    train_samples = []
    uts = []
    answers = []
    for _ in range(samples//2):
        #1. positive exemplar
        cur_sample = []
        exemplar_sample = train_data.random_edge()
        u, v, t, _ = exemplar_sample
        _, _, neighbors1, times1, edges1, mappings1 = get_temporal_subgraph_sequence(u, t, train_sampler, sample_size=samples)
        for vid, ts, eid in zip(neighbors1[1],times1[1],edges1[1]): # user的一阶邻居是item
            ts = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
            if args.empty:
                cur_sample.append((u, vid, ts))
            else:
                cur_sample.append((u, vid, ts, truncate_text(node_texts[vid]), truncate_text(edge_texts[eid])))
        for uid, ts, eid in zip(neighbors1[2],times1[2],edges1[2]): # user的二阶邻居是user
            ts = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
            vid = mappings1[2][uid]
            if args.empty:
                cur_sample.append((uid, vid, ts))
            else:
                cur_sample.append((uid, vid, ts, truncate_text(node_texts[vid]), truncate_text(edge_texts[eid])))

        _, _, neighbors2, times2, edges2, mappings2 = get_temporal_subgraph_sequence(v, t, train_sampler, sample_size=samples)
        for uid, ts, eid in zip(neighbors2[1],times2[1],edges2[1]): # item的一阶邻居是user
            ts = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
            if args.empty:
                cur_sample.append((uid, v, ts))
            else:
                cur_sample.append((uid, v, ts, truncate_text(node_texts[v]), truncate_text(edge_texts[eid])))
        for vid, ts, eid in zip(neighbors2[2],times2[2],edges2[2]): # item的二阶邻居是item
            ts = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
            uid = mappings2[2][vid]
            if args.empty:
                cur_sample.append((uid, vid, ts))
            else:
                cur_sample.append((uid, vid, ts, truncate_text(node_texts[vid]), truncate_text(edge_texts[eid])))
        train_samples.append(cur_sample)
        uts.append((u, v, t))
        answers.append('Yes')

        #2. negative exemplar
        _, v, t, _ = train_data.random_edge()
        cur_sample = []
        _, _, neighbors1, times1, edges1, mappings1 = get_temporal_subgraph_sequence(u, t, train_sampler, sample_size=samples)
        for vid, ts, eid in zip(neighbors1[1],times1[1],edges1[1]): # user的一阶邻居是item
            ts = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
            if args.empty:
                cur_sample.append((u, vid, ts))
            else:
                cur_sample.append((u, vid, ts, truncate_text(node_texts[vid]), truncate_text(edge_texts[eid])))
        for uid, ts, eid in zip(neighbors1[2],times1[2],edges1[2]): # user的二阶邻居是user
            ts = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
            vid = mappings1[2][uid]
            if args.empty:
                cur_sample.append((uid, vid, ts))
            else:
                cur_sample.append((uid, vid, ts, truncate_text(node_texts[vid]), truncate_text(edge_texts[eid])))

        _, _, neighbors2, times2, edges2, mappings2 = get_temporal_subgraph_sequence(v, t, train_sampler, sample_size=samples)
        for uid, ts, eid in zip(neighbors2[1],times2[1],edges2[1]): # item的一阶邻居是user
            ts = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
            if args.empty:
                cur_sample.append((uid, v, ts))
            else:
                cur_sample.append((uid, v, ts, truncate_text(node_texts[v]), truncate_text(edge_texts[eid])))
        for vid, ts, eid in zip(neighbors2[2],times2[2],edges2[2]): # item的二阶邻居是item
            ts = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
            uid = mappings2[2][vid]
            if args.empty:
                cur_sample.append((uid, vid, ts))
            else:
                cur_sample.append((uid, vid, ts, truncate_text(node_texts[vid]), truncate_text(edge_texts[eid])))
        train_samples.append(cur_sample)
        uts.append((u, v, t))
        answers.append('No')

    indexed_list = [(sublist, idx) for idx, sublist in enumerate(train_samples)]
    # 随机打乱带索引的列表
    random.shuffle(indexed_list)
    # 分离打乱后的子列表和对应的原始索引
    train_samples = [item[0] for item in indexed_list]
    original_indices = [item[1] for item in indexed_list]
    uts = [uts[i] for i in original_indices]
    answers = [answers[i] for i in original_indices]
    # train_samples = [x for sublist in train_samples for x in sublist]

    exemplar = f"Here are {samples} examples:\n"
    if args.empty:
        for i in range(samples):
            exemplar += (
            "Historical interactions: " + str(train_samples[i]) + "\n"
            f"Question: Will user {uts[i][0]} interact with item {uts[i][1]} at time '{uts[i][2]}'?\n"
            f"Answer: {answers[i]}\n" 
        )
    else:
        for i in range(samples):
            exemplar += (
            "Historical comments: " + str(train_samples[i]) + "\n"
            f"Question: Will user {uts[i][0]} comment on {item_word.lower()} {uts[i][1]} at time '{uts[i][2]}'?\n"
            f"Answer: {answers[i]}\n" 
        )
    
    return exemplar.strip()


def generate_dst2_prompt(u, v, t, test_sampler, node_texts, edge_texts, exemplar, item_word='Product', variant="nondst2", samples=2, args=None):
    """
    生成 DST2 提示，示例从训练集构造，问题从测试集构造
    :param train_data: 训练集
    :param neighbors, times, edges: 邻居交互对应信息
    :param variant: DST2 变体 ("v1" 表示先结构后时间, "v2" 表示先时间后结构)
    :return: prompt
    """

    if args.empty:
        # 动态图指令
        dyg_instruction = f"In a temporal graph, (u, v, t) means that user u interacted with item v at time t."
        task_instruction = (
            "Your task is to predict whether a user will interact with a specific item at a future time. "
            "You are given the historical interactions in the graph, which include information about "
            "the 1st-order and 2nd-order neighbors of the user and the item. For a user, 1st-order "
            "neighbors are items they interacted with, and 2nd-order neighbors are other users who "
            "interacted with those items. For an item, 1st-order neighbors are users who interacted "
            "with it, and 2nd-order neighbors are other items interacted with by those users."
        )

        dst2_instructions = {
            "dst2-v1": (
                "Think about structure and then time: First analyze the structural information within "
                "the historical interactions to understand the relationships between users and items, "
                "then consider the temporal patterns to predict the future connection."
            ),
            "dst2-v2": (
                "Think about time and then structure: First analyze the temporal sequence of interactions "
                "in the historical data to identify patterns over time, then examine the structural "
                "information within the historical interactions to predict the future connection."
            ),
            "nondst2": (
                "Analyze the historical interactions to identify patterns that might indicate a future interaction. "
                "Consider both the relationships between users and items and the timing of past interactions "
                "within the historical interactions to make your prediction."
            )
        }
    else:
        # 动态图指令
        dyg_instruction = (
            f"In a temporal graph, (u, v, t, {item_word.lower()}_desc, comment) means that user u commented on {item_word.lower()} v "
            f"(with description '{item_word.lower()}_desc') at time t, with the comment 'comment'. "
        )

        # 任务指令
        task_instruction = (
            f"Your task is to predict whether a user will comment on a specific {item_word.lower()} at a future time. "
            "You are given the historical comments in the graph, which include interactions that contain "
            f"information about the 1st-order and 2nd-order neighbors of the user and the {item_word.lower()}. "
            f"For a user, 1st-order neighbors are {item_word.lower()}s they commented on, and "
            f"2nd-order neighbors are other users who commented on those {item_word.lower()}s. For an {item_word.lower()}, 1st-order "
            f"neighbors are users who commented on it, and 2nd-order neighbors are other {item_word.lower()}s commented "
            f"on by those users. Predict if the user will comment on the {item_word.lower()} at the specified future time."
        )

        # DST2 指令
        dst2_instructions = {
            "dst2-v1": (
                "Think about structure and then time: First analyze the structural information "
                f"(1st-order and 2nd-order neighbors) of the user and the {item_word.lower()} to understand their relationships, "
                "then consider the temporal patterns in the historical comments to predict the future connection."
            ),
            "dst2-v2": (
                "Think about time and then structure: First analyze the temporal sequence of comments in the historical data "
                "to identify patterns over time, then examine the structural information (1st-order and 2nd-order neighbors) "
                f"of the user and the {item_word.lower()} to predict the future connection."
            ),
            # 通用推理指令（不分解时空）
            "nondst2": (
                f"Analyze the historical comments and the neighbor information of the user and the {item_word.lower()} together "
                "to identify patterns that might indicate a future comment. Consider both the relationships "
                f"(users, {item_word.lower()}s, and their connections) and the timing of past comments to make your prediction."
            )
        }
    dst2_instruction = dst2_instructions.get(variant, dst2_instructions["nondst2"])

    # 答案指令
    # answer_instruction = "Give the answer as 'Yes' or 'No' at the end of your response after 'Answer:'."
    answer_instruction = (
        "Provide your prediction directly as 'Answer: Yes' or 'Answer: No'. Do not include any "
        "explanations, reasoning, or additional text beyond this format."
    )

    cur_sample = []
    _, _, neighbors1, times1, edges1, mappings1 = get_temporal_subgraph_sequence(u, t, test_sampler, sample_size=samples)
    for vid, ts, eid in zip(neighbors1[1],times1[1],edges1[1]): # user的一阶邻居是item
        ts = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
        if args.empty:
            cur_sample.append((u, vid, ts))
        else:
            cur_sample.append((u, vid, ts, truncate_text(node_texts[vid]), truncate_text(edge_texts[eid])))
    for uid, ts, eid in zip(neighbors1[2],times1[2],edges1[2]): # user的二阶邻居是user
        ts = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
        vid = mappings1[2][uid]
        if args.empty:
            cur_sample.append((uid, vid, ts))
        else:
            cur_sample.append((uid, vid, ts, truncate_text(node_texts[vid]), truncate_text(edge_texts[eid])))

    _, _, neighbors2, times2, edges2, mappings2 = get_temporal_subgraph_sequence(v, t, test_sampler, sample_size=samples)
    for uid, ts, eid in zip(neighbors2[1],times2[1],edges2[1]): # item的一阶邻居是user
        ts = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
        if args.empty:
            cur_sample.append((uid, v, ts))
        else:
            cur_sample.append((uid, v, ts, truncate_text(node_texts[v]), truncate_text(edge_texts[eid])))
    for vid, ts, eid in zip(neighbors2[2],times2[2],edges2[2]): # item的二阶邻居是item
        ts = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
        uid = mappings2[2][vid]
        if args.empty:
            cur_sample.append((uid, vid, ts))
        else:
            cur_sample.append((uid, vid, ts, truncate_text(node_texts[vid]), truncate_text(edge_texts[eid])))

    t = datetime.datetime.fromtimestamp(t).strftime("%Y-%m-%d")
    # 当前问题
    if args.empty:
        question = (
            "Historical interactions: " + str(cur_sample) + "\n"
            f"Question: Will user {u} interact with item {v} at time '{t}'?"
        )
    else:   
        question = (
            "Historical comments: " + str(cur_sample) + "\n"
            f"Question: Will user {u} comment on {item_word.lower()} {v} at time '{t}'?"
        )

    # 组合提示
    prompt = (
        f"{dyg_instruction}\n\n"
        f"{task_instruction}\n\n"
        f"{dst2_instruction}\n\n"
        f"{answer_instruction}\n\n"
        f"{exemplar}\n\n"
        f"{question}"
    )
    return prompt

def truncate_text(text, max_words=15):
    """截断文本到指定单词数"""
    try:
        words = text.split()
        return " ".join(words[:max_words]) + "..." if len(words) > max_words else text
    except:
        return ''

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Interface for text generating by LLMs')
    parser.add_argument('--dataset_name', type=str, help='dataset to be used', default='FOOD')
    parser.add_argument('--k_hop', type=int, default=2, help='number of htops to use')
    parser.add_argument('--sample_size', type=int, default=2, help='number of neighbors to use')
    parser.add_argument('--gpu', type=int, default=3, help='number of gpu to use')
    parser.add_argument('--num_runs', type=int, default=2, help='number of runs')
    parser.add_argument('--empty', action='store_true', default=False)
    parser.add_argument('--variant', type=str, default='nondst2', choices=['nondst2','dst2-v1','dst2-v2'])
    args = parser.parse_args()
    print(args)

    dataset_name = args.dataset_name
    k_hop = args.k_hop
    sample_size = args.sample_size
    device = args.gpu
    device = torch.device(f'cuda:{device}')
    num_runs = args.num_runs
    # target_dir = f'../processed_data/{dataset_name}'
    train_ratio =  0.4
    val_ratio = 0.1

    # 加载明文信息，节点和边文本是原始未加一的，需要加一以与结构数据统一
    nodes_text = np.load(f'../DyGLLM/DG_data/{dataset_name}/raw_node.npy', allow_pickle=True) #加载节点原始文本
    empty = np.array([None])
    nodes_text = np.concatenate([empty, nodes_text]) # 添加上第零行的空值以与graph_df中的node id一致
    if args.empty:
        nodes_text[:] = None
    raw_edges = pd.read_csv(f'../DyGLLM/DG_data/{dataset_name}/raw_edges.csv', header=None,names=['edge_text'])
    raw_texts = raw_edges['edge_text'].values
    edge_texts = np.concatenate([empty, raw_texts]) # 添加上第零行的空值以与graph_df中的edge id一致
    if args.empty:
        edge_texts[:] = None
        
    # 加载结构数据，节点和边索引已加1了
    graph_df = pd.read_csv(f'../DyGLLM/processed_data/{dataset_name}/ml_{dataset_name}.csv') 
    # node_raw_features = np.load(f'dataset/{dataset_name}/{llm_name}_{dataset_name}_node.npy')
    val_time, test_time = list(np.quantile(graph_df.ts, [train_ratio, (train_ratio+val_ratio)]))

    src_node_ids = graph_df.u.values#.astype(np.longlong)
    dst_node_ids = graph_df.i.values#.astype(np.longlong)
    node_interact_times = graph_df.ts.values#.astype(np.float64)
    edge_ids = graph_df.idx.values#.astype(np.longlong)
    labels = graph_df.label.values

    full_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times, edge_ids=edge_ids, labels=labels) 

    # the setting of seed follows previous works
    random.seed(2020)

    # union to get node set
    node_set = set(src_node_ids) | set(dst_node_ids)
    num_total_unique_node_ids = len(node_set)

    # compute nodes which appear at test time
    test_node_set = set(src_node_ids[node_interact_times > val_time]).union(set(dst_node_ids[node_interact_times > val_time]))
    # sample nodes which we keep as new nodes (to test inductiveness), so then we have to remove all their edges from training
    new_test_node_set = set(random.sample(test_node_set, int(0.1 * num_total_unique_node_ids)))

    # mask for each source and destination to denote whether they are new test nodes
    new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
    new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values

    # mask, which is true for edges with both destination and source not being new test nodes (because we want to remove all edges involving any new test node)
    observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)

    # for train data, we keep edges happening before the validation time which do not involve any new node, used for inductiveness
    train_mask = np.logical_and(node_interact_times <= val_time, observed_edges_mask)

    train_data = Data(src_node_ids=src_node_ids[train_mask], dst_node_ids=dst_node_ids[train_mask],
                      node_interact_times=node_interact_times[train_mask],
                      edge_ids=edge_ids[train_mask], labels=labels[train_mask], seed=0)
    train_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=train_data.src_node_ids, dst_node_ids=train_data.dst_node_ids)
    # train_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(train_data.src_node_ids))), batch_size=1, shuffle=False)
    train_neighbor_sampler = get_neighbor_sampler(data=train_data, sample_neighbor_strategy='recent',
        time_scaling_factor=1e-6, seed=0)
    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy='recent',#与DyGLib保持一致
                                                 time_scaling_factor=1e-6, seed=1)

    # define the new nodes sets for testing inductiveness of the model
    train_node_set = set(train_data.src_node_ids).union(train_data.dst_node_ids)
    assert len(train_node_set & new_test_node_set) == 0
    # new nodes that are not in the training set
    new_node_set = node_set - train_node_set

    val_mask = np.logical_and(node_interact_times <= test_time, node_interact_times > val_time)
    test_mask = node_interact_times > test_time
    val_data = Data(src_node_ids=src_node_ids[val_mask], dst_node_ids=dst_node_ids[val_mask],
                    node_interact_times=node_interact_times[val_mask], edge_ids=edge_ids[val_mask], labels=labels[val_mask])

    # new edges with new nodes in the val and test set (for inductive evaluation)
    edge_contains_new_node_mask = np.array([(src_node_id in new_node_set or dst_node_id in new_node_set)
                                            for src_node_id, dst_node_id in zip(src_node_ids, dst_node_ids)])
    new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

    # transductive test data
    test_mask = np.logical_and(~edge_contains_new_node_mask, test_mask)
    test_data = Data(src_node_ids=src_node_ids[test_mask], dst_node_ids=dst_node_ids[test_mask],
                     node_interact_times=node_interact_times[test_mask], edge_ids=edge_ids[test_mask], labels=labels[test_mask])
    # inductive test data
    new_node_test_data = Data(src_node_ids=src_node_ids[new_node_test_mask], dst_node_ids=dst_node_ids[new_node_test_mask],
                              node_interact_times=node_interact_times[new_node_test_mask],
                              edge_ids=edge_ids[new_node_test_mask], labels=labels[new_node_test_mask])

    test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=2)
    new_node_test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_test_data.src_node_ids, dst_node_ids=new_node_test_data.dst_node_ids, seed=3)
    
    test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.src_node_ids))), batch_size=1, shuffle=False)
    new_node_test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_test_data.src_node_ids))), batch_size=1, shuffle=False) # batch_size设为1，便于将正负样本组合起来

    if dataset_name == 'FOOD':
        item_word = 'Recipe'
    elif dataset_name == 'IMDB':
        item_word = 'Movie'
    elif dataset_name == 'Librarything':
        item_word = 'Book'
    elif dataset_name in ['Beeradvocate', 'Ratebeer']:
        item_word = 'Beer'
    elif dataset_name == 'Amazon-Kindle':
        item_word = 'Product'

    target_dir = f'data/temprompt/{dataset_name}'
    if args.empty:
        target_dir += '-empty'
    os.makedirs(target_dir, exist_ok=True)

    for run in range(args.num_runs):
        set_random_seed(seed=run)
    
        # 1.transductive testing set处理
        id = 0
        result = []
        evaluate_idx_data_loader_tqdm = tqdm(test_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            # 从训练集中得到example

            evaluate_data_indices = evaluate_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                test_data.src_node_ids[evaluate_data_indices],  test_data.dst_node_ids[evaluate_data_indices], \
                test_data.node_interact_times[evaluate_data_indices], test_data.edge_ids[evaluate_data_indices]
            # 1. 处理positive edge
            u, v, t = batch_src_node_ids[0].item(), batch_dst_node_ids[0].item(), batch_node_interact_times[0].item()
            exemplar = get_train_samples(train_data, train_neighbor_sampler, nodes_text, edge_texts, samples=sample_size, args=args)
            prompt = generate_dst2_prompt(u, v, t, full_neighbor_sampler, nodes_text, edge_texts, exemplar, item_word=item_word, variant=args.variant, samples=sample_size, args=args)
            result.append({'id': f'{dataset_name}_train_{id}_{u}_and_{v}_LP', 'conversations': [{"from": "human", "value": prompt}, {"from": "gpt", "value": "yes"}],})
            id += 1
            
            # 2.处理negative edge
            _, neg_dst_id = test_neg_edge_sampler.sample(size=1)
            v = neg_dst_id.item()
            prompt = generate_dst2_prompt(u, v, t, full_neighbor_sampler, nodes_text, edge_texts, exemplar, item_word=item_word, variant=args.variant, samples=sample_size, args=args)
            result.append({'id': f'{dataset_name}_train_{id}_{u}_and_{v}_LP', 'conversations': [{"from": "human", "value": prompt}, {"from": "gpt", "value": "no"}],})
            id += 1

        filename = f"{args.variant}_lp_{k_hop}_{sample_size}_test{run}_transductive.json"
        with open(f'{target_dir}/{filename}', "w") as f:
            json.dump(result, f)

        # 2.inductive testing set处理
        id = 0
        result = []
        evaluate_idx_data_loader_tqdm = tqdm(new_node_test_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            evaluate_data_indices = evaluate_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                new_node_test_data.src_node_ids[evaluate_data_indices],  new_node_test_data.dst_node_ids[evaluate_data_indices], \
                new_node_test_data.node_interact_times[evaluate_data_indices], new_node_test_data.edge_ids[evaluate_data_indices]

            # 1.positive edge
            u, v, t = batch_src_node_ids[0].item(), batch_dst_node_ids[0].item(), batch_node_interact_times[0].item()
            exemplar = get_train_samples(train_data, train_neighbor_sampler, nodes_text, edge_texts, samples=sample_size, args=args)
            prompt = generate_dst2_prompt(u, v, t, full_neighbor_sampler, nodes_text, edge_texts, exemplar, item_word=item_word, variant=args.variant, samples=sample_size, args=args)
            result.append({'id': f'{dataset_name}_train_{id}_{u}_and_{v}_LP', 'conversations': [{"from": "human", "value": prompt}, {"from": "gpt", "value": "yes"}],})
            id += 1
            
            # 2.处理negative edge
            _, neg_dst_id = new_node_test_neg_edge_sampler.sample(size=1)
            v = neg_dst_id.item()
            prompt = generate_dst2_prompt(u, v, t, full_neighbor_sampler, nodes_text, edge_texts, exemplar, item_word=item_word, variant=args.variant, samples=sample_size, args=args)
            result.append({'id': f'{dataset_name}_train_{id}_{u}_and_{v}_LP', 'conversations': [{"from": "human", "value": prompt}, {"from": "gpt", "value": "no"}],})
            id += 1

        filename = f"{args.variant}_lp_{k_hop}_{sample_size}_test{run}_inductive.json"
        with open(f'{target_dir}/{filename}', "w") as f:
            json.dump(result, f)
            