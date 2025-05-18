import pandas as pd
import numpy as np
import random
import argparse
import networkx as nx

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

    def __init__(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray, edge_ids: np.ndarray, labels: np.ndarray):
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

    def __getitem__(self, index):
        # 通过索引访问元素
        return (self.src_node_ids[index].item(), self.dst_node_ids[index].item(), self.node_interact_times[index].item())

def get_fix_shape_subgraph_sequence_fast(node_idx, recent_neighbors, k_hop, sample_size, avoid_idx=None):
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

    for t in range(k_hop):
        last_hop = neighbors[-1]  # 获取上一跳的邻居
        current_hop = []  # 初始化当前跳的邻居

        for i in last_hop:
            if i == DEFAULT_GRAPH_PAD_ID:
                current_hop.extend([DEFAULT_GRAPH_PAD_ID] * sample_size)
                continue

            # 获取节点 i 的邻居，邻居已按时间顺序排序
            node_neighbor = list(recent_neighbors[i])
            
            # 如果是第一跳且需要避免某个节点，则从邻居中移除
            if t == 0 and avoid_idx is not None and avoid_idx in node_neighbor:
                node_neighbor.remove(avoid_idx)

            # 采样邻居
            if len(node_neighbor) >= sample_size:
                sampled_neighbor = node_neighbor[-sample_size:]  # 只取最近的 sample_size 个邻居
            else:
                sampled_neighbor = node_neighbor + [DEFAULT_GRAPH_PAD_ID] * (sample_size - len(node_neighbor))

            current_hop.extend(sampled_neighbor)

        neighbors.append(current_hop)

    # 展平邻居列表
    node_sequence = [n for hop in neighbors for n in hop]
    return node_sequence


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Interface for text generating by LLMs')
    parser.add_argument('--dataset_name', type=str, help='dataset to be used', default='FOOD')
    args = parser.parse_args()
    print(args)

    dataset_name = args.dataset_name
    # target_dir = f'../processed_data/{dataset_name}'
    train_ratio =  0.4
    val_ratio = 0.1

    # 数据中边的节点号已经加1了
    graph_df = pd.read_csv(f'dataset/{dataset_name}/ml_{dataset_name}.csv') # 结构数据对所有语言模型都一样
    # node_raw_features = np.load(f'dataset/{dataset_name}/{llm_name}_{dataset_name}_node.npy')
    val_time, test_time = list(np.quantile(graph_df.ts, [train_ratio, (train_ratio+val_ratio)]))

    src_node_ids = graph_df.u.values#.astype(np.longlong)
    dst_node_ids = graph_df.i.values#.astype(np.longlong)
    node_interact_times = graph_df.ts.values#.astype(np.float64)
    edge_ids = graph_df.idx.values#.astype(np.longlong)
    labels = graph_df.label.values

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
                      edge_ids=edge_ids[train_mask], labels=labels[train_mask])
    
    
    G = nx.Graph()
    for u, v, t in train_data:
        G.add_edge(u, v)
    
    edge_list = list(G.edges())

    # 将边转换为 edge_index 格式，节点号是加1后的
    edge_index = np.array(edge_list).T  # 转置为 [2, num_edges]
    np.save(edge_index, f"data/{dataset_name}_edge.npy")