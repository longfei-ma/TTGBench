import logging
import time
import sys
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import warnings
import shutil
import json
import torch
import torch.nn as nn

from models.TGAT import TGAT
from models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from models.CAWN import CAWN
from models.TCL import TCL
from models.GraphMixer import GraphMixer
from models.DyGFormer import DyGFormer
from models.modules import MergeLayer, MLPClassifier
from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes, create_optimizer
from utils.utils import get_neighbor_sampler
from evaluate_models_utils import evaluate_model_node_classification
from utils.metrics import get_node_classification_metrics
from utils.AblationDataLoader import get_idx_data_loader, get_node_classification_data
from utils.load_configs import get_node_classification_args

class EarlyStopping(object):

    def __init__(self, patience: int, save_model_folder: str, save_model_name: str, model_name: str = None):
        """
        Early stop strategy.
        :param patience: int, max patience
        :param save_model_folder: str, save model folder
        :param save_model_name: str, save model name
        :param logger: Logger
        :param model_name: str, model name
        """
        self.patience = patience
        self.counter = 0
        self.best_metrics = {}
        self.early_stop = False
        self.train_non_parameter_flag = False
        self.save_model_path = os.path.join(save_model_folder, f"{save_model_name}.pkl")
        self.model_name = model_name
        if self.model_name in ['JODIE', 'DyRep', 'TGN']:
            # path to additionally save the nonparametric data (e.g., tensors) in memory-based models (e.g., JODIE, DyRep, TGN)
            self.save_model_nonparametric_data_path = os.path.join(save_model_folder, f"{save_model_name}_nonparametric_data.pkl")

    def step(self, metrics: list, model: nn.Module):
        """
        execute the early stop strategy for each evaluation process
        :param metrics: list, list of metrics, each element is a tuple (str, float, boolean) -> (metric_name, metric_value, whether higher means better)
        :param model: nn.Module
        :return:
        """
        metrics_compare_results = []
        for metric_tuple in metrics:
            metric_name, metric_value, higher_better = metric_tuple[0], metric_tuple[1], metric_tuple[2]

            if higher_better:
                if self.best_metrics.get(metric_name) is None or metric_value >= self.best_metrics.get(metric_name):
                    metrics_compare_results.append(True)
                else:
                    metrics_compare_results.append(False)
            else:
                if self.best_metrics.get(metric_name) is None or metric_value <= self.best_metrics.get(metric_name):
                    metrics_compare_results.append(True)
                else:
                    metrics_compare_results.append(False)
        # all the computed metrics are better than the best metrics
        if torch.all(torch.tensor(metrics_compare_results)):
            for metric_tuple in metrics:
                metric_name, metric_value = metric_tuple[0], metric_tuple[1]
                self.best_metrics[metric_name] = metric_value
            self.save_checkpoint(model)
            self.counter = 0
            self.train_non_parameter_flag = True
        # metrics are not better at the epoch
        else:
            self.train_non_parameter_flag = False
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def save_checkpoint(self, model: nn.Module):
        """
        saves model at self.save_model_path
        :param model: nn.Module
        :return:
        """
        print(f"save model {self.save_model_path}")
        torch.save(model.state_dict(), self.save_model_path)
        if self.model_name in ['JODIE', 'DyRep', 'TGN']:
            torch.save(model[0].memory_bank.node_raw_messages, self.save_model_nonparametric_data_path)

    def load_checkpoint(self, model: nn.Module, map_location: str = None):
        """
        load model at self.save_model_path
        :param model: nn.Module
        :param map_location: str, how to remap the storage locations
        :return:
        """
        print(f"load model {self.save_model_path}")
        model.load_state_dict(torch.load(self.save_model_path, map_location=map_location))
        if self.model_name in ['JODIE', 'DyRep', 'TGN']:
            model[0].memory_bank.node_raw_messages = torch.load(self.save_model_nonparametric_data_path, map_location=map_location)
            
def touch(file_path):  
    """  
    类似于Linux中的touch命令，用于创建空文件
    """  
    with open(file_path, 'w'):  
        pass  # 不进行任何写入操作，仅创建文件（如果文件已存在，则清空其内容）

if __name__ == "__main__":

    warnings.filterwarnings('ignore')

    # get arguments
    args = get_node_classification_args()

    # get data for training, validation and testing
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, unique_labels,raw_labels = \
        get_node_classification_data(dataset_name=args.dataset_name, train_ratio=args.train_ratio, val_ratio=args.val_ratio, llm_name=args.llm_name, args=args)
    c = len(unique_labels)
    sample_labels = np.concatenate(np.array(raw_labels,dtype=object)) if args.dataset_name in ['FOOD', 'IMDB'] else train_data.labels #将multi-label的标签压扁成一维的
    class_sample_count = torch.bincount(torch.as_tensor(sample_labels[sample_labels>=0], dtype=int))
    weights = 1.0 / class_sample_count.float()
    weights = weights.to(args.device) 

    # initialize validation and test neighbor sampler to retrieve temporal graph
    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                 time_scaling_factor=args.time_scaling_factor, seed=1)

    # get data loaders
    test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)

    test_metric_all_runs = []

    for run in range(args.num_runs):
        set_random_seed(seed=run)

        args.seed = run
        save_model_name = f'seed{args.seed}'
        args.load_model_name = save_model_name
        run_start_time = time.time()
        print(f"********** Run {run + 1} starts. **********")

        print(f'configuration is {args}')

        # create model
        if args.model_name == 'TGAT':
            dynamic_backbone = TGAT(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                    time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout, device=args.device)
        elif args.model_name in ['JODIE', 'DyRep', 'TGN']:
            # four floats that represent the mean and standard deviation of source and destination node time shifts in the training data, which is used for JODIE
            src_node_mean_time_shift, src_node_std_time_shift, dst_node_mean_time_shift_dst, dst_node_std_time_shift = \
                compute_src_dst_node_time_shifts(train_data.src_node_ids, train_data.dst_node_ids, train_data.node_interact_times)
            dynamic_backbone = MemoryModel(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                           time_feat_dim=args.time_feat_dim, model_name=args.model_name, num_layers=args.num_layers, num_heads=args.num_heads,
                                           dropout=args.dropout, src_node_mean_time_shift=src_node_mean_time_shift, src_node_std_time_shift=src_node_std_time_shift,
                                           dst_node_mean_time_shift_dst=dst_node_mean_time_shift_dst, dst_node_std_time_shift=dst_node_std_time_shift, device=args.device)
        elif args.model_name == 'CAWN':
            dynamic_backbone = CAWN(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                    time_feat_dim=args.time_feat_dim, position_feat_dim=args.position_feat_dim, walk_length=args.walk_length,
                                    num_walk_heads=args.num_walk_heads, dropout=args.dropout, device=args.device)
        elif args.model_name == 'TCL':
            dynamic_backbone = TCL(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                   time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_heads,
                                   num_depths=args.num_neighbors + 1, dropout=args.dropout, device=args.device)
        elif args.model_name == 'GraphMixer':
            dynamic_backbone = GraphMixer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                          time_feat_dim=args.time_feat_dim, num_tokens=args.num_neighbors, num_layers=args.num_layers, dropout=args.dropout, device=args.device)
        elif args.model_name == 'DyGFormer':
            dynamic_backbone = DyGFormer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                         time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                         num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                         max_input_sequence_length=args.max_input_sequence_length, device=args.device)
        else:
            raise ValueError(f"Wrong value for model_name {args.model_name}!")
        node_classifier = MLPClassifier(input_dim=node_raw_features.shape[1], output_dim=c, dropout=args.dropout)
        model = nn.Sequential(dynamic_backbone, node_classifier)

        # load the saved model
        if args.empty:
            # load_model_folder = f"./saved_models_empty-{args.empty_ndim}_{args.empty_type}/{args.dataset_name}/{args.model_name}/{args.load_model_name}"
            load_model_folder = f"nc_saved_models-n{args.num_neighbors}_empty-{args.empty_ndim}/{args.dataset_name}/{args.model_name}/{args.load_model_name}"
        else:
            if args.train_ratio == 0.4: #这么写是为了统一之前跑的结果
                load_model_folder = f"./nc_saved_models-{args.llm_name}/{args.dataset_name}/{args.model_name}/{args.load_model_name}/"
            else:
                load_model_folder = f"nc_saved_models{args.train_ratio}-n{args.num_neighbors}_{args.llm_name}/{args.dataset_name}/{args.model_name}/{args.load_model_name}"
                
        early_stopping = EarlyStopping(patience=0, save_model_folder=load_model_folder,
                                        save_model_name=args.load_model_name, model_name=args.model_name)
        early_stopping.load_checkpoint(model, map_location='cpu')

        model = convert_to_gpu(model, device=args.device)
        # put the node raw messages of memory-based models on device
        if args.model_name in ['JODIE', 'DyRep', 'TGN']:
            for node_id, node_raw_messages in model[0].memory_bank.node_raw_messages.items():
                new_node_raw_messages = []
                for node_raw_message in node_raw_messages:
                    new_node_raw_messages.append((node_raw_message[0].to(args.device), node_raw_message[1]))
                model[0].memory_bank.node_raw_messages[node_id] = new_node_raw_messages

        if args.dataset_name in ['FOOD', 'IMDB']: #multi-label
            loss_func = nn.BCEWithLogitsLoss(pos_weight=weights)
        else: 
            loss_func = nn.CrossEntropyLoss(weight=weights)

        # evaluate the best model
        test_total_loss, test_metrics = evaluate_model_node_classification(model_name=args.model_name,
                                                                           model=model,
                                                                           neighbor_sampler=full_neighbor_sampler,
                                                                           evaluate_idx_data_loader=test_idx_data_loader,
                                                                           evaluate_data=test_data,
                                                                           loss_func=loss_func,
                                                                           num_neighbors=args.num_neighbors,
                                                                           time_gap=args.time_gap)

        # store the evaluation metrics at the current run
        test_metric_list = []

        # for metric_name in test_metrics.keys():
        #     test_metric = test_metrics[metric_name]
        #     print(f'test {metric_name}, {test_metric:.4f}')
        #     test_metric_list.append(test_metric)

        single_run_time = time.time() - run_start_time
        print(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

        # test_metric_all_runs.append(test_metric_list)
        test_metric_all_runs.append([test_metrics['acc'], test_metrics['micro-f1'], test_metrics['macro-f1'], test_metrics['prauc']])

    # save model result
    save_result_folder = f"./saved_results_nodes"
    os.makedirs(save_result_folder, exist_ok=True)

    test_metric_all_runs = np.array(test_metric_all_runs)*100
    # fstr = "\t"+"\t".join(test_metrics.keys())
    # print(fstr+'\n')
    # nms = test_metric_all_runs.shape[1]
    # fstr = f"{args.model_name}\t" + "\t".join([f"{test_metric_all_runs[:, i].mean():.2f} ± {test_metric_all_runs[:, i].std():.2f}" for i in range(nms)])
    # print(fstr+'\n')

    filename = f'./saved_results_nodes/{args.dataset_name}.csv'
    print(f"Saving results to {filename}")
    if args.empty:
        model_name = args.model_name + f'_empty-{args.empty_ndim}'
        # model_name = args.model_name + f'_empty-{args.empty_ndim}-n{args.num_neighbors}'
    else:    
        model_name = f'{args.model_name}_{args.llm_name}'
    with open(f"{filename}", 'a+') as write_obj:
        # for r in range(args.num_runs):
        #     write_obj.write(f"{model_name},run{r}," + f"{test_metric_all_runs[r].tolist()}\n")
                        
        write_obj.write(f"{model_name},avg," + 
                        f"{test_metric_all_runs[:, 0].mean():.2f} ± {test_metric_all_runs[:, 0].std():.2f}," +
                        f"{test_metric_all_runs[:, 1].mean():.2f} ± {test_metric_all_runs[:, 1].std():.2f}," +
                        f"{test_metric_all_runs[:, 2].mean():.2f} ± {test_metric_all_runs[:, 2].std():.2f}," +
                        f"{test_metric_all_runs[:, 3].mean():.2f} ± {test_metric_all_runs[:, 3].std():.2f}\n")

    