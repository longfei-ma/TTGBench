import torch
import time
import sys
import os
import numpy as np
import warnings
import json
import torch.nn as nn
import argparse

from models.EdgeBank import edge_bank_link_prediction
from models.TGAT import TGAT
from models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from models.CAWN import CAWN
from models.TCL import TCL
from models.GraphMixer import GraphMixer
from models.DyGFormer import DyGFormer
from models.modules import MergeLayer
from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes
from utils.utils import get_neighbor_sampler, NegativeEdgeSampler
from utils.metrics import get_link_prediction_metrics, get_node_classification_metrics
from evaluate_models_utils import evaluate_model_link_prediction_test
from utils.AblationDataLoader import get_idx_data_loader, get_link_prediction_data, Data
from utils.load_configs import get_link_prediction_args
from torch.utils.data import DataLoader
from tqdm import tqdm


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

def evaluate_edge_bank_link_prediction(args: argparse.Namespace, train_data: Data, val_data: Data, test_idx_data_loader: DataLoader,
                                       test_neg_edge_sampler: NegativeEdgeSampler, test_data: Data):
    """
    evaluate the EdgeBank model for link prediction
    :param args: argparse.Namespace, configuration
    :param train_data: Data, train data
    :param val_data: Data, validation data
    :param test_idx_data_loader: DataLoader, test index data loader
    :param test_neg_edge_sampler: NegativeEdgeSampler, test negative edge sampler
    :param test_data: Data, test data
    :return:
    """
    # generate the train_validation split of the data: needed for constructing the memory for EdgeBank
    train_val_data = Data(src_node_ids=np.concatenate([train_data.src_node_ids, val_data.src_node_ids]),
                          dst_node_ids=np.concatenate([train_data.dst_node_ids, val_data.dst_node_ids]),
                          node_interact_times=np.concatenate([train_data.node_interact_times, val_data.node_interact_times]),
                          edge_ids=np.concatenate([train_data.edge_ids, val_data.edge_ids]),
                          labels=np.concatenate([train_data.labels, val_data.labels]))

    test_metric_all_runs = []

    for run in range(args.num_runs):

        set_random_seed(seed=run)

        args.seed = run
        run_start_time = time.time()
        print(f"********** Run {run + 1} starts. **********")

        print(f'configuration is {args}')

        loss_func = nn.BCELoss()

        # evaluate EdgeBank
        print(f'get final performance on dataset {args.dataset_name}...')

        # Ensures the random sampler uses a fixed seed for evaluation (i.e. we always sample the same negatives for validation / test set)
        assert test_neg_edge_sampler.seed is not None
        test_neg_edge_sampler.reset_random_state()

        test_losses, test_metrics = [], []
        test_idx_data_loader_tqdm = tqdm(test_idx_data_loader, ncols=120)

        for batch_idx, test_data_indices in enumerate(test_idx_data_loader_tqdm):
            test_data_indices = test_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times = \
                test_data.src_node_ids[test_data_indices], test_data.dst_node_ids[test_data_indices], \
                test_data.node_interact_times[test_data_indices]

            if test_neg_edge_sampler.negative_sample_strategy != 'random':
                batch_neg_src_node_ids, batch_neg_dst_node_ids = test_neg_edge_sampler.sample(size=len(batch_src_node_ids),
                    batch_src_node_ids=batch_src_node_ids,
                    batch_dst_node_ids=batch_dst_node_ids,
                    current_batch_start_time=batch_node_interact_times[0],
                    current_batch_end_time=batch_node_interact_times[-1])
            else:
                _, batch_neg_dst_node_ids = test_neg_edge_sampler.sample(size=len(batch_src_node_ids))
                batch_neg_src_node_ids = batch_src_node_ids

            positive_edges = (batch_src_node_ids, batch_dst_node_ids)
            negative_edges = (batch_neg_src_node_ids, batch_neg_dst_node_ids)

            # incorporate the testing data before the current batch to history_data, which is similar to memory-based models
            history_data = Data(src_node_ids=np.concatenate([train_val_data.src_node_ids, test_data.src_node_ids[: test_data_indices[0]]]),
                                dst_node_ids=np.concatenate([train_val_data.dst_node_ids, test_data.dst_node_ids[: test_data_indices[0]]]),
                                node_interact_times=np.concatenate([train_val_data.node_interact_times, test_data.node_interact_times[: test_data_indices[0]]]),
                                edge_ids=np.concatenate([train_val_data.edge_ids, test_data.edge_ids[: test_data_indices[0]]]),
                                labels=np.concatenate([train_val_data.labels, test_data.labels[: test_data_indices[0]]]))

            # perform link prediction for EdgeBank
            positive_probabilities, negative_probabilities = edge_bank_link_prediction(history_data=history_data,
                    positive_edges=positive_edges,
                    negative_edges=negative_edges,
                    edge_bank_memory_mode=args.edge_bank_memory_mode,
                    time_window_mode=args.time_window_mode,
                    time_window_proportion=args.test_ratio)

            predicts = torch.from_numpy(np.concatenate([positive_probabilities, negative_probabilities])).float()
            labels = torch.cat([torch.ones(len(positive_probabilities)), torch.zeros(len(negative_probabilities))], dim=0)

            loss = loss_func(input=predicts, target=labels)

            test_losses.append(loss.item())

            test_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))

            test_idx_data_loader_tqdm.set_description(f'test for the {batch_idx + 1}-th batch, test loss: {loss.item()}')

        # store the evaluation metrics at the current run
        test_metric_dict = {}

        print(f'test loss: {np.mean(test_losses):.4f}')
        for metric_name in test_metrics[0].keys():
            average_test_metric = np.mean([test_metric[metric_name] for test_metric in test_metrics])
            print(f'test {metric_name}, {average_test_metric:.4f}')
            test_metric_dict[metric_name] = average_test_metric

        single_run_time = time.time() - run_start_time
        print(f'Run {run + 1} cost {single_run_time:.2f} seconds.')
        test_metric_all_runs.append([test_metric_dict['average_precision'], test_metric_dict['roc_auc']])

    return test_metric_all_runs, None


if __name__ == "__main__":

    warnings.filterwarnings('ignore')

    # get arguments
    args = get_link_prediction_args(is_evaluation=True)

    # get data for training, validation and testing
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = get_link_prediction_data(dataset_name=args.dataset_name, train_ratio=args.train_ratio, val_ratio=args.val_ratio, llm_name=args.llm_name, transductive=args.transductive, args=args)

    # initialize validation and test neighbor sampler to retrieve temporal graph
    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                 time_scaling_factor=args.time_scaling_factor, seed=1)

    # initialize negative samplers, set seeds for validation and testing so negatives are the same across different runs
    # in the inductive setting, negatives are sampled only amongst other new nodes
    if args.negative_sample_strategy != 'random':
        # val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids,
        #         interact_times=full_data.node_interact_times, last_observed_time=train_data.node_interact_times[-1],
        #         negative_sample_strategy=args.negative_sample_strategy, seed=0)
        # new_node_val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_val_data.src_node_ids, dst_node_ids=new_node_val_data.dst_node_ids,
        #         interact_times=new_node_val_data.node_interact_times, last_observed_time=train_data.node_interact_times[-1],
        #         negative_sample_strategy=args.negative_sample_strategy, seed=1)
        test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids,
                interact_times=full_data.node_interact_times, last_observed_time=val_data.node_interact_times[-1],
                negative_sample_strategy=args.negative_sample_strategy, seed=2)
        new_node_test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_test_data.src_node_ids, dst_node_ids=new_node_test_data.dst_node_ids,
                interact_times=new_node_test_data.node_interact_times, last_observed_time=val_data.node_interact_times[-1],
                negative_sample_strategy=args.negative_sample_strategy, seed=3)
    else:
        # val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=0)
        # new_node_val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_val_data.src_node_ids, dst_node_ids=new_node_val_data.dst_node_ids, seed=1)
        test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=2)
        new_node_test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_test_data.src_node_ids, dst_node_ids=new_node_test_data.dst_node_ids, seed=3)

    # get data loaders
    # val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    # new_node_val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    new_node_test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)

    # we separately evaluate EdgeBank, since EdgeBank does not contain any trainable parameters and has a different evaluation pipeline
    if args.model_name == 'EdgeBank':
        test_metric_all_runs, new_node_test_metric_all_runs = evaluate_edge_bank_link_prediction(args=args, train_data=train_data, val_data=val_data, test_idx_data_loader=test_idx_data_loader,
                                           test_neg_edge_sampler=test_neg_edge_sampler, test_data=test_data)

    else:
        test_metric_all_runs, new_node_test_metric_all_runs = [], []

        for run in range(args.num_runs):
            set_random_seed(seed=run)
            args.seed = run
            save_model_name = f'seed{args.seed}'
            # if args.empty:
            #     save_model_name = f'{args.model_name}_seed{args.seed}_dropout-{args.dropout}'
            # else:
            #     save_model_name = f'{args.llm_name}_{args.model_name}_seed{args.seed}_dropout-{args.dropout}'
        
            # if args.model_name in ['JODIE','TGAT','DyRep', 'TGN', 'TCL', 'GraphMixer', 'CAWN']:
            #     save_model_name += f'_num_neighbors-{args.num_neighbors}'
            # if args.model_name in ['TGAT','DyRep', 'TGN', 'TCL', 'GraphMixer']:
            #     save_model_name += f'_sample_neighbor_strategy-{args.sample_neighbor_strategy}'
            # if args.model_name == 'DyGFormer':
            #     save_model_name += f'_max_input_sequence_length-{args.max_input_sequence_length}_patch_size-{args.patch_size}'

            if args.walklm:
                save_model_name += '_walklm'
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
            link_predictor = MergeLayer(input_dim1=node_raw_features.shape[1], input_dim2=node_raw_features.shape[1],
                                        hidden_dim=node_raw_features.shape[1], output_dim=1)
            model = nn.Sequential(dynamic_backbone, link_predictor)
            print(f'model -> {model}')
            print(f'model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, '
                        f'{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.')

            # load the saved model
            if args.empty:
                # load_model_folder = f"./saved_models_empty-{args.empty_ndim}_{args.empty_type}/{args.dataset_name}/{args.model_name}/{args.load_model_name}"
                load_model_folder = f"saved_models-n{args.num_neighbors}_empty-{args.empty_ndim}_{args.empty_type}/{args.dataset_name}/{args.model_name}/{args.load_model_name}"
            else:
                # load_model_folder = f"./saved_models_{args.llm_name}/{args.dataset_name}/{args.model_name}/{args.load_model_name}"
                # load_model_folder = f"saved_models-n{args.num_neighbors}_{args.llm_name}/{args.dataset_name}/{args.model_name}/{args.load_model_name}"

                if args.train_ratio == 0.4: #这么写是为了统一之前跑的结果
                    load_model_folder = f"saved_models-n{args.num_neighbors}_{args.llm_name}/{args.dataset_name}/{args.model_name}/{args.load_model_name}"
                else:
                    load_model_folder = f"saved_models{args.train_ratio}-n{args.num_neighbors}_{args.llm_name}/{args.dataset_name}/{args.model_name}/{args.load_model_name}"

            # if args.user_aug:
            #     load_model_folder = 'user-aug' + load_model_folder

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

            loss_func = nn.BCELoss()

            # evaluate the best model
            print(f'get final performance on dataset {args.dataset_name}...')

            if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                # the memory in the best model has seen the validation edges, we need to backup the memory for new testing nodes
                val_backup_memory_bank = model[0].memory_bank.backup_memory_bank()

            test_losses, test_metrics = evaluate_model_link_prediction_test(model_name=args.model_name,
                        model=model,
                        neighbor_sampler=full_neighbor_sampler,
                        evaluate_idx_data_loader=test_idx_data_loader,
                        evaluate_neg_edge_sampler=test_neg_edge_sampler,
                        evaluate_data=test_data,
                        loss_func=loss_func,
                        num_neighbors=args.num_neighbors,
                        time_gap=args.time_gap)

            if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                # reload validation memory bank for new testing nodes
                model[0].memory_bank.reload_memory_bank(val_backup_memory_bank)

            new_node_test_losses, new_node_test_metrics = evaluate_model_link_prediction_test(model_name=args.model_name,
                        model=model,
                        neighbor_sampler=full_neighbor_sampler,
                        evaluate_idx_data_loader=new_node_test_idx_data_loader,
                        evaluate_neg_edge_sampler=new_node_test_neg_edge_sampler,
                        evaluate_data=new_node_test_data,
                        loss_func=loss_func,
                        num_neighbors=args.num_neighbors,
                        time_gap=args.time_gap)
            # store the evaluation metrics at the current run
            test_metric_dict, new_node_test_metric_dict = {}, {}

            print(f'test loss: {np.mean(test_losses):.4f}')
            for metric_name in test_metrics[0].keys():
                average_test_metric = np.mean([test_metric[metric_name] for test_metric in test_metrics])
                print(f'test {metric_name}, {average_test_metric:.4f}')
                test_metric_dict[metric_name] = average_test_metric

            print(f'new node test loss: {np.mean(new_node_test_losses):.4f}')
            for metric_name in new_node_test_metrics[0].keys():
                average_new_node_test_metric = np.mean([new_node_test_metric[metric_name] for new_node_test_metric in new_node_test_metrics])
                print(f'new node test {metric_name}, {average_new_node_test_metric:.4f}')
                new_node_test_metric_dict[metric_name] = average_new_node_test_metric

            single_run_time = time.time() - run_start_time
            print(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

            test_metric_all_runs.append([test_metric_dict['average_precision'], test_metric_dict['roc_auc'], test_metric_dict['f1_score']])
            new_node_test_metric_all_runs.append([new_node_test_metric_dict['average_precision'], new_node_test_metric_dict['roc_auc'], new_node_test_metric_dict['f1_score']])

    # save model result
    save_result_folder = f"./saved_results_ablation"
    os.makedirs(save_result_folder, exist_ok=True)

    test_metric_all_runs = np.array(test_metric_all_runs)*100

    filename = f'{save_result_folder}/{args.dataset_name}-{args.model_name}-transductive.csv'
    filename = f'{save_result_folder}/{args.dataset_name}-transductive.csv'
    print(f"Saving results to {filename}")
    if args.empty:
        model_name = args.model_name + f'_empty-{args.empty_ndim}_{args.empty_type}'
        model_name = args.model_name + f'_empty-{args.empty_ndim}_{args.empty_type}-n{args.num_neighbors}'
    else:    
        model_name = f'{args.model_name}_{args.llm_name}'
        model_name = f'{args.train_ratio},{args.model_name}_{args.llm_name}-n{args.num_neighbors}'
    if args.walklm:
        model_name += '-walklm'
    # if args.user_aug:
    #     model_name += '-user-aug'

    # with open(f"{filename}", 'a+') as write_obj:
    #     write_obj.write(f"{model_name}," + f"{args.negative_sample_strategy}," + 
    #                     f"{test_metric_all_runs[:, 0].mean():.2f} ± {test_metric_all_runs[:, 0].std():.2f}," +
    #                     f"{test_metric_all_runs[:, 1].mean():.2f} ± {test_metric_all_runs[:, 1].std():.2f}," +
    #                     f"{test_metric_all_runs[:, 2].mean():.2f} ± {test_metric_all_runs[:, 2].std():.2f}\n")
    # print(f"{args.dataset_name}-{model_name}-{args.llm_name}-{args.negative_sample_strategy}-transductive:\n" + 
    #     f"AP: {test_metric_all_runs[:, 0].mean():.2f} ± {test_metric_all_runs[:, 0].std():.2f}\n" +
    #     f"AUROC: {test_metric_all_runs[:, 1].mean():.2f} ± {test_metric_all_runs[:, 1].std():.2f}\n" + 
    #     f"F1: {test_metric_all_runs[:, 2].mean():.2f} ± {test_metric_all_runs[:, 2].std():.2f}\n")
    with open(f"{filename}", 'a+') as write_obj:
        if args.num_runs > 1:
            write_obj.write(f"{model_name}," + f"{args.negative_sample_strategy}," + 
                            f"{test_metric_all_runs[:, 0].mean():.2f} ± {test_metric_all_runs[:, 0].std():.2f}," +
                            f"{test_metric_all_runs[:, 1].mean():.2f} ± {test_metric_all_runs[:, 1].std():.2f}\n")
        else:
            write_obj.write(f"{model_name}," + f"{args.negative_sample_strategy}," + 
                            f"{test_metric_all_runs[:, 0].mean():.2f}," +
                            f"{test_metric_all_runs[:, 1].mean():.2f}\n")
    print(f"{args.dataset_name}-{model_name}-{args.llm_name}-{args.negative_sample_strategy}-transductive:\n" + 
        f"AP: {test_metric_all_runs[:, 0].mean():.2f} ± {test_metric_all_runs[:, 0].std():.2f}\n" +
        f"AUROC: {test_metric_all_runs[:, 1].mean():.2f} ± {test_metric_all_runs[:, 1].std():.2f}\n")

    if new_node_test_metric_all_runs is not None: # 不是EdgeBank时评估下inductive数据, EdgeBank时该项为None
        new_node_test_metric_all_runs = np.array(new_node_test_metric_all_runs)*100
        filename = f'{save_result_folder}/{args.dataset_name}-{args.model_name}-inductive.csv'
        filename = f'{save_result_folder}/{args.dataset_name}-inductive.csv'
        print(f"Saving results to {filename}")
        with open(f"{filename}", 'a+') as write_obj:
            if args.num_runs > 1:
                write_obj.write(f"{model_name},{args.negative_sample_strategy}," + 
                            f"{new_node_test_metric_all_runs[:, 0].mean():.2f} ± {new_node_test_metric_all_runs[:, 0].std():.2f}," +
                            f"{new_node_test_metric_all_runs[:, 1].mean():.2f} ± {new_node_test_metric_all_runs[:, 1].std():.2f}\n")
            else:
                write_obj.write(f"{model_name},{args.negative_sample_strategy}," + 
                            f"{new_node_test_metric_all_runs[:, 0].mean():.2f}," +
                            f"{new_node_test_metric_all_runs[:, 1].mean():.2f}\n")
        
        print(f"{args.dataset_name}-{model_name}-{args.llm_name}-{args.negative_sample_strategy}-inductive:\n" + 
            f"AP: {new_node_test_metric_all_runs[:, 0].mean():.2f} ± {new_node_test_metric_all_runs[:, 0].std():.2f}\n" +
            f"AUROC: {new_node_test_metric_all_runs[:, 1].mean():.2f} ± {new_node_test_metric_all_runs[:, 1].std():.2f}\n")
        # with open(f"{filename}", 'a+') as write_obj:
        #     write_obj.write(f"{model_name},{args.negative_sample_strategy}," + 
        #                 f"{new_node_test_metric_all_runs[:, 0].mean():.2f} ± {new_node_test_metric_all_runs[:, 0].std():.2f}," +
        #                 f"{new_node_test_metric_all_runs[:, 1].mean():.2f} ± {new_node_test_metric_all_runs[:, 1].std():.2f}," +
        #                 f"{new_node_test_metric_all_runs[:, 2].mean():.2f} ± {new_node_test_metric_all_runs[:, 2].std():.2f}\n")
        
        # print(f"{args.dataset_name}-{model_name}-{args.llm_name}-{args.negative_sample_strategy}-inductive:\n" + 
        #     f"AP: {new_node_test_metric_all_runs[:, 0].mean():.2f} ± {new_node_test_metric_all_runs[:, 0].std():.2f}\n" +
        #     f"AUROC: {new_node_test_metric_all_runs[:, 1].mean():.2f} ± {new_node_test_metric_all_runs[:, 1].std():.2f}\n" +
        #     f"F1: {new_node_test_metric_all_runs[:, 2].mean():.2f} ± {new_node_test_metric_all_runs[:, 2].std():.2f}\n")