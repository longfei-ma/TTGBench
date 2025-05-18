import sys
sys.path.append(".")
sys.path.append("..")

import argparse
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import torch
import torch.nn as nn
import os
from graphgpt.conversation import conv_templates, SeparatorStyle
from graphgpt.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
from graphgpt.model import *
from graphgpt.model.utils import KeywordsStoppingCriteria
from torch_geometric.data import Data
import json
import copy

import os
import random
import numpy as np

from tqdm import tqdm
import json
import os.path as osp

# os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

DEFAULT_GRAPH_TOKEN = "<graph>"
DEFAULT_GRAPH_PATCH_TOKEN = "<g_patch>"
DEFAULT_G_START_TOKEN = "<g_start>"
DEFAULT_G_END_TOKEN = "<g_end>"


def load_graph(instruct_item, pretrained_embs, task): 
    if task != 'lp': #Node classification
        graph_dict = instruct_item['graph']
        graph_edge_index = torch.Tensor(copy.deepcopy(graph_dict['edge_index'])).long()#中心节点与邻居节点构成子图后并将节点重新编号了(从0开始)
        graph_node_list = copy.deepcopy(graph_dict['node_list'])#中心节点及邻居节点的原始编号
        target_node = copy.deepcopy(graph_dict['node_idx'])#中心节点原始编号
        graph_type = copy.deepcopy(instruct_item['id']).split('_')[0]
        graph_node_rep = pretrained_embs[graph_node_list] ## 取出子图节点的embedding
        
        cur_token_len = len(graph_node_rep)   # FIXME: 14 is hardcoded patch size,中心节点及邻居节点数量
        graph_ret = Data(graph_node = graph_node_rep, edge_index=graph_edge_index, target_node = torch.tensor([target_node]))
        return {
            'graph_data': graph_ret, 
            'graph_token_len': cur_token_len,
        }
    else: 
        graph_dict = instruct_item['graph']
        graph_edge_index_1 = torch.Tensor(copy.deepcopy(graph_dict['edge_index_1'])).long()#第一个子图的edge_index，从0重新编号了
        graph_node_list_1 = copy.deepcopy(graph_dict['node_list_1'])#第一个子图的节点列表，原始编号
        target_node_1 = copy.deepcopy(graph_dict['node_idx_1'])#第一个子图的中心节点，原始编号
        graph_type = copy.deepcopy(instruct_item['id']).split('_')[0] #graph数据集名称，如pubmed
        graph_node_rep_1 = pretrained_embs[graph_node_list_1] ## 取出第一个子图节点的embedding
        
        cur_token_len_1 = len(graph_node_rep_1)   # FIXME: 14 is hardcoded patch size.分别计算两个图的 Token 长度（即节点数量）

        graph_edge_index_2 = torch.Tensor(copy.deepcopy(graph_dict['edge_index_2'])).long()#第二个子图的edge_index，从0重新编号了
        graph_node_list_2 = copy.deepcopy(graph_dict['node_list_2'])#第二个子图的节点列表，原始编号
        target_node_2 = copy.deepcopy(graph_dict['node_idx_2'])#第二个子图的中心节点，原始编号
        graph_node_rep_2 = pretrained_embs[graph_node_list_2] ## 节点特征
        cur_token_len_2 = len(graph_node_rep_2)   # FIXME: 14 is hardcoded patch size
    
        graph_ret = {
                    'graph_1': Data(graph_node = graph_node_rep_1, edge_index=graph_edge_index_1, target_node = torch.tensor([target_node_1])), #graph_edge_index_1是子图edge_index从0编号的，target_node是原始编号，graph_node是子图节点的embedding（是否从0编号无所谓）
                    'graph_2': Data(graph_node = graph_node_rep_2, edge_index=graph_edge_index_2, target_node = torch.tensor([target_node_2]))
                    }
        return {
            'graph_data': graph_ret, 
            'graph_token_len1': cur_token_len_1,
            'graph_token_len2': cur_token_len_2
        }
        
    # graph_data_all = torch.load(graph_data_path)
    # graph_dict = instruct_item['graph']
    # graph_edge_index = torch.Tensor(copy.deepcopy(graph_dict['edge_index'])).long()
    # graph_node_list = copy.deepcopy(graph_dict['node_list'])
    # target_node = copy.deepcopy(graph_dict['node_idx'])
    # graph_type = copy.deepcopy(instruct_item['id']).split('_')[0]
    # graph_node_rep = graph_data_all[graph_type].x[graph_node_list] ## 
    
    # cur_token_len = len(graph_node_rep)   # FIXME: 14 is hardcoded patch size

    # graph_ret = Data(graph_node = graph_node_rep, edge_index=graph_edge_index, target_node = torch.tensor([target_node]))

    # return {
    #     'graph_data': graph_ret, 
    #     'graph_token_len': cur_token_len
    # }


def load_prompting_file(file_path): 
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# def prepare_query(instruct_item): 


def run_eval(args, num_gpus):
    # split question file into num_gpus files
    prompt_file = load_prompting_file(args.prompting_file)
    prompt_file = prompt_file[args.start_id:args.end_id]
    chunk_size = len(prompt_file) // num_gpus
    ans_handles = []
    split_list = list(range(args.start_id, args.end_id, chunk_size))
    idx_list = list(range(0, len(prompt_file), chunk_size))
    if len(split_list) == num_gpus: 
        split_list.append(args.end_id)
        idx_list.append(len(prompt_file))
    elif len(split_list) == num_gpus + 1: 
        split_list[-1] = args.end_id
        idx_list[-1] = len(prompt_file)
    else: 
        raise ValueError('error in the number of list')

    if osp.exists(args.output_res_path) is False: 
        os.mkdir(args.output_res_path)
    
    for idx in range(len(idx_list) - 1):
        start_idx = idx_list[idx]
        end_idx = idx_list[idx + 1]
        
        start_split = split_list[idx]
        end_split = split_list[idx + 1]
        ans_handles.append(
            eval_model.remote(
                args, prompt_file[start_idx:end_idx], start_split, end_split
            )
        )

    ans_jsons = []
    for ans_handle in ans_handles:
        ans_jsons.extend(ray.get(ans_handle))

    # with open(args.output_res_path, "w") as ans_file:
    #     for line in ans_jsons:
    #         ans_file.write(json.dumps(line) + "\n")


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

def load_pretrain_embedding_graph(args):
    data_dir = f'../DyGLLM/processed_data/{args.dataset}'
    pretrained_emb = torch.as_tensor(np.load(os.path.join(data_dir, f"{args.pretrained_embedding_type}_{args.dataset}_node.npy")), dtype=torch.float)
    if args.empty: # 
        pretrained_emb = torch.zeros_like(pretrained_emb)
    return pretrained_emb # torch.Size([2708, 384])

# @ray.remote(num_gpus=1)
@torch.inference_mode()
def eval_model(args):
    # load prompting file
    # prompt_file = load_prompting_file(args.prompting_file)
    run = args.run
    set_random_seed(seed=run)
    args.seed = run
    # Model
    disable_torch_init()
    model_path = f"{args.model_path}{run}"
    if args.raw:
        model_path = model_path.replace(args.dataset, f'{args.dataset}-raw')
        args.output_path = f'{args.output_path}-raw'
    if args.empty:
        model_path = model_path.replace(args.pretrained_embedding_type, f'empty-{args.empty_ndim}')
    # model_name = os.path.expanduser(args.model_name)
    print('start loading tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(args.cache_dir, model_max_length=args.model_max_length, padding_side="right")
    print('finish loading tokenizer')


    print('start loading model')
    # 1. 使用预训练模型的配置加载模型
    model = GraphLlamaForCausalLM.from_pretrained(args.cache_dir, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    # cfg_pretrained = AutoConfig.from_pretrained(model_path)
    # model = GraphLlamaForCausalLM.from_pretrained(args.cache_dir, config=cfg_pretrained, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    # 2. 加载本地保存的配置并更新模型配置
    local_config = AutoConfig.from_pretrained(model_path)
    new_vocab_size = local_config.vocab_size  # 例如 32003
    original_vocab_size = model.config.vocab_size  # 例如 32000
    model.config.update(local_config.__dict__)
    model.model.orig_embeds_params = [model.get_input_embeddings().weight.data.clone()]

    if new_vocab_size > original_vocab_size:
        # 获取当前嵌入层
        embed_tokens = model.get_input_embeddings()
        # 扩展嵌入层到新大小
        new_embed_tokens = torch.nn.Embedding(new_vocab_size, embed_tokens.embedding_dim).to(torch.float16)
        # 复制原始权重（稍后会被保存的参数覆盖）
        with torch.no_grad():
            new_embed_tokens.weight[:original_vocab_size] = embed_tokens.weight
            # 新增 token 暂时用随机值初始化（将被 graph_projector.bin 覆盖）
            new_embed_tokens.weight[original_vocab_size:] = torch.randn(
                new_vocab_size - original_vocab_size, embed_tokens.embedding_dim
            ).to(embed_tokens.weight.dtype)
        model.set_input_embeddings(new_embed_tokens)
        
    # 3.2 添加额外模块（根据本地配置）
    model.model.graph_projector = nn.Linear(model.config.graph_hidden_size, model.config.hidden_size).to(torch.float16)
    mm_projector_weights = torch.load(os.path.join(model_path, 'graph_projector.bin'), map_location='cpu')
    mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
    model.load_state_dict(mm_projector_weights, strict=False)
    print('finish loading model')
    
    # model.get_model().initialize_graph_modules(#该函数作用是将预训练的CLIP模型加载到GraphLlamaModel.graph_tower和给GraphLlamaModel初始化graph_projector(从节点embedding到token embedding的投影Linear层)
    #         graph_tower=args.graph_tower, #'clip_gt'
    #         graph_select_layer=local_config.graph_select_layer,
    #         run=run,
    #         llm_name=args.pretrained_embedding_type
    #     )


    model = model.cuda()
    use_graph_start_end = getattr(model.config, "use_graph_start_end", False)
    tokenizer.add_tokens([DEFAULT_GRAPH_PATCH_TOKEN], special_tokens=True)
    if use_graph_start_end:
        tokenizer.add_tokens([DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN], special_tokens=True)

    # graph_tower = model.get_model().graph_tower
    
    clip_graph, args_graph= load_model_pretrained(CLIP, args.pretrain_graph_model_path, run, args.graph_tower, args)
    if args.graph_tower == "clip_gcn": 
        graph_tower = GNN(args_graph)
    elif args.graph_tower == "clip_gt":
        graph_tower = graph_transformer(args_graph)
    graph_tower = transfer_param_tograph(clip_graph, graph_tower)
    
    model.get_model().graph_tower = graph_tower.to(device='cuda', dtype=torch.float16)
    # graph_tower.to(device='cuda', dtype=torch.float16)
    graph_config = model.get_model().graph_tower.config = args_graph
    graph_config.graph_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_GRAPH_PATCH_TOKEN])[0]
    graph_config.use_graph_start_end = model.config.use_graph_start_end
    if graph_config.use_graph_start_end:
        graph_config.graph_start_token, graph_config.graph_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN])
    # # TODO: add graph token len

    pretrained_embs = load_pretrain_embedding_graph(args)
    if args.raw:
        data_dir = f"data/stage_2/{args.dataset}-raw"
    else:
        data_dir = f"data/stage_2/{args.dataset}"

    if args.empty:
        data_dir = data_dir.replace(args.dataset, f'{args.dataset}-empty')
        
    if args.task == 'nc': #
        print('generating testing answer files ...')
        prompt_file = os.path.join(data_dir, f"{args.task}_2_{args.sample_neighbor_size}_test{run}.json")
        answers_file = f'{args.output_path}/{args.dataset}-answers-test{run}-{args.task}-{args.pretrained_embedding_type}-{args.current_part}.json'
        if args.empty:
            answers_file = answers_file.replace(args.pretrained_embedding_type, 'empty')
        os.makedirs(os.path.dirname(answers_file), exist_ok=True)
        ans_file = open(answers_file, "w")
        lines = open(prompt_file, "r").readlines()
        questions = json.loads(lines[0])
        # 获取列表总长度
        total_rows = len(questions)
        print(f'total: {total_rows}')
        
        # 计算每份的大小
        part_size = total_rows // args.num_parts
        remainder = total_rows % args.num_parts
        
        # 计算当前部分的起止索引
        start_idx = args.current_part * part_size + min(args.current_part, remainder)
        end_idx = start_idx + part_size + (1 if args.current_part < remainder else 0)
        
        # 从list中取出当前部分
        questions = questions[start_idx:end_idx]
        print(f'total of current part: {len(questions)}')
        for instruct_item in tqdm(questions):
            idx = instruct_item["id"]
            graph_dict = load_graph(instruct_item, pretrained_embs, args.task)
            graph_data = graph_dict['graph_data']
            graph_token_len = graph_dict['graph_token_len']

            replace_token = DEFAULT_GRAPH_PATCH_TOKEN * graph_token_len
            replace_token = DEFAULT_G_START_TOKEN + replace_token + DEFAULT_G_END_TOKEN

            qs = instruct_item["conversations"][0]["value"]
            first_index = qs.find(DEFAULT_GRAPH_TOKEN)
            qs = qs[:first_index] + replace_token + qs[first_index+len(DEFAULT_GRAPH_TOKEN):]

            conv_mode = "vicuna_v1_1"

            if args.conv_mode is not None and conv_mode != args.conv_mode:
                print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
            else:
                args.conv_mode = conv_mode
            cur_prompt = qs

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            inputs = tokenizer([prompt])

            input_ids = torch.as_tensor(inputs.input_ids).cuda()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            graph_data.graph_node = graph_data.graph_node.to(torch.float16)
            # graph_data.edge_index = graph_data.edge_index.to(torch.float16)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    graph_data=graph_data,
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=1024,
                    stopping_criteria=[stopping_criteria])

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            
            # print(outputs)

            ans_file.write(json.dumps({"question_id": idx,
                                "prompt": cur_prompt,
                                "graph": instruct_item['graph'],
                                "text": outputs,
                                "gt":instruct_item["conversations"][1]['value'],
                                }) + "\n")
            ans_file.flush()
        ans_file.close()
    elif args.task == 'lp':
        # 分别评估transductive和inductive
        # for testing in ['transductive', 'inductive']:
        for testing in [args.test_mode]:
            print(f'generating {testing} answer files ...')
            # for prompt_file in [prompt_file_transductive, prompt_file_inductive]:  
            prompt_file = os.path.join(data_dir, f"{args.task}_2_{args.sample_neighbor_size}_test{run}_{testing}.json")
            answers_file = f'{args.output_path}/{args.dataset}-answers-{testing}{run}-{args.task}-{args.pretrained_embedding_type}-{args.current_part}.json'
            if args.empty:
                answers_file = answers_file.replace(args.pretrained_embedding_type, 'empty')
            os.makedirs(os.path.dirname(answers_file), exist_ok=True)
            ans_file = open(answers_file, "w")
            lines = open(prompt_file, "r").readlines()
            questions = json.loads(lines[0])
            # 获取列表总长度
            total_rows = len(questions)
            print(f'total: {total_rows}')
            
            # 计算每份的大小
            part_size = total_rows // args.num_parts
            remainder = total_rows % args.num_parts
            
            # 计算当前部分的起止索引
            start_idx = args.current_part * part_size + min(args.current_part, remainder)
            end_idx = start_idx + part_size + (1 if args.current_part < remainder else 0)
            
            # 从list中取出当前部分
            questions = questions[start_idx:end_idx]
            print(f'total of current part: {len(questions)}')
            for instruct_item in tqdm(questions):
                idx = instruct_item["id"]
                graph_dict = load_graph(instruct_item, pretrained_embs, args.task)
                graph_data = graph_dict['graph_data']
                graph_token_len_1 = graph_dict['graph_token_len1']
                graph_token_len_2 = graph_dict['graph_token_len2']

                replace_token_1 = DEFAULT_GRAPH_PATCH_TOKEN * graph_token_len_1
                replace_token_2 = DEFAULT_GRAPH_PATCH_TOKEN * graph_token_len_2
                replace_token_1 = DEFAULT_G_START_TOKEN + replace_token_1 + DEFAULT_G_END_TOKEN
                replace_token_2 = DEFAULT_G_START_TOKEN + replace_token_2 + DEFAULT_G_END_TOKEN

                qs = instruct_item["conversations"][0]["value"]
                first_index = qs.find(DEFAULT_GRAPH_TOKEN)
                qs = qs[:first_index] + replace_token_1 + qs[first_index+len(DEFAULT_GRAPH_TOKEN):]

                # 替换第二个<graph>为B
                second_index = qs.find(DEFAULT_GRAPH_TOKEN)
                qs = qs[:second_index] + replace_token_2 + qs[second_index+len(DEFAULT_GRAPH_TOKEN):]

                conv_mode = "vicuna_v1_1"

                if args.conv_mode is not None and conv_mode != args.conv_mode:
                    print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
                else:
                    args.conv_mode = conv_mode
                cur_prompt = qs

                conv = conv_templates[args.conv_mode].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                inputs = tokenizer([prompt])

                input_ids = torch.as_tensor(inputs.input_ids).cuda()

                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
                
                for k in graph_data:
                    graph_data[k].graph_node = graph_data[k].graph_node.to(torch.float16)
                # graph_data.edge_index = graph_data.edge_index.to(torch.float16)

                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        graph_data=graph_data,
                        do_sample=True,
                        temperature=0.2,
                        max_new_tokens=1024,
                        stopping_criteria=[stopping_criteria])

                input_token_len = input_ids.shape[1]
                n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
                if n_diff_input_output > 0:
                    print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
                outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
                outputs = outputs.strip()
                if outputs.endswith(stop_str):
                    outputs = outputs[:-len(stop_str)]
                outputs = outputs.strip()
                # print(outputs)

                ans_file.write(json.dumps({"question_id": idx,
                                    "prompt": cur_prompt,
                                    "graph": instruct_item['graph'],
                                    "text": outputs,
                                    "gt":instruct_item["conversations"][1]['value'],
                                    }) + "\n")
                ans_file.flush()
            ans_file.close()
    else:
        raise ValueError
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="FOOD")
    parser.add_argument('--run', type=int, default=0, help='number of run')
    parser.add_argument('--num_parts', type=int, default=4, help='number of data parts')
    parser.add_argument('--current_part', type=int, default=0, help='current part number')
    parser.add_argument("--task", type=str, default="nc")
    parser.add_argument("--pretrained_embedding_type", type=str, default="sbert")
    parser.add_argument("--model_max_length", type=int, default=4096)
    parser.add_argument("--model_path", type=str, default="facebook/opt-350m")
    parser.add_argument("--cache_dir", type=str, default="facebook/opt-350m")
    parser.add_argument("--pretrain_graph_model_path", type=str, default=None)
    parser.add_argument("--graph_tower", type=str, default="clip_gcn")
    parser.add_argument('--empty', action='store_true', default=False)
    parser.add_argument('--empty_ndim', type=int, default=384)
    parser.add_argument('--raw', action='store_true', default=False)
    parser.add_argument("--conv_mode", type=str, default=None)
    parser.add_argument("--sample_neighbor_size", type=int, default=2)
    parser.add_argument("--output_path", type=str, default="results")
    parser.add_argument("--llm_name", type=str, default='vicuna')
    parser.add_argument("--test_mode", type=str, default='transductive', choices=['transductive', 'inductive'])

    args = parser.parse_args()

    eval_model(args)

    # ray.init()
    # run_eval(args, args.num_gpus)


# protobuf             4.22.3