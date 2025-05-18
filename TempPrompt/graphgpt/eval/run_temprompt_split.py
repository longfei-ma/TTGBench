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


# @ray.remote(num_gpus=1)
@torch.inference_mode()
def eval_model(args):
    # load prompting file
    # prompt_file = load_prompting_file(args.prompting_file)
    run = args.run
    set_random_seed(seed=run)
    args.seed = run
    
    # model_name = os.path.expanduser(args.model_name)
    print('start loading tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(args.cache_dir, model_max_length=args.model_max_length, padding_side="right")
    print('finish loading tokenizer')

    print('start loading model')
    model = AutoModelForCausalLM.from_pretrained(args.cache_dir, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    print('finish loading model')
    model = model.cuda()
    data_dir = f"data/temprompt/{args.dataset}"
    if args.empty:
        data_dir += '-empty'
        
    if args.task == 'nc': #
        print(f'generating part-{args.current_part} of {args.num_parts} testing answer files ...')
        # for prompt_file in [prompt_file_transductive, prompt_file_inductive]:  
        prompt_file = os.path.join(data_dir, f"{args.variant}_{args.task}_2_{args.sample_neighbor_size}_test{run}.json")
        answers_file = f'{args.output_path}/{args.dataset}-answers-test{run}-{args.variant}-{args.task}-{args.current_part}.json'
        if args.empty:
            answers_file = answers_file.replace(args.dataset, f'{args.dataset}-empty')
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
        for instruct_item in tqdm(questions):
            idx = instruct_item["id"]
            qs = instruct_item["conversations"][0]["value"]
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
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    # do_sample=True,
                    # temperature=0.2,
                    do_sample=False,            # 禁用采样，使用贪婪解码
                    temperature=1.0,            # 默认温度（贪婪解码下不起作用）
                    top_k=0,                    # 禁用 top-k（贪婪解码下不起作用）
                    top_p=1.0,                  # 禁用 top-p（贪婪解码下不起作用）
                    max_new_tokens=1024,
                    )
            # 解码生成的文本
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            # 提取 Answer: 后的内容
            answer_text = generated_text.rsplit(":",1)[-1].strip()
            ans_file.write(json.dumps({"question_id": idx,
                                # "prompt": cur_prompt,
                                "text": answer_text,
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
            prompt_file = os.path.join(data_dir, f"{args.variant}_{args.task}_2_{args.sample_neighbor_size}_test{run}_{testing}.json")
            answers_file = f'{args.output_path}/{args.dataset}-answers-{testing}{run}-{args.variant}-{args.task}-{args.current_part}.json'
            if args.empty:
                answers_file = answers_file.replace(args.dataset, f'{args.dataset}-empty')
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
            for instruct_item in tqdm(questions):
                idx = instruct_item["id"]
                qs = instruct_item["conversations"][0]["value"]
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
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        # do_sample=True,
                        # temperature=0.2,
                        do_sample=False,            # 禁用采样，使用贪婪解码
                        temperature=1.0,            # 默认温度（贪婪解码下不起作用）
                        top_k=0,                    # 禁用 top-k（贪婪解码下不起作用）
                        top_p=1.0,                  # 禁用 top-p（贪婪解码下不起作用）
                        max_new_tokens=1024,
                        )
                # 解码生成的文本
                generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                
                # 提取 Answer: 后的内容
                answer_start = generated_text.rfind("Answer:",1)
                if answer_start != -1:
                    answer_text = generated_text[answer_start + len("Answer:"):].strip().lower()
                    ans_file.write(json.dumps({"question_id": idx,
                                        # "prompt": cur_prompt,
                                        "text": answer_text,
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
    parser.add_argument("--conv_mode", type=str, default=None)
    parser.add_argument("--sample_neighbor_size", type=int, default=2)
    parser.add_argument("--output_path", type=str, default="temprompt")
    parser.add_argument("--llm_name", type=str, default='vicuna')
    parser.add_argument('--variant', type=str, default='nondst2', choices=['nondst2','dst2-v1','dst2-v2'])
    parser.add_argument("--test_mode", type=str, default='transductive', choices=['transductive', 'inductive'])

    args = parser.parse_args()

    eval_model(args)

    # ray.init()
    # run_eval(args, args.num_gpus)


# protobuf             4.22.3