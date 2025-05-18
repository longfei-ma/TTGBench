import sys

import argparse
import os
import json
import copy

import os
from openai import OpenAI
import numpy as np

from tqdm import tqdm
import json
import os.path as osp

def get_response(prompt, args):
    if args.llm_name == 'deepseek-r1':
        client = OpenAI(api_key="sk-d86466663e4744979068c56c84f7944e", base_url="https://api.deepseek.com")
        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=messages
        )
        reasoning_content = response.choices[0].message.reasoning_content
        content = response.choices[0].message.content
        return {'text': content.rsplit(':', 1)[1].strip(), 'reasoning_content': reasoning_content}
    elif args.llm_name == 'deepseek-v3':
        client = OpenAI(api_key="sk-d86466663e4744979068c56c84f7944e", base_url="https://api.deepseek.com")
        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages
        )
        content = response.choices[0].message.content
        return {'text': content.rsplit(':',1)[1].strip()}

def eval_model(args):
    # load prompting file
    # prompt_file = load_prompting_file(args.prompting_file)
    run = args.run
    data_dir = f"data/temprompt/{args.dataset}"
    if args.empty:
        data_dir += '-empty'
        
    if args.task == 'nc': #
        print(f'generating part-{args.current_part} of {args.num_parts} testing answer files ...')
        # for prompt_file in [prompt_file_transductive, prompt_file_inductive]:  
        prompt_file = os.path.join(data_dir, f"{args.variant}_{args.task}_2_{args.sample_neighbor_size}_sampled_test{run}.jsonl")
        answers_file = f'{args.output_path}/{args.dataset}-{args.llm_name}-test{run}-{args.variant}-{args.task}-{args.current_part}.json'
        if args.empty:
            answers_file = answers_file.replace(args.dataset, f'{args.dataset}-empty')
        os.makedirs(os.path.dirname(answers_file), exist_ok=True)
        ans_file = open(answers_file, "w")
        questions = open(prompt_file, "r").readlines()
        # questions = json.loads(lines[0])
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
        for line in tqdm(questions):
            instruct_item = json.loads(line.strip())
            idx = instruct_item["id"]
            prompt = instruct_item["conversations"][0]["value"]
            result = {"question_id": idx, "gt":instruct_item["conversations"][1]['value']}
            result.update(get_response(prompt, args))
            ans_file.write(json.dumps(result) + "\n")
            ans_file.flush()
        ans_file.close()
    elif args.task == 'lp':
        # 分别评估transductive和inductive
        # for testing in ['transductive', 'inductive']:
        for testing in [args.test_mode]:
            print(f'generating {testing} answer files ...')
            # for prompt_file in [prompt_file_transductive, prompt_file_inductive]:  
            prompt_file = os.path.join(data_dir, f"{args.variant}_{args.task}_2_{args.sample_neighbor_size}_sampled_test{run}_{testing}.jsonl")
            answers_file = f'{args.output_path}/{args.dataset}-{args.llm_name}-{testing}{run}-{args.variant}-{args.task}-{args.current_part}.json'
            if args.empty:
                answers_file = answers_file.replace(args.dataset, f'{args.dataset}-empty')
            if os.path.exists(answers_file):
                
                pass
            else:
                os.makedirs(os.path.dirname(answers_file), exist_ok=True)
                ans_file = open(answers_file, "w")
            questions = open(prompt_file, "r").readlines()
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
            for line in tqdm(questions):
                instruct_item = json.loads(line.strip())
                idx = instruct_item["id"]
                prompt = instruct_item["conversations"][0]["value"]
                result = {"question_id": idx, "gt":instruct_item["conversations"][1]['value']}
                result.update(get_response(prompt, args))
                ans_file.write(json.dumps(result) + "\n")
                ans_file.flush()
            ans_file.close()
    else:
        raise ValueError
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="FOOD")
    parser.add_argument('--run', type=int, default=0, help='number of run')
    parser.add_argument('--num_parts', type=int, default=20, help='number of data parts')
    parser.add_argument('--current_part', type=int, default=19, help='current part number')
    parser.add_argument("--task", type=str, default="nc")
    parser.add_argument("--pretrained_embedding_type", type=str, default="sbert")
    parser.add_argument("--model_max_length", type=int, default=4096)
    parser.add_argument("--model_path", type=str, default="facebook/opt-350m")
    parser.add_argument("--cache_dir", type=str, default="facebook/opt-350m")
    parser.add_argument("--pretrain_graph_model_path", type=str, default=None)
    parser.add_argument("--graph_tower", type=str, default="clip_gcn")
    parser.add_argument('--empty', action='store_true', default=False)
    parser.add_argument('--empty_ndim', type=int, default=384)
    parser.add_argument("--sample_neighbor_size", type=int, default=2)
    parser.add_argument("--output_path", type=str, default="temprompt")
    parser.add_argument("--llm_name", type=str, default='deepseek-r1')
    parser.add_argument('--variant', type=str, default='nondst2', choices=['nondst2','dst2-v1','dst2-v2'])
    parser.add_argument("--test_mode", type=str, default='transductive', choices=['transductive', 'inductive'])

    args = parser.parse_args()

    eval_model(args)

    # ray.init()
    # run_eval(args, args.num_gpus)


# protobuf             4.22.3