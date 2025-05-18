import sys
sys.path.append("./")
sys.path.append("./utils")
import argparse
import torch
import numpy as np
import os
import json
from tqdm import tqdm
import shortuuid

from utils.constants import GRAPH_TOKEN_INDEX, DEFAULT_GRAPH_TOKEN, DEFAULT_GRAPH_PAD_ID, DEFAULT_GRAPH_START_TOKEN, DEFAULT_GRAPH_END_TOKEN
from utils.conversation import conv_templates, SeparatorStyle
from model.builder import load_pretrained_model
from utils.utils import disable_torch_init, tokenizer_graph_token, get_model_name_from_path, set_random_seed
from torch_geometric.utils import k_hop_subgraph, degree, remove_self_loops, add_self_loops
from torch_geometric.nn import MessagePassing
import math

SMALL_DATASETS=["pubmed", "cora"]


class MP(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


# def get_chunk(lst, n, k):
#     chunks = split_list(lst, n)
#     return chunks[k]

def load_pretrain_embedding_graph(data_dir, args):
    if args.pretrained_embedding_type == "simteg":
        simteg_sbert = torch.load(os.path.join(data_dir, "simteg_sbert_x.pt"))
        simteg_roberta = torch.load(os.path.join(data_dir, "simteg_roberta_x.pt"))
        simteg_e5 = torch.load(os.path.join(data_dir, "simteg_e5_x.pt"))
        pretrained_emb = torch.concat([simteg_sbert, simteg_roberta, simteg_e5], dim=-1)
    else:
        pretrained_emb = torch.as_tensor(np.load(os.path.join(data_dir, f"{args.pretrained_embedding_type}_{args.dataset}_node.npy")), dtype=torch.float)
    if args.empty: # 
        pretrained_emb = torch.zeros_like(pretrained_emb)
    
    return pretrained_emb

def load_pretrain_embedding_hop(dataset, pretrained_embedding_type, hop):
    pretrained_embs = [torch.as_tensor(np.load(f"../DyGLLM/processed_data/{dataset}/{pretrained_embedding_type}_{dataset}_node.npy"),dtype=torch.float)] + [torch.load(f"dataset/{dataset}/{pretrained_embedding_type}_{i}hop_x.pt").to(torch.float) for i in range(1, hop+1)]
    return pretrained_embs

def load_pretrain_embedding_hop_lp(data_dir, pretrained_embedding_type, hop):
    mask = torch.load(os.path.join(data_dir, f"no_test_link_mask.pt"))
    if pretrained_embedding_type == "simteg":
        simteg_sbert=[torch.load(os.path.join(data_dir, f"simteg_sbert_x.pt"))[mask]] + [torch.load(os.path.join(data_dir, f"simteg_sbert_{i}hop_x_notestlink.pt")) for i in range(1, hop + 1)]
        simteg_roberta = [torch.load(os.path.join(data_dir, f"simteg_roberta_x.pt"))[mask]] + [torch.load(os.path.join(data_dir, f"simteg_roberta_{i}hop_x_notestlink.pt")) for i in range(1, hop + 1)]
        simteg_e5 = [torch.load(os.path.join(data_dir, f"simteg_e5_x.pt"))[mask]] + [torch.load(os.path.join(data_dir, f"simteg_e5_{i}hop_x_notestlink.pt")) for i in range(1, hop + 1)]
        pretrained_embs = [torch.cat([simteg_sbert[i], simteg_roberta[i], simteg_e5[i]], dim=-1) for i in range(hop + 1)]
    else:
        pretrained_embs = [torch.load(os.path.join(data_dir, f"{pretrained_embedding_type}_x.pt"))[mask]]+  [torch.load(os.path.join(data_dir, f"{pretrained_embedding_type}_{i}hop_x_notestlink.pt")) for i in range(1, hop+1)]

    return pretrained_embs, mask

def eval_model(args):
    run = args.run
    set_random_seed(seed=run)
    args.seed = run
    # Model
    disable_torch_init()
    model_path = f"{args.model_path}{run}"
    if args.empty:
        model_path = model_path.replace(args.pretrained_embedding_type, f'empty-{args.empty_ndim}')
    model_name = get_model_name_from_path(model_path)
    print(f"Loaded from {model_path}. Model Base: {args.model_base}")
    tokenizer, model, context_len = load_pretrained_model(model_path, args.model_base, model_name,
                                                        cache_dir=args.cache_dir)
    model = model.to(torch.float16).cuda()
    data_dir = f"dataset/{args.dataset}"

    
    if args.template == "ND":
        pretrained_emb = load_pretrain_embedding_graph(data_dir, args)
        if args.empty: # 
            pretrained_emb = torch.zeros_like(pretrained_emb)
        structure_emb = torch.load(
            f"dataset/laplacian_{args.use_hop}_{args.sample_neighbor_size}.pt")

    elif args.template == "HO":
        pretrained_emb = load_pretrain_embedding_hop(args.dataset, args.pretrained_embedding_type, args.use_hop)
        if args.empty: # 
            pretrained_emb = [torch.zeros_like(x) for x in pretrained_emb]
        structure_emb = None
    else:
        raise ValueError

    print('generating testing answer files ...')
    # ND和HO两种模板使用相同的数据集
    prompt_file = os.path.join(data_dir, f"node_sampled_2_{args.sample_neighbor_size}_only_test{run}.jsonl")
            
    print(f"Load from {prompt_file}\n")
    lines = open(prompt_file, "r").readlines()

    if args.start >= 0:
        if args.end < 0:
            args.end = len(lines)
        lines = lines[args.start:args.end]
    elif args.end > 0:
        lines = lines[:args.end]

    answers_file = f'{args.output_path}/{args.dataset}-answers-test{run}-{args.task}-{args.template}-{args.pretrained_embedding_type}.json'
    if args.empty:
        answers_file = answers_file.replace(args.pretrained_embedding_type, 'empty')
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    questions = [json.loads(q) for q in lines]

    for line in tqdm(questions):
        idx = line["id"]
        if args.task in ["nd", "nda"]:
            qs=f"Please briefly describe the center node of {DEFAULT_GRAPH_TOKEN}."
        elif args.task == "nc":
            qs = line["conversations"][0]['value']
        elif args.task == "nctext":
            text = data.raw_texts[line['id']]
            text = text[:2000]
            if args.dataset == "arxiv":
                qs = f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, where nodes represent papers and edges represent co-citations, the node feature of center node is {text}. We need to classify the center node into 40 classes: cs.NA(Numerical Analysis), cs.MM(Multimedia), cs.LO(Logic in Computer Science), cs.CY(Computers and Society), cs.CR(Cryptography and Security), cs.DC(Distributed, Parallel, and Cluster Computing), cs.HC(Human-Computer Interaction), cs.CE(Computational Engineering, Finance, and Science), cs.NI(Networking and Internet Architecture), cs.CC(Computational Complexity), cs.AI(Artificial Intelligence), cs.MA(Multiagent Systems), cs.GL(General Literature), cs.NE(Neural and Evolutionary Computing), cs.SC(Symbolic Computation), cs.AR(Hardware Architecture), cs.CV(Computer Vision and Pattern Recognition), cs.GR(Graphics), cs.ET(Emerging Technologies), cs.SY(Systems and Control), cs.CG(Computational Geometry), cs.OH(Other Computer Science), cs.PL(Programming Languages), cs.SE(Software Engineering), cs.LG(Machine Learning), cs.SD(Sound), cs.SI(Social and Information Networks), cs.RO(Robotics), cs.IT(Information Theory), cs.PF(Performance), cs.CL(Computational Complexity), cs.IR(Information Retrieval), cs.MS(Mathematical Software), cs.FL(Formal Languages and Automata Theory), cs.DS(Data Structures and Algorithms), cs.OS(Operating Systems), cs.GT(Computer Science and Game Theory), cs.DB(Databases), cs.DL(Digital Libraries), cs.DM(Discrete Mathematics), please tell me which class the center node belongs to? Direct tell me the class name."
            elif args.dataset == "products":
                qs = f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, where nodes represent products sold in Amazon, and edges between products indicate they are purchased together, the node feature of center node is {text}. We need to classify the center node into 47 classes: Home & Kitchen, Health & Personal Care, Beauty, Sports & Outdoors, Books, Patio, Lawn & Garden, Toys & Games, CDs & Vinyl, Cell Phones & Accessories, Grocery & Gourmet Food, Arts, Crafts & Sewing, Clothing, Shoes & Jewelry, Electronics, Movies & TV, Software, Video Games, Automotive, Pet Supplies, Office Products, Industrial & Scientific, Musical Instruments, Tools & Home Improvement, Magazine Subscriptions, Baby Products, label 25, Appliances, Kitchen & Dining, Collectibles & Fine Art, All Beauty, Luxury Beauty, Amazon Fashion, Computers, All Electronics, Purchase Circles, MP3 Players & Accessories, Gift Cards, Office & School Supplies, Home Improvement, Camera & Photo, GPS & Navigation, Digital Music, Car Electronics, Baby, Kindle Store, Buy a Kindle, Furniture & D&#233;cor, #508510, please tell me which class the center node belongs to? Direct tell me the class name."
            elif args.dataset == "pubmed":
                qs = f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, where nodes represent papers about Diabetes and edges represent co-citations, the node feature of center node is {text}. We need to classify the center node into 3 classes: Diabetes Mellitus Experimental, Diabetes Mellitus Type1, Diabetes Mellitus Type2, please tell me which class the center node belongs to? Direct tell me the class name."
            elif args.dataset == "cora":
                qs = f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, where nodes represent papers and edges represent co-citations, the node feature of center node is {text}. We need to classify the center node into 7 classes: Case_Based, Genetic_Algorithms, Neural_Networks, Probabilistic_Methods, Reinforcement_Learning, Rule_Learning, Theory, please tell me which class the center node belongs to? Direct tell me the class name."
            else:
                raise ValueError
        elif args.task == "lp":
            # qs=f"Given two node-centered subgraphs: {DEFAULT_GRAPH_TOKEN} and {DEFAULT_GRAPH_TOKEN}, we need to predict whether these two nodes connect with each other. Please tell me whether two center nodes in the subgraphs should connect to each other."
            qs=f"Given two node-centered subgraphs: {DEFAULT_GRAPH_TOKEN} and {DEFAULT_GRAPH_TOKEN}, predict whether the two center nodes connect with each other. Answer only 'Yes' or 'No'."
        else:
            print(f"NOT SUPPORT {args.task}!!!")
            raise ValueError
        cur_prompt = qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_graph_token(prompt, tokenizer, GRAPH_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        if not isinstance(line['graph'][0], list):
            line['graph'] = [line['graph']]
        if args.template == "ND":
            graph = torch.LongTensor(line['graph'])
            mask = graph != DEFAULT_GRAPH_PAD_ID
            masked_graph_emb = pretrained_emb[graph[mask]]
            s, n, d = graph.shape[0], graph.shape[1], masked_graph_emb.shape[1]
            graph_emb = torch.zeros((s, n, d))
            graph_emb[mask] = masked_graph_emb
            if structure_emb is not None:
                graph_emb = torch.cat([graph_emb, structure_emb.unsqueeze(0).expand(s, -1, -1)], dim=-1)
        elif args.template == "HO":
            for g in range(len(line['graph'])):
                center_id = line['graph'][g][0]
                line['graph'][g] = [center_id]*(args.use_hop+1)
            graph = torch.LongTensor(line['graph'])
            center_id = graph[:, 0]
            graph_emb = torch.stack([emb[center_id] for emb in pretrained_emb], dim=1)
        else:
            raise ValueError


        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        try:
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    graph_emb=graph_emb.half().cuda(),
                    graph=graph.cuda(),
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
                    use_cache=True)

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            # outputs = outputs.strip()
        except Exception as e:
            print(f"!!!!!!Error!!!!! {e}")
            outputs=""

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                "prompt": cur_prompt,
                                "graph": line['graph'],
                                "text": outputs,
                                "gt":line["conversations"][1]['value'],
                                "answer_id": ans_id}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--llm_name", type=str, default='vicuna')
    parser.add_argument('--empty', action='store_true', default=False)
    parser.add_argument('--empty_ndim', type=int, default=384)
    parser.add_argument("--pretrained_embedding_type", type=str, default="sbert")
    parser.add_argument("--use_hop", type=int, default=2)
    parser.add_argument("--sample_neighbor_size", type=int, default=5)
    parser.add_argument("--output_path", type=str, default="results")
    parser.add_argument("--conv_mode", type=str, default="v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--start", type=int, default=-1)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--mm_use_graph_start_end",default=False, action="store_true")
    parser.add_argument("--task", type=str, default="nc")
    parser.add_argument("--dataset", type=str, default="FOOD")
    parser.add_argument("--cache_dir", type=str, default="../../checkpoint")
    parser.add_argument("--template", type=str, default="ND")
    parser.add_argument('--run', type=int, default=0, help='number of run')
    args = parser.parse_args()

    eval_model(args)
