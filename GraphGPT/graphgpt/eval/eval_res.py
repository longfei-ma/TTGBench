import random

import torch
import json
import argparse
# from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, f1_score, jaccard_score
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, label_binarize


def sbert(model_type, device):
    model = SentenceTransformer(model_type, device=device)
    return model

def get_sbert_embedding(model_type, texts, device):
    if model_type == 'sbert':
        model_type = 'all-MiniLM-L6-v2'
    sbert_model = sbert(model_type, f'cuda:{device}')
    sbert_embeds = sbert_model.encode(texts, batch_size=8, show_progress_bar=True)
    return torch.tensor(sbert_embeds)


def eval_lp(args):
    filename = f'{args.res_path}/{args.dataset}.csv'
    fw = open(filename, 'a+')
    mark_to_num = {'yes': 1.0, 'no': 0.0}
    # for testing in ['transductive', 'inductive']:
    for testing in [args.test_mode]:
        results = []
        print(f'{testing} testing ...')
        for run in range(args.num_runs):
            res_path = f'{args.res_path}/{args.dataset}-answers-{testing}{run}-{args.task}-{args.llm_name}.json'
            y_true = []
            y_pred = []
            with open(res_path, 'r') as f:
                for line in f:
                    res = json.loads(line.strip())
                    ans = 'yes' if 'yes' in res["text"] else 'no'
                    label=res["gt"].strip()
                    y_true.append(mark_to_num[label])
                    y_pred.append(mark_to_num[ans])
                    
            average_precision = average_precision_score(y_true=y_true, y_score=y_pred)
            roc_auc = roc_auc_score(y_true=y_true, y_score=y_pred)
            results.append([average_precision, roc_auc])
        results = np.array(results)*100
        
        fw.write(f'{args.dataset},{testing},args.llm_name,{results[:,0].mean():.2f} ± {results[:,0].std():.2f},{results[:,1].mean():.2f} ± {results[:,1].std():.2f}\n')
    fw.close()

def evaluate_multi_class(y_true, y_pred, classes):
    le = LabelEncoder()
    le.fit(classes)
    y_true_idx = le.transform(y_true)
    y_pred_idx = le.transform(y_pred)
    acc = accuracy_score(y_true_idx, y_pred_idx)
    weighted_f1 = f1_score(y_true_idx, y_pred_idx, average='weighted')
    
    return {
        'acc': acc,
        'f1': weighted_f1
    }


def evaluate_multi_label(y_true, y_pred, classes):
    mlb = MultiLabelBinarizer(classes=classes)
    y_true_binary = mlb.fit_transform(y_true)
    y_pred_binary = mlb.transform(y_pred)

    jacc = jaccard_score(y_true_binary, y_pred_binary, average='samples')
    micro_f1 = f1_score(y_true_binary, y_pred_binary, average='micro')
    macro_f1 = f1_score(y_true_binary, y_pred_binary, average='macro')
    
    return {
        'jacc': jacc,
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
    }


def eval_nc(args):
    with open(f'../datasets/{args.dataset}/{args.dataset}_unique_labels.json', 'r', encoding="utf-8") as f: 
        unique_labels = [x.lower() for x in json.load(f)]

    filename = f'{args.res_path}/{args.dataset}.csv'
    results = []
    fw = open(filename, 'a+')
    for run in range(args.num_runs):
        res_path = f'{args.res_path}/{args.dataset}-answers-test{run}-{args.task}-{args.llm_name}.json'
        y_true = []
        y_pred = []
        with open(res_path, 'r') as f:
            for line in f:
                res = json.loads(line.strip())
                ans = res["text"].strip().lower()
                label=res["gt"].strip().lower()
                if args.dataset in ['FOOD', 'IMDB']: #multi-label
                    ans = [s.strip() for s in ans.split(',')]
                    ans = [s for s in ans if s in unique_labels]
                    label = [s.strip() for s in label.split(',')]

                else: # multi-class
                    if label not in unique_labels:
                        continue

                    for item in unique_labels:
                        if item in ans:
                            ans = item
                            break
                    else:
                        ans = None
                
                if ans:
                    y_true.append(label)
                    y_pred.append(ans)

        if args.dataset in ['FOOD', 'IMDB']: #multi-label
            result = evaluate_multi_label(y_true, y_pred, unique_labels)
            results.append([result['jacc'], result['micro_f1'], result['macro_f1']])
        else:
            result = evaluate_multi_class(y_true, y_pred, unique_labels)
            results.append([result['acc'], result['weighted_f1']])
        
    results = np.array(results)*100
    
    if args.dataset in ['FOOD', 'IMDB']: #multi-label
        fw.write(f'{args.dataset},args.llm_name,{results[:,0].mean():.2f} ± {results[:,0].std():.2f},{results[:,1].mean():.2f} ± {results[:,1].std():.2f},{results[:,2].mean():.2f} ± {results[:,2].std():.2f}\n')
    else:
        fw.write(f'{args.dataset},args.llm_name,{results[:,0].mean():.2f} ± {results[:,0].std():.2f},{results[:,1].mean():.2f} ± {results[:,1].std():.2f}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--res_path", type=str, default="")
    parser.add_argument("--task", type=str, default="lp")
    parser.add_argument("--test_mode", type=str, default='transductive', choices=['transductive', 'inductive'])
    parser.add_argument("--dataset", type=str, default="FOOD")
    parser.add_argument("--llm_name", type=str, default="sbert")
    parser.add_argument("--sample", type=int, default=-1)
    parser.add_argument('--num_runs', type=int, default=5, help='number of runs')
    args = parser.parse_args()

    func_dict = {
        "nc": eval_nc,
        "lp": eval_lp
    }

    func=func_dict[args.task]
    func(args)
