import random

import torch
import json
import argparse
# from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, f1_score, confusion_matrix
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

def eval_arxiv_nd(res_path):
    data=torch.load("dataset/ogbn-arxiv/processed_data.pt")
    labels=data.label_texts
    short_labels = [l[0:5] for l in labels]
    ys=data.y.numpy().tolist()

    titles = data.title

    all_sample=0
    short_correct=0
    all_correct=0
    gt=[]
    out=[]
    with open(res_path, 'r') as f:
        for line in f:
            all_sample+=1
            res = json.loads(line)
            ans = res["text"]
            id=res["question_id"]
            y=ys[id]
            short_label = short_labels[y]
            label=labels[y]
            if label.strip() in ans.strip():
                all_correct+=1
            if short_label in ans:
                short_correct+=1
            out.append(ans)
            gt.append(f"This is a paper in {label} domain, it's about {titles[id]}.")
    short_acc = short_correct/all_sample
    all_acc = all_correct / all_sample
    print(f"Test samples: {all_sample}\nshort_correct: {short_correct}\nshort_acc: {short_acc:.4f}\nall_correct: {all_correct}\nall_acc: {all_acc:.4f}")
    gt_embedding = get_sbert_embedding("sbert", gt, 0)
    out_embedding = get_sbert_embedding("sbert", out, 0)
    gt_embedding=F.normalize(gt_embedding, p=2, eps=1e-6, dim=1)
    out_embedding=F.normalize(out_embedding, p=2, eps=1e-6, dim=1)
    predict_sim=(gt_embedding*out_embedding).sum(1).mean().item()
    gt_sim_matrix=torch.mm(gt_embedding, gt_embedding.transpose(0, 1)).detach().cpu()
    n=gt_sim_matrix.shape[0]
    gt_sim_matrix[torch.eye(n, dtype=torch.bool)]=0
    gt_sim=(gt_sim_matrix.sum()/(n*(n-1))).item()
    print(f"Predict similarity {predict_sim: .4f}, Pairwise similarity: {gt_sim: .4f}")


def eval_nc(res_path):
    data=torch.load("dataset/ogbn-arxiv/processed_data.pt")
    labels=data.label_texts
    short_labels = [l[0:5] for l in labels]
    ys=data.y.numpy().tolist()

    all_sample=0
    overall_correct=0
    strict_correct=0
    error=[]
    with open(res_path, 'r') as f:
        for line in f:
            all_sample+=1
            res = json.loads(line)
            ans = res["text"]
            y=ys[res["question_id"]]
            short_label = short_labels[y]
            label=labels[y]
            if label.lower().strip() == ans.lower().strip():
                strict_correct+=1
                overall_correct+=1
            elif short_label.lower() in ans.lower() and sum([la.lower() in ans.lower() for la in short_labels])==1:
                overall_correct+=1
            else:
                error.append((ans, label))
            if args.sample > 0 and all_sample >= args.sample:
                break
    overall_acc = overall_correct/all_sample
    strict_acc = strict_correct / all_sample
    print(f"Test samples: {all_sample}\nstrict_acc: {strict_acc:.4f}\noverall_acc: {overall_acc:.4f}")


def eval_lp(args):
    filename = f'{args.res_path}/{args.dataset}.csv'
    fw = open(filename, 'a+')
    mark_to_num = {'yes': 1.0, 'no': 0.0}
    # for testing in ['transductive', 'inductive']:
    for testing in [args.test_mode]:
        results = []
        print(f'{testing} testing ...')
        for run in range(args.num_runs):
            res_path = f'{args.res_path}/{args.dataset}-answers-{testing}{run}-{args.variant}-{args.task}.json'
            # answers_file = f'{args.output_path}/{args.dataset}-answers-{testing}{run}-{args.task}-{args.template}-{args.pretrained_embedding_type}.json'
            if args.empty:
                res_path = res_path.replace(args.dataset, f'{args.dataset}-empty')
            y_true = []
            y_pred = []
            with open(res_path, 'r') as f:
                for line in f:
                    res = json.loads(line.strip())
                    # ans = res["text"].strip()
                    ans = 'yes' if 'yes' in res["text"] else 'no'
                    label=res["gt"].strip()
                    # y_true.append(label)
                    # y_pred.append(ans)
                    y_true.append(mark_to_num[label])
                    y_pred.append(mark_to_num[ans])
                    # all_sample += 1
                    # if ("yes" in ans and "yes" in label) or ("yes" not in ans and "no" in label):
                    #     correct += 1
                    # if args.sample > 0 and all_sample >=  args.sample:
                    #     break
                    
            # f1 = f1_score(y_true, y_pred, pos_label='yes')
            average_precision = average_precision_score(y_true=y_true, y_score=y_pred)
            roc_auc = roc_auc_score(y_true=y_true, y_score=y_pred)
            # f1 = f1_score(y_true, y_pred)
            # results.append(f1)
            # results.append([average_precision, roc_auc, f1])
            results.append([average_precision, roc_auc])
        results = np.array(results)*100
        
        if args.num_runs > 1:
            fw.write(f'{args.dataset},{args.variant},{testing},{"empty" if args.empty else "text"},{results[:,0].mean():.2f} ± {results[:,0].std():.2f},{results[:,1].mean():.2f} ± {results[:,1].std():.2f}\n')
            print(f'{args.dataset}\t{args.variant}\t{testing}\t{"empty" if args.empty else "text"}\t{results[:,0].mean():.2f} ± {results[:,0].std():.2f}\t{results[:,1].mean():.2f} ± {results[:,1].std():.2f}\n')
        else:
            fw.write(f'{args.dataset},{args.variant},{testing},{"empty" if args.empty else "text"},{results[:,0].mean():.2f},{results[:,1].mean():.2f}\n')
            print(f'{args.dataset}\t{args.variant}\t{testing}\t{"empty" if args.empty else "text"}\t{results[:,0].mean():.2f}\t{results[:,1].mean():.2f}\n')

        # acc = correct / all_sample
        # print(f"Test samples: {all_sample}\ncorrect: {correct}\n acc: {acc:.4f}")

    fw.close()

def evaluate_multi_class(y_true, y_pred, classes):
    """
    计算多分类场景下的 Accuracy, Macro Accuracy, Macro PR-AUC, Macro F1
    参数:
        y_true: 真实标签 (list 或 numpy array, 明文标签如 ['cat', 'dog', ...])
        y_pred: 预测标签 (list 或 numpy array, 明文标签如 ['cat', 'dog', ...])
        classes: 类别列表 (list, 如 ['cat', 'dog', 'tiger'])
    返回:
        dict: 包含各指标的结果
    """
    # 将明文标签转换为数值索引
    le = LabelEncoder()
    le.fit(classes)
    y_true_idx = le.transform(y_true)
    y_pred_idx = le.transform(y_pred)
    
    # 1. Accuracy
    acc = accuracy_score(y_true_idx, y_pred_idx)
    
    # 2. Macro Accuracy
    cm = confusion_matrix(y_true_idx, y_pred_idx)
    n_classes = len(classes)
    class_counts = cm.sum(axis=1)
    per_class_acc = cm.diagonal() / class_counts
    macro_acc = np.mean(per_class_acc)
    
    # 3. Macro PR-AUC（伪实现，因无概率）
    y_true_binary = label_binarize(y_true_idx, classes=range(n_classes))
    y_pred_binary = label_binarize(y_pred_idx, classes=range(n_classes))
    macro_pr_auc = average_precision_score(y_true_binary, y_pred_binary, average='macro')
    # 注意：此值为伪 PR-AUC，仅基于正确性，不推荐使用
    
    # 4. Macro F1
    macro_f1 = f1_score(y_true_idx, y_pred_idx, average='macro')
    
    return {
        'acc': acc,
        'macro_acc': macro_acc,
        'macro_pr_auc': macro_pr_auc,
        'macro_f1': macro_f1
    }


def evaluate_multi_label(y_true, y_pred, classes):
    """
    计算多标签场景下的 Accuracy, Macro Accuracy, Macro PR-AUC, Macro F1
    参数:
        y_true: 真实标签 (list of lists, 如 [['cat'], ['dog', 'tiger'], ...])
        y_pred: 预测标签 (list of lists, 如 [['cat'], ['dog'], ...])
        classes: 类别列表 (list, 如 ['cat', 'dog', 'tiger'])
    返回:
        dict: 包含各指标的结果
    """
    # 将明文标签列表转换为二值矩阵
    mlb = MultiLabelBinarizer(classes=classes)
    y_true_binary = mlb.fit_transform(y_true)
    y_pred_binary = mlb.transform(y_pred)
    
    # 1. Accuracy（Exact Match Ratio）
    acc = accuracy_score(y_true_binary, y_pred_binary)
    
    # 2. Macro Accuracy
    n_samples, n_labels = y_true_binary.shape
    per_label_acc = [accuracy_score(y_true_binary[:, i], y_pred_binary[:, i]) for i in range(n_labels)]
    macro_acc = np.mean(per_label_acc)
    
    # 3. Macro PR-AUC（伪实现，因无概率）
    macro_pr_auc = average_precision_score(y_true_binary, y_pred_binary, average='macro')
    # 注意：此值为伪 PR-AUC，仅基于正确性，不推荐使用
    
    # 4. Macro F1
    macro_f1 = f1_score(y_true_binary, y_pred_binary, average='macro')
    
    return {
        'acc': acc,
        'macro_acc': macro_acc,
        'macro_pr_auc': macro_pr_auc,
        'macro_f1': macro_f1
    }



def eval_nc(args):
    with open(f'../DyGLLM/DG_data/{args.dataset}/{args.dataset}_unique_labels.json', 'r', encoding="utf-8") as f: #去重标签
        unique_labels = [x.lower() for x in json.load(f)]

    filename = f'{args.res_path}/{args.dataset}.csv'
    results = []
    fw = open(filename, 'a+')
    for run in range(args.num_runs):
        res_path = f'{args.res_path}/{args.dataset}-answers-test{run}-{args.variant}-{args.task}.json'
        if args.empty:
            res_path = res_path.replace(args.dataset, f'{args.dataset}-empty')
        y_true = []
        y_pred = []
        with open(res_path, 'r') as f:
            for line in f:
                res = json.loads(line.strip())
                if not res["gt"]:continue
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
        else:
            result = evaluate_multi_class(y_true, y_pred, unique_labels)
        
        results.append([result['acc'], result['macro_acc'], result['macro_f1'], result['macro_pr_auc']])
    results = np.array(results)*100
    
    if args.num_runs > 1:
        fw.write(f'{args.dataset},nc,{args.variant},{"empty" if args.empty else "text"},{results[:,0].mean():.2f} ± {results[:,0].std():.2f},{results[:,1].mean():.2f} ± {results[:,1].std():.2f},{results[:,2].mean():.2f} ± {results[:,2].std():.2f},{results[:,3].mean():.2f} ± {results[:,3].std():.2f}\n')
        print(f'{args.dataset}\tnc\t{args.variant}\t{"empty" if args.empty else "text"}\t{results[:,0].mean():.2f} ± {results[:,0].std():.2f}\t{results[:,1].mean():.2f} ± {results[:,1].std():.2f}\t{results[:,2].mean():.2f} ± {results[:,2].std():.2f}\t{results[:,3].mean():.2f} ± {results[:,3].std():.2f}\n')
    else:
        fw.write(f'{args.dataset},nc,{args.variant},{"empty" if args.empty else "text"},{results[:,0].mean():.2f},{results[:,1].mean():.2f},{results[:,2].mean():.2f},{results[:,3].mean():.2f}\n')
        print(f'{args.dataset}\tnc\t{args.variant}\t{"empty" if args.empty else "text"}\t{results[:,0].mean():.2f}\t{results[:,1].mean():.2f}\t{results[:,2].mean():.2f}\t{results[:,3].mean():.2f}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--res_path", type=str, default="")
    parser.add_argument("--task", type=str, default="lp")
    parser.add_argument("--test_mode", type=str, default='transductive', choices=['transductive', 'inductive'])
    parser.add_argument('--empty', action='store_true', default=False)
    parser.add_argument("--dataset", type=str, default="FOOD")
    parser.add_argument("--sample", type=int, default=-1)
    parser.add_argument('--num_runs', type=int, default=5, help='number of runs')
    parser.add_argument('--variant', type=str, default='nondst2', choices=['nondst2','dst2-v1','dst2-v2'])
    args = parser.parse_args()

    func_dict = {
        "nc": eval_nc,
        "lp": eval_lp,
    }

    func=func_dict[args.task]
    func(args)
