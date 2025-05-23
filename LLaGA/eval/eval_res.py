import random

import torch
import json
import argparse
# from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, label_binarize


def eval_lp(args):
    all_sample=0
    correct = 0
    filename = f'{args.res_path}/{args.dataset}.csv'
    fw = open(filename, 'a+')
    mark_to_num = {'yes': 1.0, 'no': 0.0}
    # for testing in ['transductive', 'inductive']:
    for testing in [args.test_mode]:
        results = []
        print(f'{testing} testing ...')
        for run in range(args.num_runs):
            res_path = f'{args.res_path}/{args.dataset}-answers-{testing}{run}-{args.task}-{args.template}-{args.llm_name}.json'
            # answers_file = f'{args.output_path}/{args.dataset}-answers-{testing}{run}-{args.task}-{args.template}-{args.pretrained_embedding_type}.json'
            if args.empty:
                res_path = res_path.replace(args.llm_name, 'empty')
            y_true = []
            y_pred = []
            with open(res_path, 'r') as f:
                for line in f:
                    res = json.loads(line.strip())
                    ans = 'yes' if 'yes' in res["text"] else 'no'
                    # ans = ''.join(c for c in ans if c.isprintable())
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
        
        # fw.write(f'{args.dataset},{testing},{results.mean():.2f} ± {results.std():.2f}\n')
        # print(f'{args.dataset}\t{testing}\t{results.mean():.2f} ± {results.std():.2f}')
        if args.num_runs > 1:
            fw.write(f'{args.dataset},{args.template},{testing},{"empty" if args.empty else args.llm_name},{results[:,0].mean():.2f} ± {results[:,0].std():.2f},{results[:,1].mean():.2f} ± {results[:,1].std():.2f},{results[:,2].mean():.2f} ± {results[:,2].std():.2f}\n')
            print(f'{args.dataset}\t{args.template}\t{testing}\t{"empty" if args.empty else args.llm_name}\t{results[:,0].mean():.2f} ± {results[:,0].std():.2f}\t{results[:,1].mean():.2f} ± {results[:,1].std():.2f}\t{results[:,2].mean():.2f} ± {results[:,2].std():.2f}\n')
        else:
            fw.write(f'{args.dataset},{args.template},{testing},{"empty" if args.empty else args.llm_name},{results[:,0].mean():.2f},{results[:,1].mean():.2f}\n')
            print(f'{args.dataset}\t{args.template}\t{testing}\t{"empty" if args.empty else args.llm_name}\t{results[:,0].mean():.2f} \t{results[:,1].mean():.2f}\n')

        # acc = correct / all_sample
        # print(f"Test samples: {all_sample}\ncorrect: {correct}\n acc: {acc:.4f}")

    fw.close()

# 1. Multi-class 评估
def evaluate_multi_class2(y_true, y_pred):
    # 将明文标签编码为数值
    le = LabelEncoder()
    y_true_encoded = le.fit_transform(y_true)
    y_pred_encoded = le.transform(y_pred)
    
    # 计算 weighted accuracy
    weighted_acc = accuracy_score(y_true_encoded, y_pred_encoded)  # 对于多分类，accuracy 本身可以看作 weighted
    # 计算 weighted F1
    weighted_f1 = f1_score(y_true_encoded, y_pred_encoded, average='weighted')
    
    return weighted_acc, weighted_f1

# 2. Multi-label 评估
def evaluate_multi_label2(y_true, y_pred):
    # 将多标签数据二值化
    mlb = MultiLabelBinarizer()
    y_true_binarized = mlb.fit_transform(y_true)
    y_pred_binarized = mlb.transform(y_pred)
    
    # 计算每个样本的正确预测比例（模拟 weighted accuracy）
    num_samples = len(y_true)
    sample_acc = []
    for i in range(num_samples):
        true_set = set(y_true[i])
        pred_set = set(y_pred[i])
        if len(true_set) > 0:  # 避免除以零
            acc = len(true_set & pred_set) / len(true_set)  # 交集 / 真实标签数
            sample_acc.append(acc)
    weighted_acc = np.mean(sample_acc) if sample_acc else 0.0
    
    # 计算 weighted F1
    weighted_f1 = f1_score(y_true_binarized, y_pred_binarized, average='weighted')
    
    return weighted_acc, weighted_f1


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
        res_path = f'{args.res_path}/{args.dataset}-answers-test{run}-{args.task}-{args.template}-{args.llm_name}.json'
        # answers_file = f'{args.output_path}/{args.dataset}-answers-{testing}{run}-{args.task}-{args.template}-{args.pretrained_embedding_type}.json'
        if args.empty:
            res_path = res_path.replace(args.llm_name, 'empty')
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

                y_true.append(label)
                y_pred.append(ans)

        if args.dataset in ['FOOD', 'IMDB']: #multi-label
            result = evaluate_multi_label(y_true, y_pred, unique_labels)
        else:
            result = evaluate_multi_class(y_true, y_pred, unique_labels)
        
        results.append([result['acc'], result['macro_acc'], result['macro_f1'], result['macro_pr_auc']])
    results = np.array(results)*100
    
    if args.num_runs > 1:
        fw.write(f'{args.dataset},{args.template},nc,{"empty" if args.empty else args.llm_name},{results[:,0].mean():.2f} ± {results[:,0].std():.2f},{results[:,1].mean():.2f} ± {results[:,1].std():.2f},{results[:,2].mean():.2f} ± {results[:,2].std():.2f},{results[:,3].mean():.2f} ± {results[:,3].std():.2f}\n')
        print(f'{args.dataset}\t{args.template}\tnc\t{"empty" if args.empty else args.llm_name}\t{results[:,0].mean():.2f} ± {results[:,0].std():.2f}\t{results[:,1].mean():.2f} ± {results[:,1].std():.2f}\t{results[:,2].mean():.2f} ± {results[:,2].std():.2f}\t{results[:,3].mean():.2f} ± {results[:,3].std():.2f}\n')
    else:
        fw.write(f'{args.dataset},{args.template},nc,{"empty" if args.empty else args.llm_name},{results[:,0].mean():.2f},{results[:,1].mean():.2f},{results[:,2].mean():.2f},{results[:,3].mean():.2f}\n')
        print(f'{args.dataset}\t{args.template}\tnc\t{"empty" if args.empty else args.llm_name}\t{results[:,0].mean():.2f}\t{results[:,1].mean():.2f}\t{results[:,2].mean():.2f}\t{results[:,3].mean():.2f}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--res_path", type=str, default="./results/llaga-opt-2.7b-v1-simteg_all_origin_tape_multihop-laplacian_-1-2-10-linear-only-train-pretrain_acc1_nc_test_nc.jsonl")
    parser.add_argument("--task", type=str, default="lp")
    parser.add_argument("--test_mode", type=str, default='transductive', choices=['transductive', 'inductive'])
    parser.add_argument("--template", type=str, default="ND")
    parser.add_argument('--empty', action='store_true', default=False)
    parser.add_argument("--dataset", type=str, default="arxiv")
    parser.add_argument("--llm_name", type=str, default="sbert")
    parser.add_argument("--sample", type=int, default=-1)
    parser.add_argument('--num_runs', type=int, default=5, help='number of runs')
    args = parser.parse_args()

    func_dict = {
        "nc": eval_nc,
        "lp": eval_lp,
    }

    func=func_dict[args.task]
    func(args)
