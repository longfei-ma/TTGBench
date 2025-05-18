import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score, average_precision_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import label_binarize


def get_link_prediction_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the link prediction task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()
    y_pred = (predicts >= 0.5).astype(int)

    average_precision = average_precision_score(y_true=labels, y_score=predicts)
    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)
    f1 = f1_score(y_true=labels, y_pred=y_pred)
    acc = accuracy_score(labels, y_pred)

    return {'average_precision': average_precision, 'roc_auc': roc_auc, 'f1_score': f1, 'acc': acc}


def weighted_accuracy_multi_class(y_true, y_pred_logits):
    """
    计算多分类场景下的 Weighted Accuracy（输入为 torch.tensor）
    参数:
        y_true: 真实标签 (torch.tensor, 1D)
        y_pred_logits: 预测 logits (torch.tensor, n_samples x n_classes)
    返回:
        weighted_acc: 加权准确率 (float)
    """
    # 确保输入是 torch.tensor
    y_true = torch.as_tensor(y_true)
    y_pred_logits = torch.as_tensor(y_pred_logits)
    
    # 将 logits 转换为类别预测
    y_pred = torch.argmax(y_pred_logits, dim=1)
    
    # 将张量转换为 NumPy 以使用 scikit-learn 的混淆矩阵
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true_np, y_pred_np)
    n_classes = cm.shape[0]
    
    # 每个类别的样本数
    class_counts = torch.tensor(cm.sum(axis=1), dtype=torch.float32)
    total_samples = class_counts.sum()
    
    # 每个类别的准确率
    per_class_acc = torch.tensor(cm.diagonal(), dtype=torch.float32) / class_counts
    # 类别权重
    weights = class_counts / total_samples
    
    # 加权平均
    weighted_acc = torch.sum(weights * per_class_acc).item()
    return weighted_acc

def weighted_f1_multi_class(y_true, y_pred_logits):
    """
    计算多分类场景下的 Weighted F1 Score（输入为 torch.tensor）
    参数:
        y_true: 真实标签 (torch.tensor, 1D)
        y_pred_logits: 预测 logits (torch.tensor, n_samples x n_classes)
    返回:
        weighted_f1: 加权 F1 分数 (float)
    """
    # 确保输入是 torch.tensor
    y_true = torch.as_tensor(y_true)
    y_pred_logits = torch.as_tensor(y_pred_logits)
    
    # 将 logits 转换为类别预测
    y_pred = torch.argmax(y_pred_logits, dim=1)
    
    # 转换为 NumPy 以使用 scikit-learn 的函数
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    
    # 计算每个类别的精确率、召回率和 F1 分数
    precision, recall, f1, support = precision_recall_fscore_support(y_true_np, y_pred_np, average=None)
    total_samples = float(sum(support))
    
    # 类别权重
    weights = torch.tensor(support, dtype=torch.float32) / total_samples
    
    # 加权平均 F1 分数
    weighted_f1 = torch.sum(weights * torch.tensor(f1, dtype=torch.float32)).item()
    return weighted_f1


def weighted_accuracy_multi_label(y_true, y_pred_logits, threshold=0.5):
    """
    计算多标签场景下的 Weighted Accuracy（输入为 torch.tensor）
    参数:
        y_true: 真实标签矩阵 (torch.tensor, n_samples x n_labels)
        y_pred_logits: 预测 logits 矩阵 (torch.tensor, n_samples x n_labels)
        threshold: 阈值，sigmoid 后大于此值为正标签 (默认 0.5)
    返回:
        weighted_acc: 加权准确率 (float)
    """
    # 确保输入是 torch.tensor
    y_true = torch.as_tensor(y_true)
    y_pred_logits = torch.as_tensor(y_pred_logits)
    
    # 将 logits 转换为概率（sigmoid），然后根据阈值转换为二值预测
    y_pred_probs = torch.sigmoid(y_pred_logits)
    y_pred = (y_pred_probs >= threshold).int()
    
    # 转换为 NumPy 以使用 scikit-learn 的函数
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    
    n_samples, n_labels = y_true.shape
    
    # 每个标签的样本数（正例数）
    label_counts = y_true.sum(dim=0).float()
    total_samples = n_samples  # 每个标签的样本总数为总样本数
    
    # 每个标签的准确率
    per_label_acc = [accuracy_score(y_true_np[:, i], y_pred_np[:, i]) for i in range(n_labels)]
    # 标签权重（按正例比例）
    weights = label_counts / label_counts.sum() if label_counts.sum() > 0 else torch.ones(n_labels) / n_labels
    
    # 加权平均
    weighted_acc = torch.sum(weights * torch.tensor(per_label_acc, dtype=torch.float32)).item()
    return weighted_acc

def weighted_f1_multi_label(y_true, y_pred_logits, threshold=0.5):
    """
    计算多标签场景下的 Weighted F1 Score（输入为 torch.tensor）
    参数:
        y_true: 真实标签矩阵 (torch.tensor, n_samples x n_labels)
        y_pred_logits: 预测 logits 矩阵 (torch.tensor, n_samples x n_labels)
        threshold: 阈值，sigmoid 后大于此值为正标签 (默认 0.5)
    返回:
        weighted_f1: 加权 F1 分数 (float)
    """
    # 确保输入是 torch.tensor
    y_true = torch.as_tensor(y_true)
    y_pred_logits = torch.as_tensor(y_pred_logits)
    
    # 将 logits 转换为概率（sigmoid），然后根据阈值转换为二值预测
    y_pred_probs = torch.sigmoid(y_pred_logits)
    y_pred = (y_pred_probs >= threshold).int()
    
    # 转换为 NumPy 以使用 scikit-learn 的函数
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    
    # 每个标签的精确率、召回率和 F1 分数
    precision, recall, f1, support = precision_recall_fscore_support(y_true_np, y_pred_np, average=None)
    
    # 标签权重（按正例比例）
    total_positives = float(sum(support))
    weights = torch.tensor(support, dtype=torch.float32) / total_positives if total_positives > 0 else torch.ones(len(support)) / len(support)
    
    # 加权平均 F1 分数
    weighted_f1 = torch.sum(weights * torch.tensor(f1, dtype=torch.float32)).item()
    return weighted_f1


def macro_pr_auc_multi_class(y_true, y_pred_logits):
    """
    计算多分类场景下的 Macro PR-AUC（使用 average_precision_score 的 average='macro'）
    参数:
        y_true: 真实标签 (torch.tensor, 1D, 形状为 [n_samples])
        y_pred_logits: 预测 logits (torch.tensor, 形状为 [n_samples, n_classes])
    返回:
        macro_pr_auc: 宏平均 PR-AUC (float)
    """
    # 转换为 NumPy
    y_true_np = y_true.cpu().numpy()
    y_pred_probs = torch.softmax(y_pred_logits, dim=1).cpu().numpy()  # 概率，形状 [n_samples, n_classes]

    # Macro Average Precision Score (PR-AUC)
    # 需要将 y_true 转为二值矩阵 (one-vs-rest)
    n_classes = y_pred_logits.shape[1]
    y_true_binary = label_binarize(y_true_np, classes=range(n_classes))  # 形状 [n_samples, n_classes]
    macro_pr_auc = average_precision_score(y_true_binary, y_pred_probs, average='macro')

    return macro_pr_auc

def macro_f1_multi_class(y_true, y_pred_logits, average='micro'):
    """
    计算多分类场景下的 F1 Score
    参数:
        y_true: 真实标签 (torch.tensor, 1D, 形状为 [n_samples])
        y_pred_logits: 预测 logits (torch.tensor, 形状为 [n_samples, n_classes])
    返回:
        macro_f1: 宏平均 F1 分数 (float)
    """
    y_true_np = y_true.cpu().numpy()
    y_pred = torch.argmax(y_pred_logits, dim=1).cpu().numpy()  # 类别预测，形状 [n_samples]
    f1 = f1_score(y_true_np, y_pred, average=average)
    return f1

def macro_pr_auc_multi_label(y_true, y_pred_logits):
    """
    计算多标签场景下的 Macro PR-AUC（使用 average_precision_score 的 average='macro'）
    参数:
        y_true: 真实标签矩阵 (torch.tensor, 形状为 [n_samples, n_labels])
        y_pred_logits: 预测 logits 矩阵 (torch.tensor, 形状为 [n_samples, n_labels])
    返回:
        macro_pr_auc: 宏平均 PR-AUC (float)
    """
    y_true = torch.as_tensor(y_true).cpu().numpy()
    y_pred_probs = torch.sigmoid(y_pred_logits).cpu().numpy()
    
    # 直接使用 average='macro'
    macro_pr_auc = average_precision_score(y_true, y_pred_probs, average='macro')
    return macro_pr_auc

def macro_f1_multi_label(y_true, y_pred_logits, average='micro',threshold=0.5):
    """
    计算多标签场景下的 F1 Score
    参数:
        y_true: 真实标签矩阵 (torch.tensor, 形状为 [n_samples, n_labels])
        y_pred_logits: 预测 logits 矩阵 (torch.tensor, 形状为 [n_samples, n_labels])
        threshold: 阈值，sigmoid 后大于此值为正标签 (默认 0.5)
    返回:
        macro_f1: 宏平均 F1 分数 (float)
    """
    y_true = torch.as_tensor(y_true).cpu().numpy()
    y_pred_probs = torch.sigmoid(y_pred_logits)
    y_pred = (y_pred_probs >= threshold).int().cpu().numpy()
    
    f1 = f1_score(y_true, y_pred, average=average)
    return f1

def accuracy_multi_class(y_true, y_pred_logits):
    y_true = torch.as_tensor(y_true).cpu().numpy()
    y_pred = torch.argmax(y_pred_logits, dim=1).cpu().numpy()  # 类别预测，形状 [n_samples]
    
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy

def accuracy_multi_label(y_true, y_pred_logits, threshold=0.5):
    y_true = torch.as_tensor(y_true).cpu().numpy()
    y_pred_probs = torch.sigmoid(y_pred_logits)
    y_pred = (y_pred_probs >= threshold).int().cpu().numpy()
    
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy

def macro_accuracy_multi_label(y_true, y_pred_logits, threshold=0.5):
    """
    计算多标签场景下的 Weighted Accuracy（输入为 torch.tensor）
    参数:
        y_true: 真实标签矩阵 (torch.tensor, n_samples x n_labels)
        y_pred_logits: 预测 logits 矩阵 (torch.tensor, n_samples x n_labels)
        threshold: 阈值，sigmoid 后大于此值为正标签 (默认 0.5)
    返回:
        weighted_acc: 加权准确率 (float)
    """
    # 确保输入是 torch.tensor
    y_true = torch.as_tensor(y_true)
    y_pred_logits = torch.as_tensor(y_pred_logits)
    
    # 将 logits 转换为概率（sigmoid），然后根据阈值转换为二值预测
    y_pred_probs = torch.sigmoid(y_pred_logits)
    y_pred = (y_pred_probs >= threshold).int()
    
    # 转换为 NumPy 以使用 scikit-learn 的函数
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    
    n_samples, n_labels = y_true.shape
    
    # 每个标签的准确率
    per_label_acc = [accuracy_score(y_true_np[:, i], y_pred_np[:, i]) for i in range(n_labels)]
    
    return np.mean(per_label_acc)

def macro_accuracy_multi_class(y_true, y_pred_logits):
    """
    计算多分类场景下的 Weighted Accuracy（输入为 torch.tensor）
    参数:
        y_true: 真实标签 (torch.tensor, 1D)
        y_pred_logits: 预测 logits (torch.tensor, n_samples x n_classes)
    返回:
        weighted_acc: 加权准确率 (float)
    """
    # 确保输入是 torch.tensor
    y_true = torch.as_tensor(y_true)
    y_pred_logits = torch.as_tensor(y_pred_logits)
    
    # 将 logits 转换为类别预测
    y_pred = torch.argmax(y_pred_logits, dim=1)
    
    # 将张量转换为 NumPy 以使用 scikit-learn 的混淆矩阵
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true_np, y_pred_np)
    n_classes = cm.shape[0]
    
    # 每个类别的样本数
    class_counts = cm.sum(axis=1)
    total_samples = class_counts.sum()
    
    # 每个类别的准确率
    per_class_acc = cm.diagonal() / class_counts
    
    return np.mean(per_class_acc)


def get_node_classification_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the node classification task
    :param predicts: Tensor, shape (num_samples, n_classes)
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    # predicts = F.softmax(predicts.cpu().detach(), dim=1).numpy()
    # labels = labels.cpu().numpy()
    # roc_auc = roc_auc_score(y_true=labels, y_score=predicts, average='macro', multi_class='ovr')
    # return {'roc_auc': roc_auc}

    y_pred_logits = predicts.detach().cpu()
    y_true = labels.cpu()
    if labels.shape != predicts.shape: # multi-class
        prauc = macro_pr_auc_multi_class(y_true, y_pred_logits)
        micro_f1 = macro_f1_multi_class(y_true, y_pred_logits, average='micro')
        macro_f1 = macro_f1_multi_class(y_true, y_pred_logits, average='macro')
        acc = accuracy_multi_class(y_true, y_pred_logits)
        # macro_acc = macro_accuracy_multi_class(y_true, y_pred_logits)
    else: #multi-label
        prauc = macro_pr_auc_multi_label(y_true, y_pred_logits)
        micro_f1 = macro_f1_multi_label(y_true, y_pred_logits, average='micro')
        macro_f1 = macro_f1_multi_label(y_true, y_pred_logits, average='macro')
        acc = accuracy_multi_label(y_true, y_pred_logits)
        # macro_acc = macro_accuracy_multi_label(y_true, y_pred_logits)

    result = {'prauc': prauc, 'micro-f1': micro_f1, 'macro-f1': macro_f1, 'acc': acc}

    return  result

# def get_node_classification_metrics(predicts: torch.Tensor, labels: torch.Tensor, detailed: bool=False):
#     """
#     get metrics for the node classification task
#     :param predicts: Tensor, shape (num_samples, )
#     :param labels: Tensor, shape (num_samples, )
#     :return:
#         dictionary of metrics {'metric_name_1': metric_1, ...}
#     """
#     # predicts = F.softmax(predicts.cpu().detach(), dim=1).numpy()
#     # labels = labels.cpu().numpy()
#     # roc_auc = roc_auc_score(y_true=labels, y_score=predicts, average='macro', multi_class='ovr')
#     # return {'roc_auc': roc_auc}

#     acc_list = []
#     if labels.shape != predicts.shape: # multi-class, convert format to comply with multi-label
#         y_true = labels
#         labels = labels.unsqueeze(-1)
#         y_pred = F.softmax(predicts, dim=-1)
#         predicts = predicts.argmax(dim=-1, keepdim=True)
#     else: # multi-label, 
#         y_true = labels
#         y_pred = F.sigmoid(predicts)
#         predicts = y_pred > 0.5

#     y_true = y_true.cpu().numpy()
#     y_pred = y_pred.detach().cpu().numpy()
#     labels = labels.cpu().numpy()
#     predicts = predicts.detach().cpu().numpy()
#     for i in range(labels.shape[1]):
#         correct = labels[:, i] == predicts[:, i]
#         acc_list.append(float(np.sum(correct))/len(correct))
#     result = {'accuracy': sum(acc_list)/len(acc_list)}

#     if detailed: #更详细的评估指标
#         precision_macro = precision_score(labels, predicts, average='macro')
#         result['precision_macro'] = precision_macro
#         precision_micro = precision_score(labels, predicts, average='micro')
#         result['precision_micro'] = precision_micro
#         recall_macro = recall_score(labels, predicts, average='macro')
#         result['recall_macro'] = recall_macro  
#         recall_micro = recall_score(labels, predicts, average='micro')
#         result['recall_micro'] = recall_micro
#         f1_macro = f1_score(labels, predicts, average='macro')
#         result['f1_macro'] = f1_macro  
#         f1_micro = f1_score(labels, predicts, average='micro')
#         result['f1_micro'] = f1_micro
#         # if labels.shape[1] > 1: # multi-label
#         #     roc_auc_macro = roc_auc_score(y_true, y_pred, average='macro')
#         #     result['roc_auc_macro'] = roc_auc_macro
#         #     roc_auc_micro = roc_auc_score(y_true, y_pred, average='micro')
#         #     result['roc_auc_micro'] = roc_auc_micro
#         # else: # multi-class
#         #     roc_auc = roc_auc_score(y_true, y_pred, multi_class='ovo')
#         #     result['roc_auc'] = roc_auc

#     return  result