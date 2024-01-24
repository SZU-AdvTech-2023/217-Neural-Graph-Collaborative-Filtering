import numpy as np
from sklearn.metrics import roc_auc_score


def recall(rank, ground_truth, N):
    return len(set(rank[:N]) & set(ground_truth)) / float(len(set(ground_truth)))

"返回的数据  rank, ground_truth, N"
"rank参数是一个排名过的项目列表，ground_truth是真正的正例项目的集合，N是要考虑的前N个推荐项的数量。"

def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
        计算精度
    """

    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)


def average_precision(r,cut):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    Returns:
        Average precision  平均准确率

        这是计算平均准确率的函数。它接受两个参数：r是一个二进制列表，
        表示每个推荐项的相关性（非零表示相关），cut表示要考虑的前多少个推荐项。
        该函数首先将输入列表 r 转换为NumPy数组，
        然后通过调用之前定义的 precision_at_k 函数来计算每个位置k的精确度。
        最后，它计算这些精确度的平均值，得到平均准确率。如果没有相关的项（即out为空），则返回0
    """
    r = np.asarray(r)
    out = [precision_at_k(r, k + 1) for k in range(cut) if r[k]]
    if not out:
        return 0.
    return np.sum(out)/float(min(cut, np.sum(r)))


def mean_average_precision(rs):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    Returns:
        Mean average precision    平均准确率的平均值
        这是计算平均准确率的平均值的函数。它接受一个列表rs，
        其中包含多个推荐列表，每个列表都是一个二进制列表，
        表示每个推荐项的相关性。该函数通过调用 average_precision 函数
        计算每个推荐列表的平均准确率，并返回这些平均准确率的平均值，即平均准确率的平均值。
    """
    return np.mean([average_precision(r) for r in rs])


def dcg_at_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Discounted cumulative gain  折扣累积增益
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, ground_truth, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.
    Can use binary as the previous methods.
    Returns:
        Normalized discounted cumulative gain  归一化折扣累积增益

        Low but correct defination
    """
    GT = set(ground_truth)
    if len(GT) > k :
        sent_list = [1.0] * k
    else:
        sent_list = [1.0]*len(GT) + [0.0]*(k-len(GT))
    dcg_max = dcg_at_k(sent_list, k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max
"""
折扣累积增益 (DCG):
DCG是一个累积指标，用于度量排名结果的质量，尤其是在信息检索和推荐系统中。
对于一个给定的排名列表，DCG考虑了每个项目的相关性（或得分），并对这些相关性进行折扣，其中排在前面的项目具有较高的权重。
归一化折扣累积增益 (NDCG):
NDCG是对DCG的标准化，以便在不同排名列表之间进行比较。它通过将DCG值除以理想情况下的最大可能DCG值进行标准化。
NDCG的值范围在0到1之间，1表示最佳性能。


这两个指标的使用旨在测量推荐系统在返回的项目中成功找到用户可能感兴趣的项目的能力。
更高的DCG和NDCG值表示排名结果更好，更符合用户的期望。
"""
def recall_at_k(r, k, all_pos_num):
    # if all_pos_num == 0:
    #     return 0
    r = np.asfarray(r)[:k]
    return np.sum(r) / all_pos_num
"计算在前k个推荐项中的召回率的函数"

def hit_at_k(r, k):
    r = np.array(r)[:k]
    if np.sum(r) > 0:
        return 1.
    else:
        return 0.
"计算在前k个推荐项中是否有正例的函数"
"如果在前k个推荐项中有正例，则返回1.0，否则返回0.0"


def F1(pre, rec):
    if pre + rec > 0:
        return (2.0 * pre * rec) / (pre + rec)
    else:
        return 0.


def AUC(ground_truth, prediction):
    try:
        res = roc_auc_score(y_true=ground_truth, y_score=prediction)
    except Exception:
        res = 0.
    return res