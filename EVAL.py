"""
Evaluation package
If used as main, it will read in eval results and gt file and output eval results
"""
import os
import numpy as np


def parse_retrival_results(filename):
    '''
    
    :param filename: X.results, only a single file
    :return: List of retrived results for each query.
            Each element of the list is a dict, key:v = docid:(rank, score)
    '''
    results = []
    with open(filename, 'r') as f:
        prev_q = None
        single_query = {}
        for line in f.readlines():
            q, _, doc, rank, score, _ = line.strip().split()
            if q != prev_q and prev_q is not None:
                results.append(single_query)
                single_query = {}
            single_query[int(doc)] = (int(rank), float(score))
            prev_q = q
        if prev_q is not None:
            results.append(single_query)
    return results


def parse_gt(filename):
    '''
    
    :param filename:
    :return: List of queries gt. Each gt is a dict mapping from doc id (int) to its relevance.
    '''
    gt = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            _, docs = line.strip().split(':')
            docs = docs.strip().split()
            docs_tuple = eval("(" + ','.join(docs) + ",)")
            docs_dict = {tup[0]: tup[1] for tup in docs_tuple}
            gt.append(docs_dict)
    return gt


def associate(gts, system):
    '''
    
    :param gts:
    :param system:
    :return: A list of doc matrix.
             A doc is converted into [docid res_rank score relevance],
             and a matrix will be built for each query.
    '''
    query_mats = []
    for gt, result_dict in zip(gts, system):  # one for each query
        single_query_mat = []
        for doc in gt:
            if doc in result_dict:
                res_rank, score = result_dict[doc]
                single_query_mat.append([doc, res_rank, score, gt[doc]])  # res_rank, res_score, relevance
            else:
                single_query_mat.append([doc, float('inf'), 0, gt[doc]])
        query_mats.append(np.array(single_query_mat))
    
    return query_mats


def prec_at_k(query_mat, k):
    """
        Buggy. Since I do not record how many docs are retrieved, it cannot compute precision.
    :param query_mat:
    :param k:
    :return:
    """
    num_tp = np.sum(query_mat[:, 1] <= k)
    return num_tp / k


def recall_at_k(query_mat, k = None):
    num_tp = np.sum(query_mat[:, 1] <= k)
    return num_tp / query_mat.shape[0]


def r_prec(query_mat):
    k = query_mat.shape[0]
    return prec_at_k(query_mat, k)


def ap(query_mat):
    ranks = np.sort(query_mat[:, 1])
    precisions = (np.arange(ranks.shape[0]) + 1) / ranks
    return precisions.mean()


def dcg_at_k(query_mat, k):
    '''
        k can be larger than query_mat's length, because in this function k is only a filter.
    :param query_mat:
    :param k:
    :return:
    '''
    remaining_docs = query_mat[query_mat[:, 1] <= k]
    dcg = 0
    for doc_info in remaining_docs:
        doc, rank, score, rele = doc_info
        if rank == 1:
            dcg += rele
        else:
            dcg += rele / np.log2(rank)
    return dcg


def idcg_at_k(query_mat, k):
    '''
        k <= len(query_mat). if k>len, k will be reduced to len, since docs whose rank is larger than k will only have 0 gain.
    :param query_mat:
    :param k:
    :return:
    '''
    ranked_mat = query_mat[np.argsort(query_mat[:, 3])[::-1]]  # sort by relevance score
    # if k> # docs, idcg will be the same
    k = min(k, ranked_mat.shape[0])
    idcg = 0
    for i in range(k):
        doc, rank, score, rele = ranked_mat[i]
        if i == 0:
            idcg += rele
        else:
            idcg += rele / np.log2(i + 1)
    return idcg


def ndcg_at_k(query_mat, k):
    return dcg_at_k(query_mat, k) / idcg_at_k(query_mat, k)


if __name__ == '__main__':
    # read in prediction and GT
    eval_dir = "systems/"
    out_dir = "results_/"
    gts = parse_gt(os.path.join(eval_dir, 'qrels.txt'))
    HEAD = '	P@10	R@50	r-Precision	AP	nDCG@10	nDCG@20\n'
    num_metrics = 6
    num_systems = 6
    all_out_str = HEAD
    for i in range(1, 1 + num_systems):
        system = parse_retrival_results(os.path.join(eval_dir, "S%d.results" % i))
        # build evaluate index
        query_mats = associate(gts, system)
        out_str = HEAD
        system_eval = []
        # eval separately
        for j, qmat in enumerate(query_mats):
            single_query_eval = [prec_at_k(qmat, 10), recall_at_k(qmat, 50), r_prec(qmat), ap(qmat),
                                 ndcg_at_k(qmat, 10),
                                 ndcg_at_k(qmat, 20)]
            # print("dcg\t", dcg_at_k(qmat, 10))
            # print("idcg_at_10\t", idcg_at_k(qmat, 10))
            # print("dcg\t", dcg_at_k(qmat, 20))
            # print("idcg_at_10\t", idcg_at_k(qmat, 20))
            out_str += '\t'.join([str(j + 1)] + ['{:.3f}'.format(s) for s in single_query_eval]) + '\n'
            system_eval.append(single_query_eval)
        # average over all queries for each system
        system_eval = np.array(system_eval)
        system_mean = np.mean(system_eval, axis = 0)
        out_str += '\t'.join(['mean'] + ['{:.3f}'.format(s) for s in system_mean]) + '\n'
        all_out_str += '\t'.join(['S%d' % i] + ['{:.3f}'.format(s) for s in system_mean]) + '\n'
        
        # output for each system
        with open(os.path.join(out_dir, "S%d.eval" % i), 'wb') as f:
            f.write(out_str.encode())
    
    # output All.eval
    with open(os.path.join(out_dir, "All.eval"), 'wb') as f:
        f.write(all_out_str.encode())