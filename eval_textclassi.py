import os
from chardet import detect
from argparse import ArgumentParser
import json
from EVAL import *
from baseline import get_class_ids
import numpy as np

OOV = "#OOV"


def get_args():
    parser = ArgumentParser()
    # -gt: feats.test, -pred: pred.out
    parser.add_argument("-gt", dest = 'gt', required = True)
    parser.add_argument("-pred", dest = 'pred', required = True)
    
    return parser.parse_args()


def prec(gt, pred):
    # for each class, binary labels
    num_tp = np.sum(gt * pred)
    return num_tp / pred.sum()


def rec(gt, pred):
    # for each class, binary labels
    num_tp = np.sum(gt * pred)
    return num_tp / gt.sum()


def f_score(precs, recs, beta = 1):
    # for each class, binary labels
    return ((beta ** 2 + 1) * precs * recs) / (beta ** 2 * precs + recs)


def macro_f(f_scores):
    # for all classes as a whole
    return f_scores.mean()


if __name__ == '__main__':
    
    args = get_args()
    gt_file = args.gt
    pred_file = args.pred
    
    gt_label = []
    with open(gt_file, 'r') as f:
        for line in f.readlines():
            single_label, _ = line.strip().split('\t')
            gt_label.append(int(single_label))
    gt_label = np.array(gt_label)
    pred_label = []
    with open(pred_file, 'r') as f:
        for line in f.readlines():
            line = line.strip().split()
            pred_label.append(int(line[0]))
    
    pred_label = np.array(pred_label)
    
    root_dir = 'tweetsclassification'
    class_id_file = os.path.join(root_dir, 'classIDs.txt')
    class_ids = get_class_ids(class_id_file)
    acc = np.sum(pred_label == gt_label) / pred_label.shape[0]
    precs = np.zeros(len(class_ids))
    recs = np.zeros(len(class_ids))
    for cl_name, cl_id in class_ids.items():
        cl_gt = (gt_label == cl_id).astype(np.int8)
        cl_pred = (pred_label == cl_id).astype(np.int8)
        precs[int(cl_id) - 1] = prec(cl_gt, cl_pred)
        recs[int(cl_id) - 1] = rec(cl_gt, cl_pred)
    fs = f_score(precs, recs)
    mac_f = macro_f(fs)
    print('Accuracy = %.3f' % acc)
    print('Macro-F1 = %.3f' % mac_f)
    print('Results per class:')
    for i in range(len(precs)):
        print("%d: P=%.3f R=%.3f F=%.3f" % (i + 1, precs[i], recs[i], fs[i]))
