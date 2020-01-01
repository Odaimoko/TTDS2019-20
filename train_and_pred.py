import os
from chardet import detect
from argparse import ArgumentParser
import json
import numpy as np
import nltk
import pickle
from collections import Counter

# Constant
OOV = "#OOV"
stemmer = nltk.stem.PorterStemmer()

with open('stopwords.txt', 'r') as f:
    stopwords = set(f.read().split())


def get_class_ids(file):
    with open(file, 'r') as f:
        con = f.read()
    lines = con.split('\n')
    class_to_id_dict = {}
    for l in lines:
        name, num = l.split('\t')
        class_to_id_dict[name] = int(num)
    return class_to_id_dict


def get_args():
    parser = ArgumentParser()
    # -m: modes, split by comma ','
    # fd: get features from train files and output *f*eat.*d*ict
    # cf: convert train/test file into feats.* version
    # bayes
    parser.add_argument("-method", dest = 'method', required = True)
    # train, test
    parser.add_argument("-mode", dest = 'mode', required = True)
    parser.add_argument("-train_opt", dest = 'train_opt', required = False)
    
    return parser.parse_args()


def get_features(feat_file):
    '''
        TODO
    '''
    labels = []
    feats = []
    
    with open(feat_file, 'r') as f:
        for line in f.readlines():
            single_label, feat = line.strip().split('\t')
            labels.append(int(single_label))
            feats.append([[int(i) for i in single_feat.split(":")]
                          for single_feat in feat.strip().split(" ")
                          if ":" in single_feat])
    return feats, labels


def train(feats, labels, class_ids, method, opt):
    '''
    
    :param class_ids:
    :param feats: feature list, [(feat_id,value),...
    :param labels: label list
    :param method:
    :param opt:
    :return:
    '''
    model = {}
    if method == 'bayes':
        labels = np.array(labels)
        class_prob = np.zeros(len(class_ids))
        word_prob = {cl: Counter() for cl in range(len(class_ids))}
        for cl_name, cl_id in class_ids.items():
            class_prob[cl_id - 1] = np.sum(labels == cl_id)
            idx = np.nonzero(labels == cl_id)[0]
            all_words = []
            for j in idx:
                feat = feats[j]
                for single_feature, value in feat:
                    all_words.append(single_feature)
            ct = Counter(all_words)
            # TODO: Smoothing.
            # DONE: OOV
            ct[OOV] = 1e-1
            # for k, v in ct.items():
            #     ct[k] = v +1
            num_words = sum([v for k, v in ct.items()])
            for k, v in ct.items():
                ct[k] = v / num_words
            word_prob[cl_id - 1] = ct
        class_prob = class_prob / np.sum(class_prob)
        model['cl_prob'] = class_prob
        model['word_prob'] = word_prob
    return model


def predict(feats, method, model):
    if method == 'bayes':
        all_scores = []
        class_prob = np.log(model['cl_prob'])
        word_prob = model['word_prob']
        for f in feats:
            words = [tup[0] for tup in f]
            prob_word = np.zeros(len(class_prob))
            for k, prob_dict in word_prob.items():
                wz = [w if w in prob_dict else OOV for w in words]
                prob_word[k] = sum([np.log(prob_dict[w]) for w in wz])
            all_scores.append(class_prob + prob_word)
        all_scores = np.array(all_scores)
        pred = np.argmax(all_scores, axis = 1) + 1
    return pred


if __name__ == '__main__':
    args = get_args()
    modes = args.mode.split(',')
    root_dir = 'tweetsclassification'
    test_file = os.path.join(root_dir, "feats.test")
    train_file = os.path.join(root_dir, "feats.train")
    
    feature_dict_file = os.path.join(root_dir, 'feats.dic')
    
    class_id_file = os.path.join(root_dir, 'classIDs.txt')
    class_ids = get_class_ids(class_id_file)
    
    if args.train_opt:
        opt = set(args.train_opt.split(','))
    else:
        opt = None
    
    method = args.method
    model_name = os.path.join(root_dir, 'model_%s.pkl' % method)
    pred_out = os.path.join(root_dir, 'pred_%s.out' % method)
    
    if 'train' in modes:
        feat, label = get_features(train_file)
        model = train(feat, label, class_ids, method, opt)
        with open(model_name, 'wb') as f:
            pickle.dump(model, f)
    if 'test' in modes:
        # test
        feat, label = get_features(test_file)
        with open(model_name, 'rb') as f:
            model = pickle.load(f)
        
        pred = predict(feat, method, model)
        out_str = "\n".join([str(i) for i in pred])
        with open(pred_out, 'w') as f:
            f.write(out_str)
