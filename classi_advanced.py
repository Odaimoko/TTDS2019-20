import os
from chardet import detect
from argparse import ArgumentParser
import json
from util import remove_punc
import re
import nltk
import gensim
from gensim.models import Word2Vec
# Constant
OOV = "#OOV"
stemmer = nltk.stem.PorterStemmer()

with open('stopwords.txt', 'r') as f:
    stopwords = set(f.read().split())


def get_args():
    parser = ArgumentParser()
    # -m: modes, split by comma ','
    # fd: get features from train files and output *f*eat.*d*ict
    # cf: convert train/test file into feats.* version
    parser.add_argument("-m", dest = 'mode', required = True)
    parser.add_argument("-pre_opt", dest = 'pre_opt', required = False)
    
    return parser.parse_args()


# get file encoding type


def get_encoding_type(file):
    with open(file, 'rb') as f:
        rawdata = f.read()
    return detect(rawdata)['encoding']


def get_class_ids(file):
    with open(file, 'r') as f:
        con = f.read()
    lines = con.split('\n')
    class_to_id_dict = {}
    for l in lines:
        name, num = l.split('\t')
        class_to_id_dict[name] = int(num)
    return class_to_id_dict


def get_features(all_term_set, preprocess_func, func_opt, line):
    '''

    :param all_term_set: the set of all terms
    :param preprocess_func: function, input line, return a list of tokens
    :param line: document
    :return: nothing
    '''
    terms = preprocess_func(line, func_opt)
    [all_term_set.add(t) for t in terms]


def convert_to_feat_vec(terms_with_id, preprocess_func, func_opt, line):
    '''

    :param terms_with_id:
    :param preprocess_func:
    :param line:
    :return:  list of term ids
    '''
    set_of_terms = set()
    for t in set(preprocess_func(line, func_opt)):
        if t in terms_with_id:
            set_of_terms.add(terms_with_id[t])
        else:
            set_of_terms.add(terms_with_id[OOV])
    
    return list(set_of_terms)


def pro_multi_steps(line, opt):
    '''
        @opt: set of parameters
    '''
    
    # no unicode char
    line = line.encode('ascii', 'ignore').decode()
    tokens = line.strip().split()
    if 'rm_punc' in opt:
        tokens = [remove_punc(t) for t in tokens]
    # case folding
    if 'lower' in opt:
        tokens = [t.lower() for t in tokens]
    if 'stop' in opt:
        tokens = [t for t in tokens if t not in stopwords]
    if 'exp_link' in opt:
        pass
    else:
        tokens = [t for t in tokens if not ('http:' in t or 'https:' in t)]
    if 'stem' in opt:
        tokens = [stemmer.stem(t) for t in tokens]
    if 'hash_at' in opt:
        new_tok = []
        for t in tokens:
            if t and t[0] in {'#','@'}:
                new_tok.append(t)
                new_tok.append(t[1:])
            else:
                new_tok.append(t)
        tokens = new_tok
    return tokens


if __name__ == '__main__':
    args = get_args()
    modes = args.mode.split(',')
    root_dir = 'tweetsclassification'
    test_file = os.path.join(root_dir, "Tweets.14cat.test")
    train_file = os.path.join(root_dir, "Tweets.14cat.train")
    class_id_file = os.path.join(root_dir, 'classIDs.txt')
    class_ids = get_class_ids(class_id_file)
    feature_dict_file = os.path.join(root_dir, 'feats.dic')
    
    opt = set(args.pre_opt.split(','))
    
    if 'fd' in modes:  # and not os.path.exists(feature_dict_file):
        all_terms = {OOV}
        # dont forget OOV term
        with open(train_file, 'rb') as f:
            data = f.read()
            encode = detect(data)['encoding']
        with open(train_file, 'r', encoding = encode) as f:
            for line in f.readlines():
                if line.strip():
                    tweet_id, tweet, cate = line.strip().split('\t')
                    get_features(all_terms, pro_multi_steps, opt, tweet)
        # write to feats.dic
        terms_with_id = {t: i + 1 for i, t in enumerate(all_terms)}
        with open(feature_dict_file, 'w') as f:
            json.dump(terms_with_id, f)
    else:
        # read from already saved feature dictionary
        with open(feature_dict_file, 'r') as f:
            terms_with_id = json.load(f)
    
    if 'cf' in modes:
        for file in [train_file, test_file]:
            converted_file = os.path.join(root_dir, 'feats.' +
                                          ('train' if 'train' in file else 'test'))
            print(converted_file)
            with open(file, 'rb') as f:
                data = f.read()
                encode = detect(data)['encoding']
            with open(file, 'r', encoding = encode) as f:
                os.truncate(converted_file, 0)
                with open(converted_file, 'a') as o:
                    for line in f.readlines():
                        if not line.strip():
                            continue
                        tweet_id, tweet, cate = line.strip().split('\t')
                        feat_vec = sorted(convert_to_feat_vec(
                            terms_with_id, pro_multi_steps, opt, tweet))
                        out_str = '%s\t%s # %s\n' % (
                            class_ids[cate],
                            ' '.join(
                                ["%d:1" % term_id for term_id in feat_vec]),
                            tweet_id)
                        o.write(out_str)
