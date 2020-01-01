import xml.etree.ElementTree as ET
import contractions
from collections import Counter
import pickle
from pathlib import PurePosixPath

import nltk
import string


# ────────────────────────────────────────────────────────────────────────────────

# ─── REMOVE PUNCTUATION ─────────────────────────────────────────────────────────


def remove_punc(con):
    punc = set(string.punctuation) - {'#', '@'}
    con = con.translate(str.maketrans('', '', ''.join(punc)))
    # print('Done.')
    return con


def preprocess_doc(doc_dict, stopwords):
    '''
    Preprocess the doc: casefolding, tokenize, stop, stem
    :param doc_dict: the one returned by  concat_doc
    :return:
    '''
    
    def add_doc(pos, doc):
        tokens = doc.split()
        for t in tokens:
            if term_to_index(term_dict, t, docid, pos, stopwords, porter):
                pos += 1
        return pos
    
    porter = nltk.stem.PorterStemmer()
    term_dict = {}
    # HEADLINE
    pos = 0
    docid = doc_dict['docid']
    head = contractions.fix(doc_dict['head'], slang = False)
    # Add headline information
    pos = add_doc(pos, head)
    term_dict.setdefault('#HEADLINE', {}).setdefault(docid, []).append(pos)
    text = contractions.fix(doc_dict['text'], slang = False)
    add_doc(pos, text)
    return term_dict


def term_to_index(term_dict, term, docid, pos, stopwords, stemmer):
    '''

    :param term_dict:
    :param term:
    :param docid:
    :param pos:
    :param stopwords:
    :param stemmer:
    :return: return the term if this term is added to the term_dict, "" if skipped
    '''
    term = term.lower()
    if term in stopwords:
        return ""
    if '-' in term:
        # 2TODO: what if there are other punc in the word->goes to else in the recursion
        for w in term.split('-'):
            term_to_index(term_dict, w, docid, pos, stopwords, stemmer)
    term = remove_punc(term)
    term = stemmer.stem(term)
    if term:  # term is not empty
        pos_dict = term_dict.setdefault(term, {})
        pos_dict.setdefault(docid, []).append(pos)
    return term


def merge_term_dict(all_term_dict, term_dict):
    assert type(all_term_dict) == dict and type(term_dict) == dict
    for t, v in term_dict.items():
        docid_dict = all_term_dict.setdefault(t, {})
        docid_dict.update(v)
    return
