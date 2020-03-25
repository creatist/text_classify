

import numpy as np 
import pandas as pd 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import gc
import gensim

import re

from collections import defaultdict
from string import punctuation


from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from sklearn import model_selection 


class TfidfEmbeddingVectorizer:
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = next(iter(word2vec.values())).size
        print('Self dim', self.dim)
        self.digit = re.compile(r'(\d+)')
        
    def preproc(self, text):
        return [
           t for t in text.split() if not (t.isspace() or self.digit.search(t) or t in punctuation)
        ]

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x, tokenizer=self.preproc, stop_words=None, max_df=.95, min_df=2, binary=True)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w not in punctuation and w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])
        
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


from gensim.models import word2vec
w2v_model = word2vec.Word2Vec.load("w2v.model")
w2v = dict(zip(w2v_model.wv.index2word, w2v_model.wv.syn0))
vect = TfidfEmbeddingVectorizer(w2v)


df = pd.read_csv("../data/data_preproced.csv")
print( df["label"].value_counts())
# PAYOUT        1924
# INCOME        1488
# IGNORE        1171
# STATUS         623
# BILLREMIND     534
# OVERDUES       472
# LOAN           369
# STAGING        282

label_map = {'PAYOUT': 0, 'INCOME': 1, 'IGNORE': 2, 'STATUS': 3, 
            'BILLREMIND': 4, 'OVERDUES': 5, 'LOAN': 6, 'STAGING': 7}

label_map_rev = {v:k for k,v in label_map.items()}

y = df["label"].map(label_map)
# y = df["label"]

dummies = vect.fit_transform(df.text_splited)
print(dummies.shape)

df = pd.DataFrame(dummies)

X = df

X_train, X_valid, y_train, y_valid = model_selection.\
                train_test_split(X, y, test_size=0.2, random_state=7)    
del df; gc.collect();

import lightgbm as lgb
from lightgbm import LGBMClassifier

from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report


params = {
    'multi_class': 'ovr',
    'solver': 'lbfgs'
}

lgbm_params = {
    'n_estimators': 1000,
    'max_depth': 10,
    'num_leaves':2**10+1,
    'learning_rate': 0.1,
    'objective': 'multiclass',
    'n_jobs': -1
}

# model = LGBMClassifier(**lgbm_params)
# # model = LogisticRegression(**params)
# model.fit(X_train, y_train)
# print(model)

# target_names = [label_map_rev[k] for k in  list(model.classes_)]
# print("target_names", target_names)
# y_pred = model.predict(X_valid)
# print(classification_report(y_valid, y_pred, target_names=target_names))

from sklearn.metrics import f1_score
def lgb_Metric(preds, dtrain):
    labels = dtrain.get_label()
    preds = np.round(np.clip(preds, 0, 10)).astype(int)
    score = f1_score(labels, preds, average="macro")
    return ('Metric', score, True)

model = lgb.train(params, lgb.Dataset(X_train, y_train),
                              1000,  lgb.Dataset(X_valid, y_valid),
                              verbose_eval=50, early_stopping_rounds=250,
                              feval=lgb_Metric)
