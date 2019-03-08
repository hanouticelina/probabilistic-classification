#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from functools import reduce

import numpy as np
import scipy

import utils as ut


def getPrior(df):
    t_alpha = 1.96
    target_values = df.target
    mean = target_values.mean()
    std = np.sqrt(mean * (1 - mean) / target_values.size)
    min5percent = mean - t_alpha * std
    max5percent = mean + t_alpha * std
    return {'estimation': mean,
            'min5pourcent': min5percent,
            'max5pourcent': max5percent}


class APrioriClassifier(ut.AbstractClassifier):
    """
    """

    def estimClass(self, attrs):
        return 1

    def statsOnDF(self, df):
        d = {'VP': 0, 'VN': 0, 'FP': 0, 'FN': 0, 'Precision': 0, 'Rappel': 0}
        for t in df.itertuples():
            dic = t._asdict()
            e = self.estimClass(dic)
            if(dic['target'] == 1):
                if(e == 1):
                    d['VP'] += 1
                else:
                    d['FN'] += 1
            else:
                if(e == 1):
                    d['FP'] += 1
                else:
                    d['VN'] += 1
        d['Precision'] = d['VP'] / (d['VP'] + d['FP'])
        d['Rappel'] = d['VP'] / (d['VP'] + d['FN'])
        return d
# Question 3


def reduce_update(dico, oth):
    for k in oth.keys():
        try:
            dico[k].update(oth[k])
        except:
            dico.update(oth)
    return dico


def P2D_l(df, attr):
    attr_values = df[attr].unique()
    target_attr = df.groupby(['target', attr])['target'].count()
    target = df.groupby(['target'])['target'].count()
    raw_dico = dict(target_attr / target)
    dicos = [{k_t: {k_a: raw_dico[k_t, k_a]}} for k_t, k_a in raw_dico.keys()]
    res = {}
    reduce(reduce_update, [res] + dicos)
    return res


def P2D_p(df, attr):
    attr_values = df[attr].unique()
    target_attr = df.groupby([attr, 'target'])['target'].count()
    attr = df.groupby([attr])[attr].count()
    raw_dico = dict(target_attr / attr)
    dicos = [{k_t: {k_a: raw_dico[k_t, k_a]}} for k_t, k_a in raw_dico.keys()]
    res = {}
    reduce(reduce_update, [res] + dicos)
    return res


class ML2DClassifier(APrioriClassifier):

    def __init__(self, df, attr):
        APrioriClassifier.__init__(self)
        self.df = df
        self.attr = attr
        self.likelihoods = P2D_l(df, attr)

    def estimClass(self, attrs):
        target_attr = [(c, self.likelihoods[c][attrs[self.attr]])
                       for c in self.likelihoods.keys()]
        sorted_target_attr = sorted(target_attr, key=lambda x: (x[1], -x[0]))
        return sorted_target_attr[-1][0]


class MAP2DClassifier(APrioriClassifier):

    def __init__(self, df, attr):
        APrioriClassifier.__init__(self)
        self.df = df
        self.attr = attr
        self.probabilities = P2D_p(df, attr)

    def estimClass(self, attrs):
        target_attr = [(c, p)
                       for c, p in self.probabilities[attrs[self.attr]].items()]
        sorted_target_attr = sorted(target_attr, key=lambda x: (x[1], -x[0]))
        return sorted_target_attr[-1][0]
# Question 4


def memory_size(size):
    kio = 2**10
    d = {'go': 0, 'mo': 0, 'ko': 0}
    d['ko'] = size // (kio) % kio
    d['mo'] = (size - d['ko'] * (kio)) // (kio**2) % kio
    d['go'] = (size - d['mo'] * (kio**2) - d['ko'] * (kio)) // (kio**3) % kio
    o = size - d['mo'] * (kio**2) - d['ko'] * (kio) - d['go'] * (kio**3)
    return d, o


def print_size(size, d, o, attributs):
    s = ""
    for key, value in d.items():
        if(value != 0):
            s += str(value) + str(key) + " "
    if o < size:
        s += str(o) + "o"
    print(len(attributs), " variable(s) : ", size, " octets", s)


def nbParams(data, attr=None):
    size = 1
    if attr is None:
        attributs = data.keys()
    else:
        attributs = attr

    for k in attributs:
        size *= (len(data[k].unique()))
    size *= 8
    d, o = memory_size(size)
    print_size(size, d, o, attributs)


def nbParamsIndep(data, attr=None):
    memory_size = 0
    if attr is None:
        attributs = data.keys()
    else:
        attributs = attr

    for k in attributs:
        memory_size += (len(data[k].unique()))
    memory_size *= 8
    print(len(attributs), " variable(s) : ", memory_size, " octets")

# Question 5


def drawNaiveBayes(df, attr):
    s = ""
    for k in df.keys():
        if(k != attr):
            s += " " + k
    return ut.drawGraph(attr + "->{" + s + "}")


def nbParamsNaiveBayes(df, attr, list_attr=None):
    facteur = (len(df[attr].unique()))
    size = facteur
    if list_attr is None:
        attributs = df.keys()
    else:
        attributs = list_attr

    for k in attributs:
        if(k != attr):
            size += facteur * (len(df[k].unique()))
    size *= 8
    d, o = memory_size(size)
    print_size(size, d, o, attributs)


def ml_params(df):
    d = {}
    col_names = df.columns.values
    return {k: P2D_l(df, k) for k in col_names if k != 'target'}


class MLNaiveBayesClassifier(APrioriClassifier):
    def __init__(self, df):
        self.params = ml_params(df)

    def estimProbas(self, data):
        target_0 = 1
        target_1 = 1
        # TODO: update w/ reduce
        for k in self.params.keys():
            if(k != 'target' and k != 'Index'):
                if(data[k] in self.params[k][0].keys()):
                    target_0 *= self.params[k][0][data[k]]
                else:
                    target_0 = 0
                if(data[k] in self.params[k][1].keys()):
                    target_1 *= self.params[k][1][data[k]]
                else:
                    target_1 = 0
        d = {0: target_0, 1: target_1}
        # print(d)
        return d

    def estimClass(self, data):
        d = self.estimProbas(data)
        if(d[0] >= d[1]):
            return 0
        return 1


class MAPNaiveBayesClassifier(APrioriClassifier):
    def __init__(self, df):
        self.params = params(df)
        self.estimation = getPrior(df)['estimation']

    def estimProbas(self, data):
        # TODO
        pass

    def estimClass(self, data):
        d = self.estimProbas(data)
        if(d[0] >= d[1]):
            return 0
        return 1

# Question 6


def isIndepFromTarget(df, attr, x):
    attr_values = df[attr].unique()
    dico = np.zeros((len(attr_values), 2))
    index_of = {v: i for i, v in enumerate(attr_values)}
    target_attr = df.groupby(['target', attr])['target'].count().to_dict()
    for (cl, v), n in target_attr.items():
        dico[index_of[v]][cl] += n
    chi2, p, lib, expected = scipy.stats.chi2_contingency(dico)
    return x < p
