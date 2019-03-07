#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from functools import reduce

import numpy as np

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


class APrioriClassifier(ut.AbstractClassifier):
    """
    """

    def estimClass(self, attrs):
        return 1

    def statsOnDF(self, df):
        d = {'VP': 0, 'VN': 0, 'FP': 0, 'FN': 0, 'Precision': 0, 'Rappel': 0}

        for i in range(df.shape[0]):
            attrs = ut.getNthDict(df, i)
            if attrs['target'] == 1:
                if self.estimClass(attrs) == 1:
                    d['VP'] += 1
                else:
                    d['FN'] += 1
            if attrs['target'] == 0:
                if self.estimClass(attrs) == 0:
                    d['VN'] += 1
                else:
                    d['FP'] += 1

        d['Precision'] = d['VP'] / (d['VP'] + d['FP'])
        d['Rappel'] = d['VP'] / (d['VP'] + d['FN'])

        return d


class ML2DClassifier(APrioriClassifier):

    def __init__(self, df, attr):
        APrioriClassifier.__init__(self)
        self.df = df
        self.attr = attr
        self.likelihoods = P2D_l(df, attr)

    def estimClass(self, attrs):
        target_attr = [(t, self.likelihoods[t][attrs[self.attr]])
                       for t in self.likelihoods.keys()]
        sorted_target_attr = sorted(target_attr, key=lambda x: (x[1], -x[0]))
        return sorted_target_attr[-1][0]


def npParams(df, attrs):
    pass
    # card =
