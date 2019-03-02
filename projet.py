#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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


def P2D_l(df, attr):
    attr_values = df[attr].unique()
    target_attr = df.groupby(['target', attr])['target'].count()
    target = df.groupby(['target'])['target'].count()
    dico, dico_t, df.groupby([attr])[attr].count()
    dict(dico / dico_t)
    # TODO


class APrioriClassifier(ut.AbstractClassifier):
    """
    """

    def estimClass(self, attrs):
        return 1

    def statsOnDF(self, df):
        d = {'VP': 0, 'VN': 0, 'FP': 0, 'FN': 0, 'Precision': 0, 'Rappel': 0}

        for i in range(df.size):
            if ut.getNthDict(df, i)['target'] == 1:
                if self.estimClass() == 1:
                    d['VP'] += 1
                else:
                    d['FN'] += 1
            if ut.getNthDict(df, i)['target'] == 0:
                if self.estimClass() == 0:
                    d['VN'] += 1
                else:
                    d['FP'] += 1

        d['Precision'] = d['VP'] / (d['VP'] + d['FP'])
        d['Rappel'] = d['VP'] / (d['VP'] + d['FN'])

        return d


def npParams(df, attrs):
    pass
    # card =
