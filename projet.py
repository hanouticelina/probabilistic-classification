#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import scipy

import utils as ut


# Question 1
def getPrior(df, class_value=1):
    t_alpha = 1.96
    target_values = df.target
    freq = len(target_values[target_values ==
                             class_value]) / len(target_values)
    std = np.sqrt(freq * (1 - freq) / target_values.size)
    min5percent = freq - t_alpha * std
    max5percent = freq + t_alpha * std
    return {'estimation': freq,
            'min5pourcent': min5percent,
            'max5pourcent': max5percent}


# Question 2
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


def params(df, P2D):
    d = {}
    col_names = df.columns.values
    return {k: P2D(df, k) for k in col_names if k != 'target' and k != 'Index'}


class MLNaiveBayesClassifier(APrioriClassifier):
    def __init__(self, df):
        self.params = params(df, P2D_l)
        self.classes = df['target'].unique()

    def estimProbas(self, data):
        def coefficients(value):
            return [lh[value][data[attr]] if data[attr] in lh[value] else 0
                    for attr, lh in self.params.items()]

        dico = {c: reduce(lambda x, y: x * y, coefficients(c))
                for c in self.classes}
        return dico

    def estimClass(self, data):
        dico = self.estimProbas(data)
        estimates = sorted(dico.items())
        return max(estimates, key=lambda x: x[1])[0]


class MAPNaiveBayesClassifier(APrioriClassifier):
    def __init__(self, df):
        self.params = params(df, P2D_l)  # params(df, P2D_p)
        self.classes = df['target'].unique()
        # self.exp = len(self.params) - 1
        self.priors = {c: getPrior(df, class_value=c)[
            'estimation'] for c in self.classes}

    def estimProbas(self, data):
        def coefficients(value):
            # return [ap[data[attr]][value] if data[attr] in ap and value in ap[data[attr]] else 0
            #         for attr, ap in self.params.items()]
            return [lh[value][data[attr]] if data[attr] in lh[value] else 0
                    for attr, lh in self.params.items()]

        dico = {c: self.priors[c] * reduce(lambda x, y: x * y, coefficients(c))
                for c in self.classes}
        return MAPNaiveBayesClassifier.normaliseDico(dico)

    @classmethod
    def normaliseDico(cls, dico):
        # C'est une distribution de probabilité => normalisation nécessaire
        proba = sum(dico.values())
        return {k: (v / proba if proba > 0. else 1 / len(dico)) for k, v in dico.items()}

    def estimClass(self, data):
        dico = self.estimProbas(data)
        estimates = sorted(dico.items())
        return max(estimates, key=lambda x: x[1])[0]

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


class ReducedMLNaiveBayesClassifier(MLNaiveBayesClassifier):
    def __init__(self, df, threshold):
        MLNaiveBayesClassifier.__init__(self, df)
        deletion = []
        for attr, dico in self.params.items():
            if isIndepFromTarget(df, attr, threshold):
                deletion.append(attr)
        for attr in deletion:
            del self.params[attr]

    def draw(self):
        s = ""
        for k in self.params:
            s += " " + k
        return ut.drawGraph('target' + "->{" + s + "}")


class ReducedMAPNaiveBayesClassifier(MAPNaiveBayesClassifier):
    def __init__(self, df, threshold):
        MAPNaiveBayesClassifier.__init__(self, df)
        deletion = []
        for attr, dico in self.params.items():
            if isIndepFromTarget(df, attr, threshold):
                deletion.append(attr)
        for attr in deletion:
            del self.params[attr]

    def draw(self):
        s = ""
        for k in self.params:
            s += " " + k
        return ut.drawGraph('target' + "->{" + s + "}")


def mapClassifiers(dico, train):
    precision = [v.statsOnDF(train)['Precision'] for k, v in dico.items()]
    recall = [v.statsOnDF(train)['Rappel'] for k, v in dico.items()]
    labels = dico.keys()
    fig, ax = plt.subplots()
    ax.scatter(precision, recall, c="red", marker="x")
    for i, l in enumerate(labels):
        ax.annotate(l, (precision[i], recall[i]))
    plt.show()


def getPriorAttribute(df, attr):
    total = len(df)
    probas = df.groupby([attr])[attr].count() / total
    return probas


def getJoint(df, attrs):
    freqs = df.groupby(attrs)[attrs[0]].count()
    total = freqs.sum()
    return freqs / total


def MutualInformation(df, x, y):
    prior_x = getPriorAttribute(df, x)  # P(x)
    prior_y = getPriorAttribute(df, y)  # P(y)
    joint = getJoint(df, [x, y])        # P(x, y)
    probas_quotient = (joint / prior_x) / prior_y
    log_probas = probas_quotient.apply(np.log2)  # log2(_)
    result = joint * log_probas
    return result.sum()


def divide(num, den):
    """Divides the first argument by the second.

    Parameters
    ----------
    num : pandas.Series
        The numerator whose index length is greater than 2.
    den : pandas.Series
        The denominator whose index length is 2.

    Returns
    -------
    pandas.Series
        The quotient of the two Series.

    Notes
    -----
    The index of the denominator must be a subset of that of the numerator.
    Plus, the two Series must have the same index level.
    """
    for x in den.index.levels[0]:
        num[x] = num[x] / den[x]
    return num


def ConditionalMutualInformation(df, x, y, z):
    prior_z = getPriorAttribute(df, z)      # P(z)
    joint_z_x_y = getJoint(df, [z, x, y])   # P(z, x, y)
    joint_z_x = getJoint(df, [z, x])        # P(z, x)
    joint_z_y = getJoint(df, [z, y])        # P(z, y)
    probas_quotient = joint_z_x_y * prior_z
    probas_quotient = divide(probas_quotient, joint_z_x)
    probas_quotient = divide(probas_quotient, joint_z_y)
    log_probas = probas_quotient.apply(np.log2)  # log2(_)
    result = joint_z_x_y * log_probas
    return result.sum()


def MeanForSymetricWeights(matrix):
    size = np.sqrt(matrix.size)
    size *= size - 1
    return matrix.sum() / size


def SimplifyConditionalMutualInformationMatrix(matrix):
    mean = MeanForSymetricWeights(matrix)
    matrix[...] = np.where(matrix < mean, 0, matrix)


def Kruskal(df, matrix):  # TODO: il y a deux arêtes qui ne devraient pas? apparaître => vérifier `cmis`
    def union(f_i, f_j, d, i, j):
        nonlocal _set, arcs
        if depth(f_i) <= depth(f_j):
            _set[f_i] = f_j, depth(f_i)
            if depth(f_i) == depth(f_j):
                _set[f_j] = f_j, (depth(f_j) + 1)
        else:
            _set[f_j] = f_i, depth(f_j)

    def find(i):
        while i != _set[i][0]:
            i = _set[i][0]
        return i

    def depth(i):
        return _set[i][1]

    keys = df.keys()
    pairs = [(i, j, matrix[i][j]) for i in range(matrix.shape[0])
             for j in range(i + 1, matrix.shape[1]) if matrix[i][j] > 0.]
    pairs.sort(key=lambda x: -x[2])
    arcs = []
    _set = [(i, 1) for i in range(len(matrix))]  # parent, depth
    for i, j, d in pairs:
        f_i = find(i)
        f_j = find(j)
        if f_i != f_j:
            # print(i, j)
            # print(_set, f_i, f_j)
            union(f_i, f_j, d, i, j)
            arcs.append((keys[i], keys[j], d))
    return arcs
