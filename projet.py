#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import scipy

import utils as ut


# Question 1
def getPrior(df, class_value=1):
     """Calcule la probabilite a priori de la classe 1 et de l'intervalle de confiance a 95% pour cette probabilité


    Parameters
    ----------
    df : pandas.DataFrame
        dataframe contenant les données issues de la base initiale
    class_value : int
        Valeur de la classe pour laquelle on souhaite estimer la probabilité et l'intervalle de confiance à 95%


    Returns
    -------
    Float
        estimation : l'estimation de la probabilité a priori de la classe
    Float
        min5pourcent : plus petite valeur dans l'intervalle de confiance
    Float
        max5pourcent : plus grande valeur dans l'intervalle de confiance
    """
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
    classe qui représente un classifier à priori et estime  naivement la classe de chaque individu pour la classe majoritaire ( et donc la classe 1)
    """

    def estimClass(self, attrs):
            """
            renvoie l'estimation de la classe pour un individu

            Parameters
            -----
            attrs : dict
                dictionnaire des attributs de l'individu
            Returns
            -----
                1 : l'estimation a priori de la classe de l'individu
            """
        return 1

    def statsOnDF(self, df):
            """
            renvoie un dictionnaire contenant les valeurs VP, VN, FN, FP ainsi que le rappel et la précision

            Parameters
            -----
            df : pandas.DataFrame
                dataframe contenant les données issues de la base initiale
            Returns
            -----
                Dictionnaire contenant:
                VP :  nombre d'individus avec target=1 et classe prévue=1
                VN :  nombre d'individus avec target=0 et classe prévue=0
                FP :  nombre d'individus avec target=0 et classe prévue=1
                FN :  nombre d'individus avec target=1 et classe prévue=0
                Precision : proportion d'identifications positives correcte
                Rappel : proportion de résultats positifs réels identifiée correctement
            Notes
            ------
            pandas.DataFrame.itertuples :


            """
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
    """
    Calcule la probabilité 𝑃(𝑎𝑡𝑡𝑟|𝑡𝑎𝑟𝑔𝑒𝑡)

    Parameters
    -----
    attrs : dict
        dictionnaire des attributs de l'individu
    df : pandas.DataFrame
        dataframe contenant les données issues de la base initiale
    Returns
    -----
        Dictionnaire associant à chaque valeur t de target un dictionnaire qui associe à chaque attribut a la probabilité 𝑃(𝑎𝑡𝑡𝑟=𝑎|𝑡𝑎𝑟𝑔𝑒𝑡=𝑡).
    Notes
    -----

    """
    attr_values = df[attr].unique()
    target_attr = df.groupby(['target', attr])['target'].count()
    target = df.groupby(['target'])['target'].count()
    raw_dico = dict(target_attr / target)
    dicos = [{k_t: {k_a: raw_dico[k_t, k_a]}} for k_t, k_a in raw_dico.keys()]
    res = {}
    reduce(reduce_update, [res] + dicos)
    return res


def P2D_p(df, attr):
    """
    Calcule la probabilité 𝑃(𝑡𝑎𝑟𝑔𝑒𝑡|𝑎𝑡𝑡𝑟)

    Parameters
    -----
    attrs : dict
        dictionnaire des attributs de l'individu
    df : pandas.DataFrame
        dataframe contenant les données issues de la base initiale
    Returns
    -----
        Dictionnaire associant à chaque valeur a des attributs un dictionnaire qui associe à chaque valeur de target a la probabilité 𝑃(𝑡𝑎𝑟𝑔𝑒𝑡|𝑎𝑡𝑡𝑟).
    Notes
    -----

        """
    attr_values = df[attr].unique()
    target_attr = df.groupby([attr, 'target'])['target'].count()
    attr = df.groupby([attr])[attr].count()
    raw_dico = dict(target_attr / attr)
    dicos = [{k_t: {k_a: raw_dico[k_t, k_a]}} for k_t, k_a in raw_dico.keys()]
    res = {}
    reduce(reduce_update, [res] + dicos)
    return res


class ML2DClassifier(APrioriClassifier):
    """
    Classe qui représente un classifieur basée sur le principe du maximum de vraisemblance (Maximum likelihood)

    Attributs
    -----
    df : pandas.DataFrame
        dataframe contenant les données issues de la base initiale
    attr : string
        attribut observé
    likelihoods : dict
        dictionnaire contenant la vraisemblance d'observer attr pour chacune des valeurs prises par target

    """
    def __init__(self, df, attr):
        APrioriClassifier.__init__(self)
        self.df = df
        self.attr = attr
        self.likelihoods = P2D_l(df, attr)

    def estimClass(self, attrs):
        """
        renvoie l'estimation de la classe pour un individu

        Parameters
        -----
        attrs : dict
            dictionnaire des attributs de l'individu
        Returns
        -----
        int
            Position du maximum trouvé dans la table likelihoods
        """
        target_attr = [(c, self.likelihoods[c][attrs[self.attr]])
                       for c in self.likelihoods.keys()]
        sorted_target_attr = sorted(target_attr, key=lambda x: (x[1], -x[0]))
        return sorted_target_attr[-1][0]


class MAP2DClassifier(APrioriClassifier):
    """
    Classe qui représente un classifieur basée sur le principe du maximum a Posteriori

    Attributs
    -----
    df : pandas.DataFrame
        dataframe contenant les données issues de la base initiale
    attr : string
        attribut observé
    probabilities : dict
        dictionnaire contenant la distribution a posteriori de target après avoir observé attr
    """

    def __init__(self, df, attr):
        APrioriClassifier.__init__(self)
        self.df = df
        self.attr = attr
        self.probabilities = P2D_p(df, attr)

    def estimClass(self, attrs):
        """
        renvoie l'estimation de la classe pour un individu

        Parameters
        -----
        attrs : dict
            dictionnaire des attributs de l'individu
        Returns
        -----
        int
            Position du maximum trouvé dans la table probabilities
        """
        target_attr = [(c, p)
                       for c, p in self.probabilities[attrs[self.attr]].items()]
        sorted_target_attr = sorted(target_attr, key=lambda x: (x[1], -x[0]))
        return sorted_target_attr[-1][0]


# Question 4
def memory_size(size):
    """
    Calcule la taille en mémoire d'un ensemble de tables etant donnée leurs cardinalités
    Parameters
    -----
    size : int
        Cardinalité des tables pour lesquelles on souhaite calculer la taille en mémoire
    Returns
    -----
    d : dict
        dictionnaire repésentant la taille en mémoire des tables en kilooctets, megaoctets et gigaoctets
    o : int
        taille en mémoire des tables en octets
    """
    kio = 2**10
    d = {'go': 0, 'mo': 0, 'ko': 0}
    d['ko'] = size // (kio) % kio
    d['mo'] = (size - d['ko'] * (kio)) // (kio**2) % kio
    d['go'] = (size - d['mo'] * (kio**2) - d['ko'] * (kio)) // (kio**3) % kio
    o = size - d['mo'] * (kio**2) - d['ko'] * (kio) - d['go'] * (kio**3)
    return d, o


def print_size(size, d, o, attributs):
    """
    Affiche le dictionnaire representant la taille en mémoire des tables 𝑃(𝑡𝑎𝑟𝑔𝑒𝑡|𝑎𝑡𝑡𝑟1,..,𝑎𝑡𝑡𝑟𝑘)
    Parameters
    -----
    size : int
        Cardinalité des tables pour lesquelles on souhaite calculer la taille en mémoire
    d : dict
        dictionnaire repésentant la taille en mémoire des tables en kilooctets, megaoctets et gigaoctets
    o : int
        taille en mémoire des tables en octets
    attributs : list
        liste des attributs necessaire pour construire la prédiction de target
    Returns
    -----
    Affiche le dictionnaire de la taille memoire des tables
    """
    s = ""
    for key, value in d.items():
        if(value != 0):
            s += str(value) + str(key) + " "
    if o < size:
        s += str(o) + "o"
    print(len(attributs), " variable(s) : ", size, " octets", s)


def nbParams(data, attr=None):
    """
    Calcule la taille en mémoire des tables 𝑃(𝑡𝑎𝑟𝑔𝑒𝑡|𝑎𝑡𝑡𝑟1,..,𝑎𝑡𝑡𝑟𝑘)
    Parameters
    -----
    data : pandas.DataFrame
        dataframe contenant les données issues de la base initiale
    attr : list
        liste contenant ['target', 'attr1', 'attr2',...,'attrK']
    Returns
    -----
        Affiche la taille en mémoire des tables 𝑃(𝑡𝑎𝑟𝑔𝑒𝑡|𝑎𝑡𝑡𝑟1,..,𝑎𝑡𝑡𝑟𝑘)
    Notes
    -----
    Ici , un float est représenté sur 8 octets
    """
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
    """
    Calcule la taille en mémoire nécessaire pour représenter les tables et en supposant l'indépendance des variables
    Parameters
    -----
    data : pandas.DataFrame
        dataframe contenant les données issues de la base initiale
    attr : list
        liste contenant ['target', 'attr1', 'attr2',...,'attrK']
    Returns
    -----
        Affiche la taille en mémoire des tables

    """
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
    """
    Dessine le graphe representant un modèle naive bayes
    Parameters
    -----
    df: pandas.DataFrame
        dataframe contenant les données issues de la base initiale
    attr : string
        nom de la colonne qui représente la classe
    Returns
    -----
        Graphe du modèle naive bayes
    """
    s = ""
    for k in df.keys():
        if(k != attr):
            s += " " + k
    return ut.drawGraph(attr + "->{" + s + "}")


def nbParamsNaiveBayes(df, attr, list_attr=None):
    """
    Calcule la taille en mémoire nécessaire pour représenter les tables et en supposant l'indépendance des variables
    Parameters
    -----
    df : pandas.DataFrame
        dataframe contenant les données issues de la base initiale
    attr : string
        nom de la colonne qui représente la classe
    list_attr : list
        liste contenant ['target', 'attr1', 'attr2',...,'attrK']

    Returns
    -----
        Affiche la taille en mémoire des tables

    """
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
    """
    Classe qui représente un classifieur basée sur le principe du maximum de vraisemblance et qui utilise l'hypothèse du Naive Bayes
    Attributs
    -----
    params: dict
        dictionnaire contenant les vraisemblances d'observer chaque attribut sachant les valeurs prises par target

    classes: numpy.array
        array numpy contenant les valeurs prises par target (les classes)

    """
    def __init__(self, df):
        self.params = params(df, P2D_l)
        self.classes = df['target'].unique()

    def estimProbas(self, data):
        """
        Calcule la vraisemblance
        Parameters
        -----
        data: pandas.DataFrame
            dataframe contenant les données issues de la base initiale
        Returns
        -----
        dict
            Dictionnaire contenant la vraisemblance d'observer les attributs d'un individu pour chacune des valeurs prises par target
        """
        def coefficients(value):

            return [lh[value][data[attr]] if data[attr] in lh[value] else 0
                    for attr, lh in self.params.items()]

        dico = {c: reduce(lambda x, y: x * y, coefficients(c))
                for c in self.classes}
        return dico

    def estimClass(self, data):
        """
        renvoie l'estimation de la classe pour un individu

        Parameters
        -----
        data: pandas.DataFrame
            dataframe contenant les données issues de la base initiale
        Returns
        -----
        int
        Position du maximum trouvé dans la table des probabilités
        """
        dico = self.estimProbas(data)
        estimates = sorted(dico.items())
        return max(estimates, key=lambda x: x[1])[0]


class MAPNaiveBayesClassifier(APrioriClassifier):
    """
    Classe qui représente un classifieur basée sur le principe du maximum a posteriori et qui utilise l'hypothèse du Naive Bayes
    Attributs
    -----
    params: dict
        dictionnaire contenant les vraisemblances d'observer chaque attribut sachant les valeurs prises par target

    classes: numpy.array
        array numpy contenant les valeurs prises par target (les classes)

    """
    def __init__(self, df):
        self.params = params(df, P2D_l)  # params(df, P2D_p)
        self.classes = df['target'].unique()
        # self.exp = len(self.params) - 1
        self.priors = {c: getPrior(df, class_value=c)[
            'estimation'] for c in self.classes}

    def estimProbas(self, data):
        """
        Calcule la vraisemblance
        Parameters
        -----
        data: pandas.DataFrame
            dataframe contenant les données issues de la base initiale
        Returns
        -----
        dict
            Dictionnaire contenant la vraisemblance d'observer les attributs d'un individu pour chacune des valeurs prises par target
        """
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


def getPriorAttribute(df, attr):  # P(attr)
    total = len(df)
    probas = df.groupby([attr])[attr].count() / total
    return probas


def getJoint(df, attrs):  # P(attrs)
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
    def union(f_i, f_j, d):
        nonlocal _set
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
            union(f_i, f_j, d)
            arcs.append((keys[i], keys[j], d))
    return arcs


def ConnexSets(arcs):
    """Finds the connected components of the given graph.

    Parameters
    ----------
    arcs : list of (hashable, hashable, int)
        The list of arcs of the form: (node, node, distance), that define
        a graph.

    Returns
    -------
    list of (set of hashable)
        The list of connected components of the graph.


    """

    def find(x):
        """Finds the representative and the component of the given node.

        Parameters
        ----------
        x : hashable
            The node.

        Returns
        -------
        hashable or None
            The representative of the node's component.
        (set of hashable) or None
            The node's component.

        """
        nonlocal dico
        try:
            while True:
                rep = dico[x]
                if rep[0] == x:
                    return rep
                x = rep[0]

        except:
            return None, None

    dico = {}  # dict of hashable: (hashable, set of hashable)
    components = []  # list of (set of hashable)
    for x, y, _ in arcs:
        rep_x, set_x = find(x)
        rep_y, set_y = find(y)
        if rep_x is None:  # the representative of `y` absorbs that of `x`
            if rep_y is None:  # if the rep of `y` does not exist yet,
                set_y = {y}    # `y` is its rep itself
                components.append(set_y)
                dico[y] = (y, set_y)
            set_y.add(x)
            dico[x] = (y, set_y)
        else:
            if rep_y is None:  # the rep of `x` absorbs that of `y`
                set_x.add(y)
                dico[y] = (rep_x, set_x)
            elif rep_x != rep_y:  # do the same if reps are different
                set_x.update(set_y)
                components.remove(set_y)
                dico[y] = (rep_x, set_x)
            # do nothing if already in the same component
    return components


def OrientConnexSets(df, arcs, target):  # TODO: modify
    def adjacent_vertices(arcs):
        adjacents = {}
        for x, y, _ in arcs:
            try:
                adjacents[x].append(y)
            except:
                adjacents[x] = [y]
            try:
                adjacents[y].append(x)
            except:
                adjacents[y] = [x]
        return adjacents

    def find_root(component):
        nonlocal mutual_info
        max_val = -1.
        root = None
        for attr in component:
            if mutual_info[attr] > max_val:
                max_val = mutual_info[attr]
                root = attr
        return root

    def add_oriented_arcs(component, root):
        nonlocal adjacents, visited, oriented_arcs
        stack = [root]
        while stack != []:
            attr = stack.pop()
            visited[attr] = True
            for adj in adjacents[attr]:
                if visited[adj] is True:
                    continue
                stack.append(adj)
                oriented_arcs.append((attr, adj))

    adjacents = adjacent_vertices(arcs)
    mutual_info = {attr: MutualInformation(df, target, attr)
                   for attr in df.keys() if attr != target}
    components = ConnexSets(arcs)
    component_and_roots = [(compo, find_root(compo)) for compo in components]
    # endpoint = {attr: False for attr in df.keys() if attr != target}
    visited = {attr: False for attr in df.keys() if attr != target}
    oriented_arcs = []
    for compo, root in component_and_roots:
        add_oriented_arcs(compo, root)
    return oriented_arcs


# direct method, i.e. w/o finding roots
def OrientConnexSets_prev(df, arcs, target):
    def adjacent_vertices(arcs):
        adjacents = {}
        for x, y, _ in arcs:
            try:
                adjacents[x].append(y)
            except:
                adjacents[x] = [y]
            try:
                adjacents[y].append(x)
            except:
                adjacents[y] = [x]
        return adjacents

    mutual_info = [(attr, MutualInformation(df, target, attr))
                   for attr in df.keys() if attr != target]
    endpoint = {attr: False for attr in df.keys() if attr != target}
    visited = {attr: False for attr in df.keys() if attr != target}
    mutual_info.sort(key=lambda x: x[1])
    adjacents = adjacent_vertices(arcs)
    oriented_arcs = []
    attr, _ = mutual_info.pop()
    stack = [attr]
    for _ in range(len(visited)):
        if stack == []:
            while visited[mutual_info[len(mutual_info) - 1][0]] is True:
                mutual_info.pop()
            stack.append(mutual_info.pop()[0])
        attr = stack.pop()
        visited[attr] = True
        try:
            for adj in adjacents[attr]:
                if visited[adj] is True:
                    continue
                stack.append(adj)
                if endpoint[adj] is True:  # only ona input arc per vertex
                    oriented_arcs.append((adj, attr))
                else:
                    endpoint[adj] = True
                    oriented_arcs.append((attr, adj))
                # adjacents[adj].remove(attr)
            # adjacents.pop(attr)
        except:
            continue
    return oriented_arcs


def P2D_l_TAN(df, cond, attr):  # P(attr | cond, 'target')
    joint_target_cond_attr = getJoint(df, ['target', cond, attr])
    joint_target_cond = getJoint(df, ['target', cond])
    raw_dico = dict(divide(joint_target_cond_attr, joint_target_cond))
    dicos = [{(k_t, k_c): {k_a: raw_dico[k_t, k_c, k_a]}}
             for k_t, k_c, k_a in raw_dico.keys()]
    res = {}
    reduce(reduce_update, [res] + dicos)
    return res


class MAPTANClassifier(APrioriClassifier):
    def __init__(self, df):
        self._init_arcs(df)
        self.single_params = params(df, P2D_l)
        self.double_params = {}
        # print(self.params)
        self._update_params(df)
        self.classes = df['target'].unique()
        self.priors = {c: getPrior(df, class_value=c)[
            'estimation'] for c in self.classes}

    def _init_arcs(self, df):
        matrix = np.array([[0 if x == y else ConditionalMutualInformation(df, x, y, 'target')
                            for x in df.keys() if x != 'target']
                           for y in df.keys() if y != 'target'])
        SimplifyConditionalMutualInformationMatrix(matrix)  # side-effect
        liste_test = [('trestbps', 'chol', 0.6814942282235203),
                      ('age', 'trestbps', 0.641718295908513),
                      ('age', 'thalach', 0.6365766485465845),
                      ('chol', 'oldpeak', 0.5246930555244587),
                      ('oldpeak', 'slope', 0.25839871090530614),
                      ('chol', 'ca', 0.2528327956181666)]
        # liste_test = Kruskal(df, matrix)
        self.oriented_arcs = OrientConnexSets(
            df, liste_test, 'target')

    def _update_params(self, df):
        for tail, head in self.oriented_arcs:
            self.single_params.pop(head)
            self.double_params[head, tail] = P2D_l_TAN(df, tail, head)

    def estimProbas(self, data):
        def coefficients(value):
            liste = [lh[value][data[attr]] if data[attr] in lh[value] else 0
                     for attr, lh in self.single_params.items()]
            liste += [(tan[value, data[cond]][data[attr]] if data[attr] in tan[value, data[cond]] else 0.)
                      if (value, data[cond]) in tan else 0.
                      for (attr, cond), tan in self.double_params.items()]
            return liste

        dico = {c: self.priors[c] * reduce(lambda x, y: x * y, coefficients(c))
                for c in self.classes}
        return normaliseDico(dico)

    def draw(self):
        children = ""
        for attr in self.single_params:
            children += " " + attr
        for attr, _ in self.double_params:
            children += " " + attr
        arcs = 'target->{' + children + '}'
        for tail, head in self.oriented_arcs:
            arcs += ';' + tail + '->' + head
        return ut.drawGraph(arcs)
