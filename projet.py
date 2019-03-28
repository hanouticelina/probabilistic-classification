#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import scipy

import utils as ut


# Question 1
def getPrior(df, class_value=1):
    """Calcule la probabilité a priori de la classe donnée et un intervalle
     de confiance de risque 5 %.


    Parameters
    ----------
    df : pandas.DataFrame
        La base d'examples.
    class_value : int, optional
        La valeur de la classe en question.


    Returns
    -------
    dict of str: float
        L'estimation de la probabilité a priori de la classe, ainsi qu'un
        intervalle de confiance de risque 5 %.

    Notes
    -----
    Les données sont repérées par les clefs `"estimation"`, `"min5pourcent"` et
    `"max5pourcent"`.

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
    """Un classifieur attribuant la classe majoritaire à chaque example.
    """

    def estimClass(self, attrs):
        """Estime la classe d'un individu donné.

        Parameters
        ----------
        attrs : dict of str: int
            La table d'association contenant la valeur pour chaque nom d'attribut
            de l'individu.

        Returns
        -------
        int
            La classe estimée de l'individu.

        """
        return 1

    def statsOnDF(self, df):
        """Calcule des statistiques sur la base donnée.

        Les statistiques considérées dans cette méthode sont le rappel, la
        précision et le nombre d'examples de chaque classe qui ont été bien ou
        mal classés.

        Parameters
        ----------
        df : pandas.DataFrame
            La base d'examples dont on veut les statistiques.

        Returns
        -------
        stats : dict of str: int
            Des statistiques sur la base d'examples donnée.

        Notes
        -----
        Les statistiques sont repérées par les clefs
            `"VP"` : le nombre d'individus de classe 1 bien classés,
            `"VN"` : le nombre d'individus de classe 0 bien classés,
            `"FP"` : le nombre d'individus de classe 0 mal classés,
            `"FN"` : le nombre d'individus de classe 1 mal classés,
            `"precision"` : la proportion d'identifications positives correctes,
            `"rappel"` : le proportion d'examples positifs correctement classés.

        """
        stats = {'VP': 0, 'VN': 0, 'FP': 0,
                 'FN': 0, 'Precision': 0, 'Rappel': 0}
        for ex in df.itertuples():
            example = ex._asdict()
            estimate = self.estimClass(example)
            if example['target'] == 1:
                if estimate == 1:
                    stats['VP'] += 1
                else:
                    stats['FN'] += 1
            else:
                if estimate == 1:
                    stats['FP'] += 1
                else:
                    stats['VN'] += 1
        stats['Precision'] = stats['VP'] / (stats['VP'] + stats['FP'])
        stats['Rappel'] = stats['VP'] / (stats['VP'] + stats['FN'])

        return stats


# Question 3
def getPriorAttribute(df, attr):  # P(attr)
    """Calcule la distribution de probabilité d'un attribut.

    Parameters
    ----------
    attr : str
        Le nom de l'attribut en question.

    Returns
    -------
    pandas.Series
        La distribution de probabilité de l'attribut.

    """
    freqs = df.groupby([attr])[attr].count()
    total = len(df)
    return freqs / total


def getJoint(df, attrs):  # P(attrs)
    """Calcule la distribution de probabilité jointe de plusieurs attributs.

    Parameters
    ----------
    attrs : list of str
        Les noms de l'ensemble d'attributs en question.

    Returns
    -------
    probas : pandas.Series
        La distribution de probabilité jointe des attributs.

    """
    freqs = df.groupby(attrs)[attrs[0]].count()
    total = len(df)
    # print(freqs, total, freqs / total)
    return freqs / total


def reduce_update(dico, oth):
    """Rajoute les dictionnaires du deuxième argument dans le premier argument.

    Parameters
    ----------
    dico : dict of number:(dict of number:float)
        Le dictionnaire à mettre à jour.
    oth : dict of number:(dict of number:float)
        Le dictionnaire avec les données à transmettre.

    Returns
    -------
    dico : dict of number:(dict of number:float)
        Le dictionnaire mis à jour.
    """
    for k in oth.keys():
        try:
            dico[k].update(oth[k])
        except:
            dico.update(oth)
    return dico


def P2D_l(df, attr):
    """Calcule la probabilité d'un attribut sachant la classe.

    Parameters
    ----------
    df : pandas.DataFrame
        La base d'examples.
    attr : str
        Le nom de l'attribut en question.

    Returns
    -------
    dict of int: (dict of number: float)
        Un dictionnaire associant à la classe `t` un dictionnaire qui associe à
        la valeur `a` de l'attribut la probabilité
        .. math:: P(attr=a|target=t).

    """
    raw_dico = dict(getJoint(df, ['target', attr]) /
                    getPriorAttribute(df, 'target'))
    dicos = [{k_t: {k_a: proba}} for (k_t, k_a), proba in raw_dico.items()]
    res = {}
    reduce(reduce_update, [res] + dicos)
    return res


def P2D_p(df, attr):
    """Calcule la probabilité de la classe sachant un attribut.

    Parameters
    ----------
    df : pandas.DataFrame
        La base d'examples.
    attr : str
        Le nom de l'attribut en question.

    Returns
    -------
    dict of number: (dict of int: float)
        Un dictionnaire associant à la valeur `a` de l'attribut un dictionnaire
        qui associe à la classe `t` la probabilité
        .. math:: P(target=t|attr=t).

    """
    raw_dico = dict(getJoint(df, [attr, 'target']
                             ) / getPriorAttribute(df, attr))
    dicos = [{k_t: {k_a: proba}} for (k_t, k_a), proba in raw_dico.items()]
    res = {}
    reduce(reduce_update, [res] + dicos)
    return res


class ML2DClassifier(APrioriClassifier):
    """Un classifieur basé sur le maximum de vraisemblance.

    Parameters
    ----------
    attr : str
        Le nom de l'attribut observé.
    likelihoods : dict of int: (dict of number: float)
        La vraisemblance d'observer une valeur de l'attribut pour chacune des
        valeurs de la classe.

    """

    def __init__(self, df, attr):
        APrioriClassifier.__init__(self)
        self.attr = attr
        self.likelihoods = P2D_l(df, attr)

    def estimClass(self, attrs):
        """Estime la classe d'un individu donné.

        Parameters
        ----------
        attrs : dict of str: int
            La table d'association contenant la valeur pour chaque nom d'attribut
            de l'individu.

        Returns
        -------
        int
            La classe qui maximise la vraisemblance de l'attribut de l'individu.

        """
        target_likelihood = [(c, self.likelihoods[c][attrs[self.attr]])
                             for c in self.likelihoods.keys()]
        sorted_target_likelihood = sorted(
            target_likelihood, key=lambda x: (x[1], -x[0]))
        return sorted_target_likelihood[-1][0]


class MAP2DClassifier(APrioriClassifier):
    """Un classifieur basée sur le maximum a posteriori.

    Parameters
    ----------
    attr : str
        Le nom de l'attribut observé.
    probabilities : dict of number: (dict of int: float)
        La distribution a posteriori des classes après avoir observé chacune des
        valeurs de l'attribut.

    """

    def __init__(self, df, attr):
        APrioriClassifier.__init__(self)
        self.attr = attr
        self.probabilities = P2D_p(df, attr)

    def estimClass(self, attrs):
        """Estime la classe d'un individu donné.

        Parameters
        ----------
        attrs : dict of str: int
            La table d'association contenant la valeur pour chaque nom d'attribut
            de l'individu.

        Returns
        -------
        int
            La classe de plus grande probabilité sachant la valeur prise par
            l'attribut.

        """
        target_attr = [(c, p)
                       for c, p in self.probabilities[attrs[self.attr]].items()]
        sorted_target_attr = sorted(target_attr, key=lambda x: (x[1], -x[0]))
        return sorted_target_attr[-1][0]


# Question 4
def memory_size(size):
    """Calcule la taille en mémoire d'un ensemble de tables etant donnée leurs
    cardinalités.

    Parameters
    ----------
    size : int
        Cardinalité des tables pour lesquelles on souhaite calculer la taille
        en mémoire.

    Returns
    -------
    d : dict of str : int
        dictionnaire repésentant la taille en mémoire des tables en kilooctets,
        megaoctets et gigaoctets.
    o : int
        taille en mémoire des tables en octets.

    """
    kio = 2**10
    d = {'go': 0, 'mo': 0, 'ko': 0}
    d['ko'] = size // (kio) % kio
    d['mo'] = (size - d['ko'] * (kio)) // (kio**2) % kio
    d['go'] = (size - d['mo'] * (kio**2) - d['ko'] * (kio)) // (kio**3) % kio
    o = size - d['mo'] * (kio**2) - d['ko'] * (kio) - d['go'] * (kio**3)
    return d, o


def print_size(size, d, o, attributs):
    """Affiche le dictionnaire representant la taille en mémoire des tables .. math:: P(target|attr1,..,attrK).

    Parameters
    ----------
    size : int
        Cardinalité des tables pour lesquelles on souhaite calculer la taille
        en mémoire.
    d : dict of str : int
        dictionnaire repésentant la taille en mémoire des tables en kilooctets,
        megaoctets et gigaoctets.
    o : int
        taille en mémoire des tables en octets.
    attributs : list of str
        liste des attributs necessaire pour construire la prédiction de target.

    """
    s = ""
    for key, value in d.items():
        if(value != 0):
            s += str(value) + str(key) + " "
    if o < size:
        s += str(o) + "o"
    print(len(attributs), " variable(s) : ", size, " octets", s)


def nbParams(data, attr=None):
    """Calcule et affiche la taille en mémoire des tables .. math:: P(target|attr1,..,attrK).

    Parameters
    ----------
    data : pandas.DataFrame
        dataframe contenant les données issues de la base initiale.
    attr : list of str
        liste contenant les attributs pris en considération dans le calcul.

    Notes
    -----
    On considère qu'un float est représenté sur 8 octets.
    Cette méthode utilise les méthodes auxiliaire  memory_size et print_size.

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
    """Calcule et affiche la taille en mémoire nécessaire pour représenter les
    tables en supposant l'indépendance des variables.

    Parameters
    ----------
    data : pandas.DataFrame
        La base d'examples.
    attr : list of str
        liste contenant les attributs pris en considération dans le calcul.

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
    """Dessine un graphe orienté représentant naïve Bayes

    Parameters
    ----------
    df: pandas.DataFrame
        La base d'examples.
    attr : str
        nom de la colonne qui représente la classe et qui est utilisé comme racine.

    Returns
    -------
        Graphe du modèle naive bayes
    """
    s = ""
    for k in df.keys():
        if(k != attr):
            s += " " + k
    return ut.drawGraph(attr + "->{" + s + "}")


def nbParamsNaiveBayes(df, attr, list_attr=None):
    """Calcule et affiche la taille en mémoire nécessaire pour représenter les
    tables et en supposant l'indépendance des variables.

    Parameters
    ----------
    df : pandas.DataFrame
        La base d'examples.
    attr : str
        nom de la colonne qui représente la classe et qui est utilisé comme racine.
    list_attr : list of str
        liste contenant les attributs pris en considération dans le calcul.

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


def params(df, P2D):  # TODO: doc
    """
    renvoie un dictionnaire les vraisemblances d'observer chaque attribut sachant les valeurs prises par target.
    Parameters
    ----------
    df : pandas.DataFrame
        La base d'examples.
    P2D : function
        fonction qui calcule une probabilité conditionnelle, soit la probabilité de la classe sachant un attribut (P2D_p)
        ou Calcule la probabilité d'un attribut sachant la classe (P2D_l)
    Returns
    ---------
    dict of int : (dict of number: float)
        un dictionnaire les vraisemblances d'observer chaque attribut sachant les valeurs prises par target.
    """
    return {attr: P2D(df, attr) for attr in df.keys() if attr != 'target'}


class MLNaiveBayesClassifier(APrioriClassifier):
    """Un classifieur basée sur le maximum de vraisemblance et supposant
    l'indépendance contionnelle sachant la classe entre toute paire d'attributs.

    Parameters
    ----------
    params: dict of int: (dict of number: float)
        dictionnaire contenant les vraisemblances d'observer chaque attribut
        sachant les valeurs prises par target.

    classes: numpy.array
        array numpy contenant les valeurs prises par target (les classes).

    """

    def __init__(self, df):
        self.params = params(df, P2D_l)
        self.classes = df['target'].unique()

    def estimProbas(self, data):
        """Calcule la vraisemblance.

        Parameters
        ----------
        data: pandas.DataFrame
            La base d'examples.

        Returns
        -------
        dict of int: float
            Dictionnaire contenant la vraisemblance d'observer les attributs
            d'un individu pour chacune des valeurs prises par target.
        """
        def coefficients(value):
            return [lh[value][data[attr]] if data[attr] in lh[value] else 0
                    for attr, lh in self.params.items()]

        dico = {c: reduce(lambda x, y: x * y, coefficients(c))
                for c in self.classes}
        return dico

    def estimClass(self, data):
        """Estime la classe d'un individu donné.

        Parameters
        ----------
        attrs : dict of str: int
            La table d'association contenant la valeur pour chaque nom d'attribut
            de l'individu.

        Returns
        -------
        int
            La classe de plus grande probabilité sachant la valeur prise par
            l'attribut.

        """
        dico = self.estimProbas(data)
        estimates = sorted(dico.items())
        return max(estimates, key=lambda x: x[1])[0]


def normaliseDico(dico):
    # C'est une distribution de probabilité => normalisation nécessaire
    proba = sum(dico.values())
    return {k: (v / proba if proba > 0. else 1 / len(dico)) for k, v in dico.items()}


class MAPNaiveBayesClassifier(APrioriClassifier):
    """Un classifieur basée sur le maximum a posteriori et supposant
    l'indépendance contionnelle sachant la classe entre toute paire d'attributs.

    Parameters
    ----------
    params:dict of int: (dict of number: float)
        dictionnaire contenant les vraisemblances d'observer chaque attribut
        sachant les valeurs prises par target.

    classes: numpy.array
        array numpy contenant les valeurs prises par target (les classes).

    """

    def __init__(self, df):
        self.params = params(df, P2D_l)  # params(df, P2D_p)
        self.classes = df['target'].unique()
        self.priors = {c: getPrior(df, class_value=c)[
            'estimation'] for c in self.classes}

    def estimProbas(self, data):
        """Calcule la vraisemblance.

        Parameters
        ----------
        data: pandas.DataFrame
            La base d'examples.

        Returns
        -------
        dict of int: float
            Dictionnaire contenant la vraisemblance d'observer les attributs
            d'un individu pour chacune des valeurs prises par target.
        """
        def coefficients(value):
            return [lh[value][data[attr]] if data[attr] in lh[value] else 0
                    for attr, lh in self.params.items()]

        dico = {c: self.priors[c] * reduce(lambda x, y: x * y, coefficients(c))
                for c in self.classes}
        return normaliseDico(dico)

    def estimClass(self, data):
        dico = self.estimProbas(data)
        estimates = sorted(dico.items())
        return max(estimates, key=lambda x: x[1])[0]


# Question 6
def isIndepFromTarget(df, attr, x):
    """
    Verifie si l'attribut attr est indépendant de target au seuil x%.

    Parameters
    ----------
    df: pandas.DataFrame
    La base d'examples.
    attr : str
    nom de l'attribut pour lequel on souhaite verifier si il y'a indépendance avec target.
    x : float
    seuil de confiance.
    Returns
    -------
    boolean
    True si attr est indépendant de target au seuil de x%, False sinon.
    """
    attr_values = df[attr].unique()
    dico = np.zeros((len(attr_values), 2))
    index_of = {v: i for i, v in enumerate(attr_values)}
    target_attr = df.groupby(['target', attr])['target'].count().to_dict()
    for (cl, v), n in target_attr.items():
        dico[index_of[v]][cl] += n

        chi2, p, lib, expected = scipy.stats.chi2_contingency(dico)
        return x < p


class ReducedMLNaiveBayesClassifier(MLNaiveBayesClassifier):
        """
        Classifieur par maximum de vraissemblance utilisant le modèle naïve Bayes reduit.

        Notes
        ---------
        Le tableau deletion permet de recupérer tous les attributs indépendant de target au seuil threshold%
        et ensuite on les supprime de params afin de ne pas les prendre en consideration."""
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
    """
    Classifieur basé sur le principe du maximum a posteriori et utilisant le modèle naïve Bayes reduit.

    Notes
    ---------
    Le tableau deletion permet de recupérer tous les attributs indépendant de target au seuil threshold%
    et ensuite on les supprime de params afin de ne pas les prendre en consideration.
    """
    def __init__(self, df, threshold):
        MAPNaiveBayesClassifier.__init__(self, df)
        deletion = []
        for attr, dico in self.params.items():
            if isIndepFromTarget(df, attr, threshold):
                deletion.append(attr)
                for attr in deletion:
                    del self.params[attr]

                    def draw(self):
                        """
                        Dessine un graphe orienté représentant naïve Bayes réduit.
                        """
                        s = ""
                        for k in self.params:
                            s += " " + k
                            return ut.drawGraph('target' + "->{" + s + "}")


def mapClassifiers(dico, train):
    """
    Representation graphique des classifieurs selon leurs couple (Precision, Rappel).
    Parameters
    ----------
    dico: dict of str:instance
        dictionnaire {nom:instance de classifieur}.
    train : pandas.DataFrame
        La base d'examples.

    Returns
    ----------
        Affiche une représentation dans l'espace de ces classifiers selon les valeurs (Precision, Rappel).
    """
    precision = [v.statsOnDF(train)['Precision'] for k, v in dico.items()]
    recall = [v.statsOnDF(train)['Rappel'] for k, v in dico.items()]
    labels = dico.keys()
    fig, ax = plt.subplots()
    ax.scatter(precision, recall, c="red", marker="x")
    for i, l in enumerate(labels):
        ax.annotate(l, (precision[i], recall[i]))
    plt.show()


def MutualInformation(df, x, y):  # I(x;y)
    """
    Calcule l'information mutuelle entre les colonnes x et y dans le dataFrame df.
    Parameters
    ----------
    df : pandas.DataFrame
        La base d'examples.
    x: str
        nom d'une colonne dans df.
    y: str
        nom d'une colonne dans df.
    Returns
    ----------
    float
        l'information mutuelle entre x et y.
    """
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
    res = num.copy()
    for x in den.index.levels[0]:
        res[x] = num[x] / den[x]
    return res


def ConditionalMutualInformation(df, x, y, z):  # I(x;y|z)
    """
    Calcule l'information mutuelle conditionnelle entre les colonnes x et y dans le dataFrame df en considérant le fait qu'elles soient indépendantes de z.
    Parameters
    ----------
    df : pandas.DataFrame
        La base d'examples.
    x: str
        nom d'une colonne dans df.
    y: str
        nom d'une colonne dans df.
    z: str
        nom d'une colonne dans df.
    Returns
    ----------
    float
        l'information mutuelle conditionnelle entre x et y.
    """
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
    """
    Calcule la moyenne des poids pour une matrice symétrique de diagonale nulle.
    Parameters
    ----------
    matrix : numpy.ndarray
        matrice symétrique de diagonale nulle.
    Returns
    ---------
    float
        moyenne des poids d'une matrice symétrique de diagonale nulle.
    """
    size = np.sqrt(matrix.size)
    size *= size - 1
    return matrix.sum() / size


def SimplifyConditionalMutualInformationMatrix(matrix):
    """
    Annule toutes les valeurs plus petites que sa moyenne dans une matrice
    symétrique de diagonale nulle.
    Parameters
    ----------
    matrix : numpy.ndarray
        matrice symétrique de diagonale nulle.

    """
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


def P2D_l_TAN(df, cond, attr):  # P(attr | 'target', cond)
    joint_target_cond_attr = getJoint(df, ['target', cond, attr])
    joint_target_cond = getJoint(df, ['target', cond])
    raw_dico = dict(divide(joint_target_cond_attr, joint_target_cond))
    dicos = [{(k_t, k_c): {k_a: proba}}
             for (k_t, k_c, k_a), proba in raw_dico.items()]
    res = {}
    reduce(reduce_update, [res] + dicos)
    return res


class MAPTANClassifier(APrioriClassifier):
    def __init__(self, df):
        self._init_arcs(df)
        self.df = df
        self.single_params = params(df, P2D_l)
        self.double_params = {}
        self._update_params(df)
        self.classes = df['target'].unique()
        self.priors = {c: getPrior(df, class_value=c)[
            'estimation'] for c in self.classes}

    def _init_arcs(self, df):
        matrix = np.array([[0 if x == y else ConditionalMutualInformation(df, x, y, 'target')
                            for x in df.keys() if x != 'target']
                           for y in df.keys() if y != 'target'])
        SimplifyConditionalMutualInformationMatrix(matrix)  # side-effect
        self.oriented_arcs = OrientConnexSets(
            df, Kruskal(df, matrix), 'target')

    def _update_params(self, df):
        for cond, attr in self.oriented_arcs:
            self.single_params.pop(attr)
            self.double_params[attr, cond] = P2D_l_TAN(df, cond, attr)

    def estimProbas(self, data):
        def coefficients(value):
            liste = [lh[value][data[attr]] if data[attr] in lh[value] else 0
                     for attr, lh in self.single_params.items()]
            liste += [(tan[value, data[cond]][data[attr]] if data[attr] in tan[value, data[cond]] else 0.)
                      if (value, data[cond]) in tan else 1 / len(tan)
                      for (attr, cond), tan in self.double_params.items()]
            return liste

        dico = {c: self.priors[c] * reduce(lambda x, y: x * y, coefficients(c))
                for c in self.classes}
        return normaliseDico(dico)

    def estimClass(self, data):
        dico = self.estimProbas(data)
        estimates = sorted(dico.items())
        return max(estimates, key=lambda x: x[1])[0]

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
