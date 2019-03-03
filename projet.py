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

    def estimClass(self,attrs):
        return 1

    def statsOnDF(self,df):
        d = {'VP':0, 'VN':0, 'FP':0, 'FN':0, 'Precision':0, 'Rappel':0}
        for t in df.itertuples():
            dic=t._asdict()
            e =  self.estimClass(dic)
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
        d['Precision'] = d['VP']/(d['VP']+d['FP'])
        d['Rappel'] = d['VP']/(d['VP']+d['FN'])
        return d

def nbParams(data, attr = None):
    memory_size= 1
    d ={'go':0, 'mo':0, 'ko':0}
    kio = 2**10
    if attr is None :
        attributs = data.keys()
    else :
        attributs = attr

    for k in attributs:
        memory_size *= (len( data[k].unique()))
    memory_size *= 8

    d['ko'] = memory_size //(kio)  % kio
    d['mo'] = (memory_size - d['ko']*(kio))//(kio**2) % kio
    d['go'] = (memory_size - d['mo']*(kio**2) - d['ko']*(kio)) // (kio**3)  % kio
    o = memory_size - d['mo']*(kio**2) - d['ko']*(kio) - d['go']*(kio**3)

    s = ""
    for key,value in d.items():
        if(value != 0):
            s +=str(value)+str(key)+" "
    if o < memory_size : s += str(o)+"o"
    print(len(attributs), " variable(s) : ", memory_size, " octets", s)
    return memory_size

def nbParamsIndep(data, attr = None):
    memory_size= 0
    if attr is None :
        attributs = data.keys()
    else :
        attributs = attr

    for k in attributs:
        memory_size += (len( data[k].unique()))
    memory_size *= 8
    print(len(attributs), " variable(s) : ", memory_size, " octets")
    return memory_size
