#!/usr/bin/python
#-*-coding: utf-8 -*-
"""
@author:     Pierre Puiseux
@license:    LGPL
"""
from datetime import timedelta, timezone

import numpy as np
from numpy import ndarray, nan, infty, asarray
from numbers import Number
import math


"""
# Pour info 
# datetime.datetime.timestamp() est, à 10**(-6) sec près, LE NOMBRE DE SECONDES depuis le 01-01-1970
# le timestamp de '2001-09-09 01:46:40' est 10*12 = 1 000 000 000 000 (mille milliards de mille sabords)
>>> date_to_milliseconds('2001-09-09 01:46:40')/(10**12)
... 1.0
# Le zéro utc des timestamp est '1970-01-01 00:00:00' (j'avais 16 ans)
>>> date_to_milliseconds('1970-01-01 00:00:00')
Out[79]: 0
>>> date_to_milliseconds('1970-01-01 00:00:00.001')
Out[81]: 1
# Je suis né à -507 513 600 000 = moins 507,5 milliards de milli-secondes
# Je suis né à -507 513 600     = moins un demi milliard de secondes
>>> date_to_milliseconds('1953-12-02')
Out[82]: -507513600000
# La datetime utc minimale accessible est '0001-01-01 00:00:00+00:00'
>>> DT.datetime(1,1,1,0,0,0,tzinfo=DT.timezone.utc).timestamp()
Out[95]: -62135596800.0
# La datetime utc maximale accessible est '9999-12-31 23:59:59.999999'
>>> DT.datetime(9999, 12, 31, 23, 59, 59, 999999, tzinfo=DT.timezone.utc).timestamp()
Out[95]: 253402300800.0

"""
DATE_TIME_FORMAT = '%A %d/%m/%Y à %Hh%M'
"""
>>> dt = datetime.datetime.now()
>>> print(dt)
==> 2022-01-17 17:48:20.153525 #Presque isoformat...
>>> print(dt.timestamp())
==> 1642438100.153525
>>> print(dt.strftime('%A %d/%m/%Y à %Hh%M')) # Mon format
==> Monday 17/01/2022 à 17h48
>>> print(dt) # isoformat
==> 2022-01-17T17:48:20.153525
"""
timezone
def dtfmt(dt:timedelta, fmt:str='dict') -> tuple or dict :#(w,d,h,m,s)
    """On travaille sur le nombre entier de secondes,
    :returns tuple : (d, h, m, s) le nb de jours, d'heures, de minutes, de secondes"""
    nbs = int(dt.total_seconds())
    spj = 24 * 3600
    spw = spj*7
    spy = spj*365
    y, s = divmod(nbs,spy)
    w, s = divmod(s,spw)
    d, s = divmod(s, spj)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if fmt == 'dict' :
        return dict(annees=y, semaines=w, jours=d, heures=h, minutes=m, secondes=s)
    else :
        T = y, w, d, h, m, s
        while T[0]==0 :
            T = T[1:]
        return T if T else 0
    # return '%dsem., %dj & %dh:%dmn:%ss' % (w, d, h, m, s)

def style(s, style):
    return style + s + '\033[0m'

def gris(s):
    return style(s, '\033[90m')

def red(s):
    return style(s, '\033[91m')

def green(s):
    return style(s, '\033[92m')

def yellow(s):
    return style(s, '\033[93m')

def blue(s):
    return style(s, '\033[94m')

def pink(s):
    return style(s, '\033[95m')

def bleupale(s):
    return style(s, '\033[96m')

def blancgras(s):
    return style(s, '\033[97m')

def rayé(s):
    return style(s, '\033[9m')

def bold(s):
    return style(s, '\033[1m')

def encadré(s):
    return style(s, '\033[51m')


def underline(s):
    return style(s, '\033[4m')


def fmt(r, prec=3):
    """(récursif) Formate un tableau ou dict avec prec décimales"""
    if prec == -1 : return r
    cls = type(r)
    if cls in (int,bool) :
        return r

    if issubclass(cls, Number) :
        if np.isnan(r) or math.isnan(r) or np.isinf(r) or math.isinf(r) :
            return r
#         prec = 10.0**prec
        return round(r, prec)
    elif issubclass(cls, (list, tuple)) :
        return cls([fmt(x,prec) for x in r])
    elif isinstance(r, ndarray) :
        if len(r.shape)>0 :#On a des array(3.14) de shape==() qui posent probleme
            return asarray([fmt(x,prec) for x in r])
        else :
#             debug(r,r.shape, type(r), repr(r), float(r))
            return round(float(r),prec)
#     elif isinstance(r, QPointF) :
#         return fmt((r.x(),r.y()),prec)
    elif issubclass(cls, dict):
        return dict((key,fmt(value,prec)) for (key, value) in r.items())
    else :
        return r

def toUnicode(r):
    """(récursif) Formate un tableau ou dict en unicode
    pour remplacer les QString('toto') par u'toto' => cf suspentage"""
    cls = type(r)
    if issubclass(cls, (list, tuple)) :
        return cls([toUnicode(x) for x in r])
    elif isinstance(r, ndarray) :
        return asarray([toUnicode(x) for x in r])
    elif issubclass(cls, dict):
        return dict((key,toUnicode(value)) for (key, value) in r.items())
    else :
        return r

if __name__=='__main__' :
    if 1 :
        # def style(s, style):
        #     return style + s + '\033[0m'
        #
        # def randomcolor(s):
        #     return style(s, '\033[4m')
        #
        for i in range(11) :
            for j in range(11) :
                n = 10 * i + j
                if n > 108: break
                print("\033[%dm code : %3d\033[m"%(n, n))
        exit()
    X = [[1.268811468596966, 0.3365727279966774], [1.1390328407287598, 0.07332829385995865]]
    X = [[-0.5279636979103088, nan], [0.7137287259101868, -infty]]
    X = [[-0.5279636979103088, nan], [0.7137287259101868, -infty], [1.268811468596966, 0.3365727279966774], [1.1390328407287598, 0.07332829385995865], [1.2571670716747245, 0.2148051133421408], [1.2206038430660453, -0.0507648238639925], [1.5545598268508911, -0.0048885527066886425]]

    print(fmt(X,4))
#     exit()
    print(fmt(tuple(X)))
    print(fmt(asarray(X)))
    D = {'Cx0': 0.02, 'Cxp': 0.01, 'Cxs': 0.016, 'S': 28.201750906217605, 'b': 5.897692607389747, 'c': 0.28999999986302366, 'd': 0.0, 'l': 2.89304324, 'mp': 85.0, 'mv': 3.0, 's': 7.3726443586, 'xGv': 0.9539636527457533, 'zGv': 1.7292584212368416, 'zs': -2.578503039144598}
    print(fmt(D))
#     mexit()
    D = {'nbpes': [7, 11], 'ts': [0.0, 0.08280030143208625, ], 'modes': ['courbure']}
    print(fmt((D,D)))
    print(fmt(dict(dico1=D, dico2=D)))
    methode = ('cubic', ((1, 0, -0.25), (2, 0, 0)))
    print(fmt(methode))
#     modech = {'QPointF':PyQt5.QtCore.QPointF(-1.0020506804808074, 0.0025114052142376124),'nbpes': [7, 11, 33, 15], 'ts': [0.0, 0.08280030143208625, 0.18859864810049215, 0.8996632881984424, 1.0], 'modes': ['courbure', 'courbure', 'courbure', 'linear']}
#     print(fmt(modech))
    I = 5*[0,1]
    print(fmt(I))
    print(fmt(-9.717034075195107e+15))
