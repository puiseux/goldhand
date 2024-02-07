#!/usr/bin/python
#-*-coding: utf-8 -*-

"""
Created on 11 mai 2012
__updated__ = "2021-02-14"
@author: puiseux
utilitaires chaines de caractères,
fonctions geometriques qui marchent en 2d ET 3D
"""
# from datetime import datetime
import datetime as DT

import pandas
from dateparser import parse


utc = DT.timezone.utc
from itertools import islice
import os
import pickle
from random import random
import shutil
import sys

# from coinmarketcapapi import Response

from binance.helpers import date_to_milliseconds
from numpy import (delete, where, nan, argmax, logical_not, hstack, linspace)
from numpy import asarray, zeros  # , ones, array
from numpy.linalg import norm
from path import Path
from format import fmt
import numpy as np
from debog import debug, rdebug, mexit
# from basic.myconfig import MyConfig
from typing import Union, Optional

import dateparser
import math
import pytz
from datetime import datetime

def arrangeDataFrame(df:pandas.DataFrame,
                     form:str='%Y/%m/%d %H:%M',
                     timecol:str='time',
                     sortcol='time',
                     duplicatecols=None) -> pandas.DataFrame:
    """Un arrangement des dict qui arrivent de Binance et qui sont transformés en dataframe.
        - On crée une colonne 'datetime' HUMAN READABLE, à partir
            # du timestamp de la colonne <timecol> qui doit être un entier en millisecondes
            # et du format <form>
        - on assigne la colonne 'datetime' comme index
        - on trie (Z->A) sur les valeurs de la colonne <sortcol>
        - on supprime les duplicatas,
            i.e. ceux qui on les mêmes valeurs dans les colonnes <duplicatecols>
    """
    # df = pd.DataFrame(lth)
    if duplicatecols is None:
        duplicatecols = ['id', 'time']
    ldt = [DT.datetime.fromtimestamp(timestamp=ts / 1000, tz=utc) for ts in df[timecol]]
    ldt = [dt.strftime(form) for dt in ldt]
    # On rajoute une colonne HUMAN READABLE, 'datetime'
    df['datetime'] = ldt
    # On en fait l'index
    df.set_index('datetime', inplace=True)
    # tri avec les plus récents au début
    df.sort_values(by=sortcol, ascending=False, inplace=True)
    # On vire les duplicata,
    df.drop_duplicates(subset=duplicatecols, inplace=True)
    return df

def select(objects, selector, sort=False, verbose=False):
    """
    :returns S: un set (si sort==False) ou une list (si sort!=False) d'objets
        sélectionnés dans objects par la fonction selector
    :param objects: un iterable QUELCONQUE, contenant des objets auxquels
        on peut appliquer selector
    :param selector : une fonction de selection qui s'applique aux éléments de objects
        qui retourne True ou False : selector(obj) = True ou False
    :param sort: False ou True ou fonction de tri entre éléments de objects
    • si sort==False : pas de tri ni de objects ni de la selection
    • si sort==True :
        1. objects n'est pas trié
        2. la selection est triée comme objects,
    • si sort est une fonction,
        1. objects est trié avec sorted(objects, key=sort)
        2. la selection est triée comme objects
    """
    if sort : #sort!=False
        if sort is not True : # sort contient une fonction de tri
            objects = sorted(objects, key=sort)
        selection = list()
        for obj in  objects :
            try :
                if selector(obj) : selection.append(obj)
            except Exception as msg:
                if verbose : print(str(msg), 'objet=', str(obj))
                continue
        return selection
    else :
        selection = set()
        for obj in  objects :
            try :
                if selector(obj) : selection.add(obj)
            except Exception as msg:
                if verbose : print(str(msg), 'objet=', str(obj))
                continue
        return selection

def iterable(obj):
    try :
        iter(obj)
        return True
    except Exception :
        return False
    # return isinstance(obj, collections.Iterable)

def loadDict(self, filename, type_=''):
    with open(filename,'r'+type_) as f :
        D = eval(f.read())
    for k,v in D.items():
        setattr(self, k, v)
    return D

def pointsUniques(A, eps=1.0e-3):
    """
    #http://scipy-user.10969.n7.nabble.com/remove-duplicate-points-td4636.html
    :param A: ndarray((nbp,dim),dtype=float ou int) tableau de points contenant
        possiblement des points doubles
        Il vaut mieux que dim<<nbp, pour tri.
    :returns j_unique : le tableau des n° de points uniques
    :returns j_sorted : la permutation telle que A[j_sorted] est trié
    :returns unique_mask : un masque (tableau booleens) t.q
        j_unique == j_sorted[unique_mask]
    NB la liste des n° de points doubles donc ipd = j_sorted[logical_not(unique_mask)]
    :exemple :
    >>> Js, Um, Ju = pointsUniques(P, eps=1.0e-6)
    >>> npd = Js[logical_not(Um)] #num des points doubles
    >>> W = list()
    >>> for i,j in zip(Ju[:-1],Ju[1:]) : #Intervales [i,j] i<j
    >>>     W.append(list(range(i,j)))
    >>> print(W)  # les listes d'indices des points multiples
    [[71, 220], [289, 299, 405]] #un point double, un point triple
    >>> for w in W : print (P[w].tolist())
    [[2.927, 0.274], [2.927, 0.274]]
    [[1.308, 0.442], [1.308, 0.442], [1.308, 0.442]]
    """
    j_sorted = np.lexsort(A.T)
    v = None
    for q in A.T:
        q = q[j_sorted] # q = le tableau X des x_i, idem Y, idem Z
#         print('q=',q.tolist())
        w = (np.abs(q[1:] - q[:-1])>eps)# liste de True ou False : w[i] vrai si q[i] == q[i+1]
#         print('w=',w.tolist())
        if v is None:
            v = w
        else:
            v |= w #v[i] = v[i] ou w[i]
#         print('v=',v.tolist())
    unique_mask = np.hstack([True, v])
    j_unique = j_sorted[unique_mask]
#     j_double = j_sorted[logical_not(unique_mask)]
    return j_sorted, unique_mask, j_unique

def pointsDoubles(P, eps=1.0e-3, complete=False):
    """
    :param P: ndarray((nbp,dim),dtype=float ou int) tableau de points contenant
        possiblement des points doubles
        Il vaut mieux que dim<<nbp, pour tri.
    :returns W : un set de points multiples.
        chaque point multiple est une liste de numéros des points identiques
    :TODO : on peut avoir W={[0,1],[1,2]} qu'il faudrait merger en W={[0,1,2]}
    """
    Js, Um, Ju = pointsUniques(P, eps=eps)
    npd = Js[logical_not(Um)] #num des points doubles
#     W = [tuple(where(norm(P-pd, axis=1)<=eps)[0]) for pd in P[npd]]
    W = list()
    for i,j in zip(Ju[:-1],Ju[1:]) : #Intervales [i,j] i<j
        W.append(list(range(i,j)))
    last = len(P)-1
    if last in Ju :
        W.append([last])
    else : #last est point double, il va dans le dernier cluster
        W[-1].append(last)
#     W = set(W)
    return (W, npd, Ju) if complete else W

def supprimerPoints(A, S, C=asarray([],dtype=int)):
    """
    ATTENTION, supprime également toutes les connexions de C qui contiennent un j∈S
    suppression dans A des points numeros j∈S et des connexions contenant j.
    :param A: ndarray((nbp,dim),dtype=float ou int) tableau de points a nettoyer.
        Il vaut mieux que dim<<nbp, pour tri.
    :param S: ndarray(n,dtype=int) numéros des points à supprimer max(S)∈[0, nbp[
        toutes les connexions faisant référence à un j∈S sont supprimées
    :param C: ndarray((n,coord),dtype=int) tableau de connexions entre les points
        de A, avec max(C)<nbp et coord=1,2,3 ou plus.

    :return A, C: le tableau de points modifié et les connexions mises à jour.

    """
    #On supprime les points j∈S
    As = delete(A, S, axis=0)
#     print('A =',A.tolist())
#     print('S =',S)
#     print('As =',As.tolist())
    A = As
    S = sorted(list(set(S)))
    #On supprime les connexions CONTENANT les points j∈S
    tokeep = []
    for k, c in enumerate(C) :
#         print ('(k,c) =', (k,c.tolist()))
        keep = True
        for i in c :
            if i in S :
                keep = False
                continue
#         print('keep=',keep)
        if keep : tokeep.append(k)
#     print('C=',C.tolist(), 'tokeep=',tokeep)
    C = C[tokeep]
    #Décalage des n° j∈C.flat() tq j>js, js∈S
    for js in S[::-1] :
        for c in C : c[where(c>js)] -= 1
    return A, C

def supprimerAutoConnexions(C):
    """
    :param C: ndarray((nc,coord),dtype=int), un tableau de connexions i.e. une
        liste de nc cellules [i_1,i_2,...,i_coord] coord est la coordinance,
        i.e le nb de points des cellules (line=>2, triangle=>3 etc...)
    une auto-connexion est une cellule [i,i,...,i]. On les supprime
    """
    tokeep = []
    for k,c in enumerate(C) :
        if not np.all(c==c[0]) :
            tokeep.append(k)
#     debug(tokeep=tokeep)
    return C[tokeep]

def supprimerPointsDoubles(A, C=asarray([], dtype=int), eps=1.0e-3):
    """
    suppression points doubles de A avec MISE À JOUR DE C, le cas échéant
    :param A: ndarray((nbp,dim),dtype=float ou int) tableau de points a nettoyer.
        Il vaut mieux que dim<<nbp, pour tri.
    :param C: ndarray((nc,coord),dtype=int) tableau de connexions entre les points
        de A, avec max(C.flat)<nbp. et coord=1,2,3,... est la coordinance,
        i.e. le nb de (n° de) points de chaque connexion.
    :returns A1, C1: le tableau de points et les connexions mises à jour.
    :Attention : A(?) et C sont modifiés in situ, si on veut les conserver,
        il faut en faire une copie avant l'appel à supprimerPointsDoubles()
    """
    j_sorted, unique_mask, j_unique = pointsUniques(A, eps)
#     debug(j_sorted=j_sorted, A=A.shape)
    if len(C)==0 :
        #Cas ou il n'y a pas de tableau de connexions C
        j_unique.sort()#On veut les points uniques, mais dans l'ordre initial
        return A[j_unique], []
    #Cas ou il y a un tableau de connexions C à traiter
    #les n° des points doubles de A, triés
    avirer = sorted(j_sorted[logical_not(unique_mask)])
#     debug(avirer=avirer)
    J = C.view()
    J.shape = -1#on fait de C un tableau a une seule dimension
#     avirer = sorted([i for i,v in zip(j_sorted, unique_mask) if not v])
#     newJ = np.arange(0, 1+np.max(J))#il peut y avoir des n° de points>max(J)
    newJ = np.arange(0, 1+max(j_sorted))
    #2. Nouveaux pointeurs sur points
    # Si je vire le point j, je le remplace par le point newJ[j]=ju
    # newJ ne contient que les (numéros de) points à garder.
    for k, (v, j) in enumerate(zip(unique_mask, j_sorted)) :
        if v :
            ju = j#v=True => on garde le point j
        else :
            newJ[j] = ju #le point j est remplacé par ju
    #3. Je supprime effectivement les n° de points avirer, les points agarder ont
    # un nouveau numéro que je met dans newJ
    # si je supprime le point j, tous les points k>j reculent d'une place k-=1
    for j in avirer[::-1] :
        #il faut partir du numero le +élevé
        #sinon les n° avirer doivent être aussi décalés de -1
        newJ[newJ>j] -= 1
    for k, j in enumerate(J) :
        J[k] = newJ[j]
    j_unique.sort()#On les veut dans l'ordre initial
    C,_ = supprimerPointsDoubles(C, eps=0)
    #C = supprimerAutoConnexions(C) => hors sujet
    #transforme les connexions (i,j) en (min(i,j), max(i,j))=> hors sujet
    #C.sort(axis=1)=> hors sujet
    return A[j_unique], C

def supprimerPointsDoubles1(A, eps=1.0e-3, newnum=True):
    """
    suppression points doubles de A, PAS DE MISE À JOUR DE CONNEXIONS
    Dans A soit un point multiple que l'on retrouve en position i1<i2<i3...
    Alors
    - A[i2], A[i3],... sont supprimés.
    - les points sont renumérotés la nouvelle numérotation est retournée dans newJ
        - tous les indices i2, i3,... sont transformés en i1
        - les indices de C sont mis à jour pour tenir compte du chgt de
            position des points d'indice >= i2
            (si on supprime le point i, les n° de points>=i sont décrémentés de 1)

    :param A: ndarray((nbp,dim),dtype=float ou int) tableau de points a nettoyer.
        Il vaut mieux que dim<<nbp, pour tri.
    :param eps: float, deux points sont considérés comme identiques si leur distance est <eps
    :param newnum : bool, True si l'on veut la nouvelle numérotation, False sinon
    :returns A1: le tableau de points mis à jour.
    :returns newJ: ndarray(nbp, dtype=int) nouvelle numérotation
        si J est un(e liste de) n° de point(s) de l'ancienne numérotation, alors
        newJ[J] est le (la liste des) n° de points dans la nouvelle numérotation
    """
    #1. calcul des points doubles
    j_sorted, unique_mask, j_unique = pointsUniques(A, eps)
    if not newnum :
        j_unique.sort()#On les veut dans l'ordre initial
        return A[j_unique]
    #
    #2. Nouvelle numérotation
    #les n° des points doubles de A, triés
    avirer = sorted(j_sorted[logical_not(unique_mask)])
    # Si je vire un point n°j, je le remplace par le point n° newJ[j]=ju
    newJ = np.arange(0, 1+max(j_sorted))#nouvelle numérotation
    for (v, j) in zip(unique_mask, j_sorted) :
        if v :
            ju = j#v=True => on garde le point j
        else :
            newJ[j] = ju #le point j est remplacé par ju
    #si je supprime le point j, tous les points k>j reculent d'une place k-=1
    for j in avirer[::-1] :
        #il faut partir du numero le +élevé
        #sinon les n° avirer doivent être aussi décalés de -1
        newJ[newJ>j] -= 1
    j_unique.sort()#On les veut dans l'ordre initial
    return A[j_unique], newJ

def locate(t, T, eps = 1.0e-5):
    """(2D et 3D)
    Localise t dans T=ndarray(n).
    On doit avoir T[0]<=t<=T[-1] et T croissant, len(T)>=2
    retourne (True_ou_False, index),
    - (True, k) si t=T[k] à eps près (t est une des valeurs T[k])
    - (False, k) si T[k] < t < T[k+1] (t n'est pas un T[k])
    """
    g = where(T<=t+eps)[0]
    d = where(T>=t-eps)[0]
    return (True, g[-1]) if g[-1]==d[0] else (False, g[-1])

def computeLongueurs(C, P):
    """
    Calcule et retourne les longueurs des connexions C, en référence aux points de P
    :param C: ndarray((nbc,coo),dtype=int) les nbc connexions. Chaque connexion
        est de coordinance coo (coo(line)=2, coo(triangle)=3, coo(quad)=4 ...) et
        contient les numéros [i, j, k...] des points de P connectés.
        On doit nécessairement avoir max(C.flat) < len(P)
    :param P: les points de R^dim en général dim = 1, 2 ou 3
        - ndarray((nbp,dim), dtype=float) les coord des points
        - ou bien tuple (len(P)=dim) de ndarray((nbp,1), dtype=float),
            en dim=3, P = (X,Y,Z)
    NON, L est TOUJOURS EN CREATION
        :param L: None ou ndarray((nbc,1),dtype=float), le tableau des longueurs
            à créer si L=None, à modifier sinon
    FIN NON
    :returns L:ndarray((nbc,1),dtype=float), les longueurs des connexions
    """
    if C.shape[1]!=2 : raise NotImplementedError('TODO, si besoin')
    if isinstance(P, tuple) : P = hstack(P) # pour P=(X,Y,Z)
    return norm(P[C[:,0]]-P[C[:,1]],axis=1,ord=2)

def computeLongueurContour(tab):
    """ Calcule la longueur d'un contour 2D ou 3D """
    npt = len(tab)
    lon = 0.0
    for i in range(npt-1):
        lon+=dist(tab[i],tab[i+1])
    return lon

def computeCordeAndNBA(points):
    """
    :param points:ndarray((n,dim), dtype=float)
    retourne la distance et le numéro du point le plus éloigné de BF=points[0]
    Dés que la distance décroit, on y est.
    """
    n = norm(points-points[0], axis=1, ord=2)
    i = argmax(n)
    return n[i],i

def dist2(p1,p2):
    """retourne le carré de la distance de p1 à p2 en norme n=2"""
    try :
        return sum(v**2 for v in p2-p1)
    except TypeError : # Si p1 ou p2=None
        return nan

def dist(p1,p2):
    """retourne la distance de p1 à p2 en norme n=2"""
    try :
        return math.sqrt(sum(v**2 for v in p2-p1))
    except TypeError : # Si p1 ou p2=None
        return nan

def prodScal(u,v):
    """ Produit scalaire des vecteurs u et v """
    return sum(ui*vi for ui,vi in zip(u,v))
    return sum([u[i]*v[i] for i in range(len(u))])
#     return u[0]*v[0] + u[1]*v[1] + u[2]*v[2]

def signe(x):
    eps = 1.0e-6
    if x < -eps:
        return -1
    else:
        return 1

def moyenneMobileClosed(points, molecule=None, nnodes=None):
    """
    Lissage moyenne mobile pour les polygones fermés self[-1]==self[0]
    nnodes est la liste des numeros de points à lisser. On suppose que 0 n'y est pas
    """
    if molecule is None:
        molecule = [1., 2., 1.]
    molecule=np.asarray(molecule)#np.asarray([1., 2., 1.])/4.
    molecule/=np.sum(np.absolute(molecule))
    new=points
    if nnodes is None :
        nnodes=list(range(len(points)))
    else :
        nnodes.sort()
        if nnodes[0]==0 and nnodes[-1]==len(points) :
            debug('non implémenté')
            return new
        else : pass
    old=new[:-1]
    n=len(old)
#         deb, fin = 0, n
    deb,fin=nnodes[0],nnodes[-1]
    for k in range(deb,fin) :
        pm=old[k-1]
        p=old[k]
        if k==n-1 :
            pp=old[0]
        else:
            pp=old[k+1]
        new[k]=molecule[0]*pm+molecule[1]*p+molecule[2]*pp
    new[-1]=new[0]
    return new

def moyenneMobile(X,n=1):#,molecule=[1.,2.,1.]):
    """
    Lissage moyenne mobile pour fonction simple i -> X[i]
    Y[i] = (X[i-1]+2X[i]+X[i+1])/4, 0<i<n
    Y[0] = (       2X[0]+X[1])/3, i=0
    Y[n] = (X[n-1]+2X[n]       )/3, i=n
    """
    Y = X.copy()
    for k in range(1, len(X)-1):
        Y[k] = 0.25*(X[k-1] + 2*X[k] + X[k+1])
    Y[0]  = (        2*X[0] + X[1])/3.0
    Y[-1] = (X[-2] + 2*X[-1]      )/3.0
    if n==1 :
        return Y
    else :
        return moyenneMobile(Y, n-1)

def moyenneMobile1(X, n=1):#,molecule=[1.,2.,1.]):
    """
    n lissages par moyenne mobile pour fonction simple i -> X[i]
    Y[i] = (X[i-1]+2X[i]+X[i+1])/4, 0<i<n
    Y[0] = X[0]                   , i=0
    Y[n] = X[n]                   , i=n
    """
    Y = X.copy()
    for k in range(1, len(X)-1):
        Y[k] = 0.25*(X[k-1] + 2*X[k] + X[k+1])
    Y[0]  = X[0]
    Y[-1] = X[-1]
    if n==1 :
        return Y
    else :
        return moyenneMobile1(Y, n-1)

def maintenant():
    """La date et l'heure formatées "human readable\""""
    return str(DT.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))

def mtime(path:Path, formated=False)->float:
        """retourne la date de modification de path sous forme float"""
        t = path.getmtime() if path.isfile() else -math.inf
        if not formated :
            return t
        else :
            return fmt(t)

def toDict(cles,valeurs):
    """
    - cles = "cle1 cle2 cle3 ..."
    - valeurs = "val1 val2 val3...", les valeurs sont des entiers ou des reels
    retourne un dictionnaire cle,valeurs
    dict(zip(cles,valeurs) devrait marcher aussi
    """
    d = {}
    for key,value in zip(cles.split(),valeurs.split()) :
        try: w=int(value)
        except ValueError : w=float(value)
        d[key.lower()]=w
    return d

def findAll(tag,lines,first=0,last=-1):
    """
    Retourne une liste triée des numeros de ligne de
    TOUTES les occurences de tag dans lines[first:last]
    """
#     debug(len(lines),tag=repr(tag), first_last=(first,last))
    n=first-1
    n0=findRel(tag,lines,n+1,last)
    if n0 is None : return ()
    N=[]
    while n0 is not None and n0>=0 :
        n=n0+n+1
        N.append(n)
        n0=findRel(tag,lines,n+1,last)

    return tuple(N)

def findAllLines(tag,lines,first=0,last=-1):
    """
    Comme findAll(), mais il faut que la ligne complete (nettoyée) soit égale à tag, pas seulement une partie de la ligne.
    Par exemple : line = 'TOTO_EST_CONTENT' ne match pas avec tag='TOTO'
    """
    liste=findAll(tag,lines,first,last)
    newlist=[]
    for n in liste :
        if lines[n].strip()==tag :
            newlist.append(n)
    return tuple(newlist)

def findRel(tag,lines,first=0,last=-1):
    """
    Cherche 'tag' dans lines[first:last] et retourne
        * le numero de ligne i-first de la premiere occurence trouvee,
          c'est à dire le numéro de ligne RELATIF : dans self.lines[0:]
        * None si le tag n'est pas trouvé
    """
    found=find(tag,lines,first,last)
    if found is None : return None
    else : return found-first

def find0(tag,lines,first=0,last=-1):
    """
    Cherche 'tag' dans lines[first:last] et retourne
        * le numero de ligne i-first de la premiere occurence trouvee,
          c'est à dire le numéro de ligne RELATIF : dans self.lines[0:]
        * None si le tag n'est pas trouvé
    """
    found=findRel(tag,lines,first=first,last=last)
    if found is not None : return  found+first
    else : return None

def find(tag,lines,first=0,last=-1):
    """
    Cherche 'tag' dans lines[first:last] et retourne
        * le numero de ligne i de la premiere occurence trouvee,
          c'est à dire le numéro de ligne ABSOLU : dans self.lines[0:]
        * None si le tag n'est pas trouvé
    """
    if last<0 : last=len(lines)+last
    if first is None : first=0
    elif first<0 : first=len(lines)+first
    if not tag : return None
    i=first-1
    while i<last:
        i+=1
        try :
            if lines[i].find(tag)>=0 :
                return i
        except IndexError :
            return None
    return None

def load(filinname):
    try :
        filin=open(filinname,'r')
        return pickle.load(filin)
    #essayer
    except IOError :
        msg = """Impossible d'ouvrir le fichier dump %s, pas de lecture.
    Essayer les instructions suivantes :
        >>> filin = open(filinname,'rb')
        >>> pickle.load(f,encoding='latin1')
        """%filinname
        print(msg, file=sys.stderr)

def diff(A, k=1):
    """
    J'en ai marre de taper tout le temps 'dA = A[1:]-A[:-1]'
    :param A: un tableau de points np.ndarray ou liste de shape (n, dim), dim>=1
    :param k: int, ordre de différenciation.
    :return dA: ndarray((n-k,dim)), les différences k-iemes de A:
        dA[i] = A[i+1]-A[i] si k=1
        diff(A,k) = diff(diff(...(diff(A))...) k fois
    P.ex. diff(A,2) équivaut à diff(diff(A)).
    >>> A = asarray([1,2,3,4,10])
    >>> diff(A)
    [1 1 1 6]
    >>> diff(A,2)
    [0 0 5]
    >>> diff(diff(A))
    [0 0 5]
    """
    A = asarray(A)
    while k :
        A = A[1:]-A[:-1]
        k -= 1
    return A

def absCurv(X, normalise=False):
    """Abscisse curviligne des points de X (2d/3d)
    :param X: ndarray((n,dim))
    :param normalise: bool, True si l'abs. curv. est ramenée à [0, 1]"""
    l = len(X)
    if l==0 : return []
    elif l==1 : return [0]
#     T = norm(diff(X), axis=0)=> NOOOON !
    T = zeros(l)
    for k in range(l-1) :
        T[k+1] = T[k] + norm(X[k]-X[k+1])
    if normalise and T[-1] != 0.0 :
        T /= T[-1]
        #sinon T=[0,0,...] On peut très bien entrer ici avec 3 points identiques
    return T

def pointsDoublesConsecutifs(points, eps=1.0e-10, ordre=2):
    """Calcule uniquement les (n° de) points doubles consécutifs
    Cf fonction pointsUniques()"""
    if len(points.shape) == 1 :#shape = (n,) marche pas. il faut (n,1)
        points = points.view()
        points.shape = (-1,1)
    return where(norm(diff(points),ord=ordre, axis=1)<=eps)[0].tolist()

def supprimerPointsDoublesConsecutifs(points, eps=0.0, vires=False, ordre=2):
    """a refaire en utilisant pointsUniques()"""
    avirer = pointsDoublesConsecutifs(points, eps, ordre=ordre)
    if avirer : points = delete(points, avirer, 0)
    return (points, avirer) if vires else points

def safeRemoveTree(rep:Path) :
    if rep.isdir() :
        contents = sorted(os.listdir(rep))
    elif rep.isfile() :
        contents = [rep]
    if contents :
        debug (paragraphe="Suppression de %s. Contenu :"%rep.name)
        for c in contents :
            c = Path(c)
            if c.isdir() : print ('%30s (D)'%c.name)
            elif c.isfile() : print ('%30s (F)'%c.name)
            else :  print ('%30s (?)'%c.name)
    else :
        debug(paragraphe="%s est vide"%rep.name)
    ans = 'w'
    while ans not in ('y','n','') :
        ans = input('\nOK pour suppression ? y/n (defaut : n)').lower()
    if ans == 'y' :
        rdebug("JE REMOVE %s"%rep)
        return shutil.rmtree(rep)
    else :
        rdebug("CANCELLED REMOVE %s"%rep)
        return None

def nth(iterable, n, default=None):
    """Returns the nth item or a default value
    cf https://docs.python.org/fr/3.8/library/itertools.html#itertools.zip_longest"""
    return next(islice(iterable, n, None), default)

def perfo(func):#where='', mindelta=0.001):
    u"""
    last minute, mieux (?) : le package codetiming
    ------------------
    https://realpython.com/python-timer/#a-python-timer-decorator

    Simplification de ce qui suit :
    -----------------------------
    cf http://gillesfabio.com/blog/2010/12/16/python-et-les-decorateurs/
    def decorate(arg1, arg2, arg3):
        def decorated(func):
            def wrapper(*args, **kwargs):
                # Pré-traitement
                response = func(*args, **kwargs)
                # Post-traitement
                return response
            return wrapper
        return decorated
    utilisation :
    >>> @perfo
    >>> def toto(args) :
    >>>     print(args)
    ------------
    simplifié en :
    ------------
    def decorate(func):
        def wrapper(*args, **kwargs):
            # Pré-traitement
            response = func(*args, **kwargs)
            # Post-traitement
            return response
        wrapper.__doc__ = func.__doc__
        wrapper.__name__ = func.__name__
        return wrapper
    """
    def wrapper(*args, **kwargs):
        w  = func.__name__
        date_deb = DT.datetime.now()
        response = func(*args, **kwargs)
        delta = (DT.datetime.now()-date_deb).total_seconds()
#             if delta>mindelta :
        print ('|PERF| %-20s : '%w, delta)
        return response
    return wrapper
#     return decorated
#########################################################
# des fonctions pour transformer x en mega, giga, ...
#  et les écrire au format tableur (français)
#########################################################
def one(x) :
    return ('%f' % x).replace('.',',')

def kilo(x):
    return ('%f' % (x / 1000)).replace('.',',')

def mega(x):
    return ('%f' % (x / 1000000)).replace('.',',')

def giga(x):
    return ('%f' % (x / 1000000000)).replace('.',',')

def tera(x):
    return ('%f' % (x / 1000000000000)).replace('.',',')

def localGetter(fpath: Path):
    """Utilisé dans les oracles en mode sandbox,
    pour simuler un acces au serveur sans dépenser d'unités de compte"""
    # debug('localGetter', fpath=fpath)
    # mexit()
    try:
        if fpath.ext == '.csv':
            dump = pandas.read_csv(fpath)
        elif fpath.ext == '.pyd':
            with open(fpath, 'r') as f:
                pydata = f.read()
            # debug(pydata=pydata)
            dump = eval(pydata)
        elif fpath.ext == '.txt':
            with open(fpath, 'r') as f:
                text = f.read()
            dump = text
        else:
            raise NotImplementedError("le fichier %s a une extension (%r) non prise en charge" \
                                      % (fpath, fpath.ext))
    except SyntaxError as msg:
        # fichier cache peut être vide ou malformé, on l'efface
        # fpath.remove()
        print('%s : fichier %s malformé, à supprimer' % (msg, fpath))
    return dump

Timestamp = Union[DT.datetime, DT.date, int, float]
# def f(x:Timestamp) declarer x de type date au sens large,
# pour transformer des datetime ou date en timestamp:int (horodatage en francais)
# t = DT.datetime   => t.timestamp()
# t = DT.date       => t.timestamp()
# t = float         => int(t)
# t = int           => t
# Pour transformer des int, float en datetime ou date:
# t = int ou float  => DT.datetime.fromtimestamp(t,tz=DT.tzinfo)
# t = int ou float  => DT.date.fromtimestamp(t)
# voir dans binance.helpers la fonction date_to_milliseconds()

def btime(ms:int) :
    """Binance time => readable time"""
    return '%s'%DT.datetime.fromtimestamp(ms/1000)

def verifyDates(deb:Union[str, int, float, DT.datetime, DT.date, DT.time, None],
                fin:Union[str, int, float, DT.datetime, DT.date, DT.time, None]):
    """
    On vérifie que :
        - deb<=fin
        - deb et fin dans [origine, utcnow()], avec origine=1970-01-01 à 01:00:00 (timestamp=0)
    Si tout est OK, retourne (deb, fin) sous forme de (timestamp, timestamp) en (secondes, secondes)
    """
    if deb is None :
        #La date ou j'ai commencé à acheter des cryptos
        deb="2021-04-01"
    deb = verifyOneDate(deb)
    if fin is None :
        fin = 'now utc'
    fin = verifyOneDate(fin)
    if deb >= fin:
        raise ValueError("Dates invalides : on devrait avoir debut(%s) < fin(%s)" \
                         % (str(DT.date.fromtimestamp(deb)),
                            str(DT.date.fromtimestamp(fin))))
    return deb, fin #en secondes

def verifyOneDate(date: Union[None, str, int, float, DT.datetime, DT.date, DT.time]):
    """
    Verifie que date est entre
        - l'origine des temps,  i.e. DT.datetime.fromtimestamp(0) => 1970-01-01 à 01:00:00
        - et maintenant,        i.e. DT.datetime.now(tz=utc).timestamp()
    :param date: str, datetime, date, time, int ou float
    :returns la date en secondes
    """
    if date is None :
        date = datetime(year=2017, month=12, day=11)
    if isinstance(date, str):
        date = date_to_milliseconds(date)/1000 #=> date en secondes
    elif isinstance(date, (DT.datetime, DT.date, DT.time)):
        date = date_to_milliseconds(str(date))/1000 #=> date en secondes
    elif isinstance(date, (int, float)):
        # date doit être un timestamp en secondes, on n'y touche pas
        pass
    else:
        raise ValueError("date <%s> non pris en charge" % date.__class__.__name__)

    if not 0 <= date < DT.datetime.now(tz=utc).timestamp():
        # date doit etre compris entre 0.0 et now().timestamp() secondes
        raise ValueError("Date invalide : on devrait avoir (min)%s < %s <= %s(max)" \
                         % (DT.date.fromtimestamp(0),
                            DT.date.fromtimestamp(date),
                            DT.date.today()))
    return int(date) #=> en secondes

def strFromDate(dt: str or int or float or DT.datetime or DT.date) -> str:
    fmt = '%Y-%m-%d' if isinstance(dt,DT.date) else '%Y-%m-%d %H:%M'
    return parse(str(dt)).strftime(fmt)

def strToDate(date_str: str) -> DT.datetime:
    """Converti une date UTC en secondes depuis le temps 0
    If using offset strings add "UTC" to date string e.g. "now UTC", "11 hours ago UTC"
    See dateparse docs for formats http://dateparser.readthedocs.io/en/latest/
    :param date_str: date in readable format, i.e. "January 01, 2018", "11 hours ago UTC", "now UTC"
    """
    # get epoch value in UTC
    epoch: datetime = datetime.utcfromtimestamp(0).replace(tzinfo=pytz.utc)
    # parse our date string
    d: Optional[datetime] = dateparser.parse(date_str, settings={'TIMEZONE': "UTC"})
    # if the date is not timezone aware apply UTC timezone
    if d.tzinfo is None or d.tzinfo.utcoffset(d) is None:
        d = d.replace(tzinfo=pytz.utc)
    seconds = int((d - epoch).total_seconds())
    return DT.datetime.fromtimestamp(seconds)

    # return the difference in time
    # return int((d - epoch).total_seconds() * 1000.0)


if __name__=="__main__":
    if 0 :
        debug(yellow('str_to_date("2021-11-22 08:05:38")'))
        d = strToDate("2021-11-22 08:05:38")
        print(pink(str(d)))
        s = strFromDate(d)
        print(pink(s))
        mexit()
    if 1 :
        debug(titre='verifyOneDate()')
        d = verifyOneDate('2021-12-02')
        print(DT.datetime.fromtimestamp(d,tz=utc))
        d = verifyOneDate(None)
        print(DT.datetime.fromtimestamp(d, tz=utc))
        d = verifyOneDate('1953-12-02')
        mexit()
    debug('MyConfig()')
    conf = MyConfig()
    mexit()
    A = asarray([])
    print(A[1:])
    print(diff(A,3))
    A = asarray([1,2])
    print(diff(A))
    A = asarray([1,2,3,4,10])
    print(diff(A))
    print(diff(A,2))
    print(diff(diff(A)))
    mexit()
    dx = 0.6541
    X = asarray([random() for _ in range(10)])*10
    X = fmt(linspace(0,10,50))


