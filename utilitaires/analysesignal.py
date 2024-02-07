#!/usr/bin/python
#coding:utf-8
#Pierre Puiseux, 14 décembre 2019
# import sys
# from typing import Any, Union

from numpy import zeros, linspace, absolute, arange, asarray
# import scipy.signal.convolve as convolve
from pandas import Timestamp, Series, DataFrame, Period
from pandas import DataFrame
# from pandas.io.json._json import JsonReader

from utils import diff
from debog import mexit, debug, className
from scipy import signal, fft
from math import sqrt, exp,log
import json, os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np  # noqa
import pandas as pd  # noqa
# from pandas import DataFrame

# from freqtrade.constants import DEFAULT_CONFIG
# from freqtrade.data.dataprovider import DataProvider
# from freqtrade.misc import file_load_json, json_load
# from freqtrade.resolvers import ExchangeResolver
# from freqtrade.strategy import IStrategy

# --------------------------------
# Add your lib to import here
# import talib.abstract as ta
# import freqtrade.vendor.qtpylib.indicators as qtpylib

"""
Pour filtrer un signal et plein d'autres choses : a lire absolument
http://www.f-legrand.fr/scidoc/docimg/numerique/filtre/introfiltres/introfiltres.html
"""
filter_ = signal.convolve

"""
1 Lire le signal,
2 régler le paramètre dt, le pas de temps d'echantillonnage (constant)
Le signal S est une somme S = THF + HF + BF
    THF=bruit, très hautes fréquences
    BF=basses fréquences,
    HF=hautes fréquences

3 filtrer le bruit (très hautes fréquences), pour cela régler le paramètre nbpg, nb de points de convolution.
    Il faut que 2*P+1 points couvrent les parasites, mais pas plus. Tester plusieurs fois
4 régler la fenètre de temps
5 Extraire les BF : elle sont obtenues par lissage du signal, avec un filtre gaussien.
    Régler ce filtre avec nbpg, a nouveau

That's all
"""
def filtreGaussien(nbpg, epsilon):
    #filtre gaussien, largeur 2*nbpg+1
    #filtre passe bas
    sigma=nbpg/sqrt(-2.0*log(epsilon))
    som = 0.0
    b_gauss = zeros(2*nbpg+1)
    for k in range(2*nbpg+1):
        b_gauss[k] = exp(-(k-nbpg)**2/(2*sigma**2))
        som += b_gauss[k]
    return b_gauss/som

def seconds(dt:np.timedelta64) -> float :
    """Conversion d'une durée dt (np.timedelta64) en secondes"""
    return dt/np.timedelta64(1,'s')

def decomposer(prix, dates, nbpg, eps):
    """
    Pour le calage des paramètres sur un exemple (oscillations parapente test structure),
    voir analysesignal0.py
    """
    prix = prix
    T = dates
    dt = seconds(diff(T)[0])
    print('dt=',dt)
    # print('prix=',prix.shape)
    #########################################
    ## suppression très hautes fréquences
    # On bouffe nbpg*dt secondes à chaque bout
    #########################################
    plt.plot(T[nbpg:-nbpg], prix[nbpg:-nbpg], label='original')
    # moyenne sur dt*(2*nbpg+1) secondes => 0.5 secondes si nbpg=2
    gaussien = filtreGaussien(nbpg=nbpg, epsilon=eps)
    prix = filter_(prix, gaussien, mode='valid') #prix est de la bonne longueur
    plt.plot(T[nbpg:-nbpg], prix, label='filtré (nbpg=%d, eps=%.1g)' % (nbpg, eps))
    plt.legend()
    plt.show()
    # sys.exit()

    ##prix est le signal de base, filtré, débarrassé des très hautes fréquences
    N = len(prix)
    # dt = 60         #pulsation en sec
    # freq = 1 / dt   # frequence = 10Hz
    T00 = linspace(0, N * dt, N)
    T0, T1 = T00[0], T00[-1]
    ###########################################
    ##Fenetre de temps
    # fenetre = (t0, t1) = (75, 85)  # secondes
    (t0, t1) = 0, N*dt
    ###########################################
    n0, n1 = int(round(t0 / dt, 0)), int(round(t1 / dt, 0))

    prix0 = prix[n0:1 + n1]  # les data à analyser
    T = T00[n0:1 + n1]
    print('\nN, prix0.shape, n0,n1', str((N, prix0.shape, n0, n1)))
    #############################################################
    ## décomposition du signal filtré
    ## par extraction de basses fréquences avec le filtre gaussien
    ##############################################################
    P0 = prix0  # .copy()
    # nbpg = 15
    gaussien = filtreGaussien(nbpg=nbpg, epsilon=eps)
    Pb = filter_(P0, gaussien, mode='valid')
    ##Les hautes fréquences Ph sont obtenues par soustraction des basses fréquences au signal d'origine :
    # Ph = P0-Pb
    ###
    Ph = prix0[nbpg:-nbpg] - Pb  # Ph = Les hautes frequences
    T = T[nbpg:-nbpg]
    Ne = len(Ph)  # nb pts echantillon
    ############################################################
    ## calcul du spectre HF et BF
    ############################################################
    hspectre = absolute(fft(Ph)) / Ne
    bspectre = absolute(fft(Pb)) / Ne
    frequences = arange(Ne, ) / (dt * Ne)

    ############################################################
    ## tracé
    ############################################################
    fig, ((sbax, shax), (fbax, fhax)) = plt.subplots(2, 2)
    sbax.set_title("Basses fréquences")
    sbax.grid(True)
    sbax.plot(T, Pb)
    sbax.set_xlabel("temps (s)")
    sbax.set_ylabel("Prix (€)")
    fbax.plot(frequences, bspectre)
    fbax.set_xlabel("f (Hz)")
    fbax.set_ylabel("A")
    fbax.grid(True)

    shax.set_title("Hautes fréquences")
    shax.grid(True)
    shax.plot(T, Ph)
    shax.set_xlabel("temps (s)")
    shax.set_ylabel("Prix (€)")
    fhax.plot(frequences, hspectre)
    fhax.set_xlabel("f (Hz)")
    fhax.set_ylabel("A")
    fhax.grid(True)
    plt.show()


if __name__=='__main__' :
    #############################################
    ##     Lecture et création DataFrame
    #############################################
    datadir = Path("/Users/puiseux/GitHub/freqtrade/user_data/data/binance")
    frame = pd.read_json(datadir/ 'BTC_EUR-1m.json')
    # Transformation en pandas.DataFrame
    dates = [Timestamp(d, unit='ms') for d in frame[0]]
    p0 = Period(dates[0], freq=dates[1]-dates[0])
    print(p0, p0.freqstr,p0.end_time)
    print (p0+1)
    mexit()
    # t0 = dates[0]
    # dt = seconds(dates[1]-dates[0])
    dt = dates[1]-dates[0]
    dates = asarray(dates)
    dT = diff(dates)
    # print(Period(dt) for dt in dT)
    plt.plot([dt.seconds/60 for dt in dT])
    p = p0
    for k, t in enumerate(dates) :
        p = p0 + k
        if dt.second!=60 :
            print('Date ', dates[k], ', pas de cotation pendant %d secondes.' % dt.seconds)
    plt.show()
    print(dT)
    mexit()
    # dT = Series([(d-dates[0]).total_seconds()/60 for d in dates])
    print(dt,'dt = %d(s)'%dt.seconds)
    print(dates[0]," => ",dates[-1], "(%s)"%(dates[-1]-dates[0]).seconds)
    #Chandelle : open      high       low     close     volume
    frame = DataFrame({'date':dates,'open':frame[1],'high':frame[2], 'low':frame[3], 'close':frame[4], 'volume':frame[5]})
    #prix moyen chandelle
    tprice = 0.25*(frame['open']+frame['high']+frame['low']+frame['close'])
    frame['tprice'] = tprice
    if 0 : #filtrage TODO : caler nbpg et eps suivant le dt et la fenetre [t0,tn]
        fenetre = slice(100,200)
        decomposer(prix=tprice[fenetre], dates=frame['date'][fenetre], nbpg=5, eps=0.01)
    # fenetre = (dates[0],dates[-1])
    X, T0 = tprice, Series([(d-dates[0]).total_seconds()/60 for d in frame['date']])
    print(len(T0), min(T0), max(T0))
    # mexit()
    dX, T1 = diff(X), T0[:-1]
    print(len(T1))
    # k = 1
    d2X, T2 = diff(X,2), T0[1:-1]
    print (len(T2))
    m, l = 100, len(X)
    plt.figure()
    plt.plot(dX[:-1],d2X,'r.',ms=0.1, )
    plt.xlabel('dX')
    plt.ylabel('$d^2X$')
    plt.axis([-m, m, -m, m])
    plt.title('$(dX,d^2X)$')

    plt.figure()
    plt.plot(X[:-1],dX,'b.',ms=0.1)
    plt.xlabel('X')
    plt.ylabel('$dX$')
    plt.axis([min(X),max(X),-m,m])
    plt.title('$(X,dX)$')

    plt.figure()
    plt.plot(X[1:-1],d2X,'g.',ms=0.1)
    plt.xlabel('X')
    plt.ylabel('$d^2X$')
    plt.axis([min(X),max(X),-m, m])
    plt.title('$(X,d^2X)$')

    plt.figure()
    plt.plot(T0, X,'b.',ms=0.1)
    plt.xlabel('t (mn)')
    plt.ylabel('$X(t)$')
    plt.title('$Y=X(t)$')

    plt.figure()
    plt.plot(T1, dX,'b.',ms=0.1)
    # plt.axis([0, len(X), -m, m])
    plt.xlabel('t (mn)')
    plt.ylabel('$dX(t)$')
    plt.title('$Y=dX(t)$')


    plt.figure()
    plt.plot(T2, d2X,'b.',ms=0.1)
    # plt.axis([, -m, m])
    plt.xlabel('t (mn)')
    plt.ylabel('$d^2X(t)$')
    plt.title('$Y=d^2X(t)$')

    plt.show()
    # plt.axis([0, len(X), -m, m])
    #1 lire data
    # print(os.getcwd())
    # home = Path(os.getcwd()).parent.parent
    # confile = home / DEFAULT_CONFIG
    # confile = Path('/Users/puiseux/GitHub/freqtrade') / DEFAULT_CONFIG
    # print (confile)
    # exit
    # with open(confile, 'rt') as f :
    # cfgdict=json.load(f)

    # print(type(cfgdict))
    # exit()
    # cfgdict['datadir'] = home / 'user_data' / 'data' / 'binance'
    # exchange = ExchangeResolver.load_exchange(cfgdict['exchange']['name'], cfgdict)
    # dp = DataProvider(config=cfgdict,exchange=exchange)
    # print(dp.available_pairs)
    # frame = dp.get_pair_dataframe(pair="BTC/EUR")
    # datadir = Path("/Users/puiseux/GitHub/freqtrade/user_data/data/binance")
    # frame = pd.read_json(datadir/ 'BTC_EUR-1m.json')
    # print(type(frame))
    # print('frame', frame.describe())
    # print ("frame.dtypes",frame.dtypes)
    # print("frame.axes",frame.axes)
    # print(frame.head())
    # debug(un=frame.shape, unun=frame[0])

    # print("tprice.shape", tprice)
    # mexit()
    # frame.plot(x='date',y='tprice', title='BTC_EUR-1m')
    # plt.show()
    # exit()
    #2 tracer typical_price
    # date = frame['date']
    # tprice = qtpylib.typical_price(frame) #panda.Series
    # frame['tprice'] = tprice
    # print('tprice =', tprice.head(10))
    # T = frame['date']
    # print('date =', T.head(10))
    # t = T[0]
    # if 0 :
    #     for t in T[:10]:
    #         print(t)
    #     for s,t in zip(T[:10],T[1:11]):
    #         dt = t-s
    #         print(dt, dt.seconds, dt.value)
    #         print(type(t.freq))
    # # frame.plot(x='date',y='tprice')
    # plt.show()
    # decomposer()
