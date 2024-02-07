#!/usr/bin/python
#coding:utf-8
#Pierre Puiseux, 14 décembre 2019
import pandas
from numpy import zeros, linspace, absolute, arange
# import scipy.signal.convolve as convolve
from matplotlib import pyplot as plt
from pathlib import Path

from pandas import DataFrame
from scipy import signal, fft
from math import sqrt, exp,log

from debog import mexit

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

3 filtrer le bruit (très hautes fréquences), pour cela régler le paramètre P_gauss, nb de points de convolution.
    Il faut que 2*P+1 points couvrent les parasites, mais pas plus. Tester plusieurs fois
4 régler la fenètre de temps
5 Extraire les BF : elle sont obtenues par lissage du signal, avec un filtre gaussien.
    Régler ce filtre avec P_gauss, a nouveau

That's all
"""
def filtreGaussien(P_gauss, epsilon):
    #filtre gaussien, largeur 2*P_gauss+1
    #filtre passe bas
    sigma=P_gauss/sqrt(-2.0*log(epsilon))
    som = 0.0
    b_gauss = zeros(2*P_gauss+1)
    for k in range(2*P_gauss+1):
        b_gauss[k] = exp(-(k-P_gauss)**2/(2*sigma**2))
        som += b_gauss[k]
    return b_gauss/som
if 1 :
    fin = Path('/Users/puiseux/GitHub/crypto/data/ohlc/Coinbase-BTC-EUR-1d.csv')
    df = pandas.read_csv(fin)
    print(df)
    df['avg'] = 0.25*(df['open']+df['close']+df['low']+df['high'])
    df['open'].plot()
    df['avg'].plot()
    df['close'].plot()
    plt.show()
    mexit()
###Lecture data
fin = Path('/Users/puiseux/Google Drive/Aerotest-FFVL/Aerotest/NervuresConfidentiel/2016-08-20-Stantick2.txt')
with open(fin) as f :
    lines = f.readlines()
# print (lines)
# print (len(lines))
R00 = zeros(len(lines),float)
for k,line in enumerate(lines) :
    # print (line)
    if line[0] == '#':continue
    try :
        w1,w2 = line.split(';')
        R00[k] = float(w1) + float(w2)
    except ValueError :
        # print(line)
        pass
##Fin lecture
#########################################
## suppression très hautes fréquences
#On bouffe 10*dt=1 seconde à chaque bout
#########################################
P_gauss = 3
plt.plot(R00[P_gauss:-P_gauss], label='original')
#moyenne sur dt*(2*P_gauss+1) secondes => 0.5 secondes si P_gauss=2
gaussien = filtreGaussien(P_gauss=P_gauss, epsilon=0.01)
R00 = filter_(R00, gaussien, mode='valid')#[P_gauss:-P_gauss]
plt.plot(R00, label='filtré (gauss(%d)'%P_gauss)
plt.legend()
# plt.show()
# exit()

##R00 est le signal de base, filtré
N = len(R00)
dt = 0.1#secondes, pulsation 0.1 s
freq = 1/dt #frequence = 10Hz
T00 = linspace(0, N*dt,N)
T0, T1 = T00[0], T00[-1]
###########################################
##Fenetre de temps
fenetre = (t0, t1) = (75,85)#secondes
###########################################
n0, n1 = int(round(t0/dt,0)), int(round(t1/dt,0))
R0 = R00[n0:1+n1]#les data à analyser
T = T00[n0:1+n1]
print('\nN, R0.shape, n0,n1', str((N, R0.shape, n0,n1)))
#############################################################
## décomposition du signal filtré
## par extraction de basses fréquences avec le filtre gaussien
##############################################################
R = R0#.copy()
P_gauss = 15
gaussien = filtreGaussien(P_gauss=P_gauss, epsilon=0.01)
Rb = filter_(R, gaussien, mode='valid')
##Les hautes fréquences Rh sont obtenues par soustraction des basses
###fréquences au signal d'origine : Rh = R-Rb
Rh = R0[P_gauss:-P_gauss]-Rb#Rh = Les hautes frequences
T = T[P_gauss:-P_gauss]
Ne = len(Rh)#nb pts echantillon
############################################################
## calcul du spectre HF et BF
############################################################
hspectre = absolute(fft.fft(Rh))/Ne
bspectre = absolute(fft.fft(Rb))/Ne
frequences = arange(Ne,)/(dt*Ne)

############################################################
## tracé
############################################################
fig, ((sbax,shax),(fbax,fhax)) = plt.subplots(2, 2)
sbax.set_title("Basses fréquences")
sbax.grid(True)
sbax.plot(T,Rb)
sbax.set_xlabel("temps (s)")
sbax.set_ylabel("Force (N)")
fbax.plot(frequences, bspectre)
fbax.set_xlabel("f (Hz)")
fbax.set_ylabel("A")
fbax.grid(True)

shax.set_title("Hautes fréquences")
shax.grid(True)
shax.plot(T,Rh)
shax.set_xlabel("temps (s)")
shax.set_ylabel("Force (N)")
fhax.plot(frequences, hspectre)
fhax.set_xlabel("f (Hz)")
fhax.set_ylabel("A")
fhax.grid(True)
plt.show()
