#!/usr/local/bin/python
#-*-coding: utf-8 -*-
"""
Created on 9 juin 2018

@author: puiseux
"""
import sys
from pathlib import Path
import time

"""
Quelques fonctions de débogage qui ont évolué au cours du temps.

Ecriture dans le fichier log 'AXile.log'
----------------------------------------

1- trace et alert

    >>> trace(x, y=yzt) écrit dans le fichier log niveau INFO
    >>> alert(x, y=yzt) écrit dans le fichier log niveau WARNING

    par exemple supposant que nums est une variable valant (1,2,3)
    pour l'appel suivant,
    >>> trace('voila:', numeros=nums)
    la sortie ressemble  à

    GuiCloison::backup [glideobject.py, line 327]  :
        voila ; numeros = (1,2,3)
    où
    - GuiCloison::backup sont la classe et la méthode, d'où est appelé trace (alert)
    - [glideobject.py, line 327], sont le nom du fichier, le numéro de ligne,
    -     voila ; numeros = (1,2,3) sont les variables

2- strace et salert

    >>> strace(x, y='toto') écrit dans le fichier log niveau INFO
    >>> salert(x, y='toto') écrit dans le fichier log niveau WARNING

    la sortie est analogue à trace et alert, un peu remaniée,
    sans le nom de classe et ressemble à ceci :

    [glideobject.py:327 (backup)] valeur_de_x ; y='toto'

    où
        - glideobject.py:327 sont le nom du fichier, le numéro de ligne,
        - (backup) est la fonction d'où est appelé strace (salert)

Ecriture sur la console
-----------------------

1-debug, rdebug

    >>> debug(x, y='toto', z='tutu')
        fait la même chose que strace, sur la console stdout
    >>> rdebug(x, y='toto', z='tutu')
        fait la même chose que salert, sur la console stderr (rouge dans eclipse)

2- stack et rstack

    >>> stack(commentaire)
        écrit le commentaire et la pile des appels sur stdout
    >>> rstack(commentaire)
        écrit le commentaire et la pile des appels sur stderr (rouge dans eclipse)

    la sortie ressemble à ceci :

    ========================
    === Pile des appels ====

        le commentaire

        la pile des appels
        ...

    ======  Fin pile  ======
    ========================

"""
# def phrase():
# #     try :
#     f=open(DATA_DIR'dico'))
# #     except IOError:
# #         return
#     words=qui,quoi,comment=[],[],[]
#     n=0
#     for line in f.readlines():
#         line=line.strip()
#         if line : words[n].append(line)
#         else : n+=1
#     return ' '.join((3*'*',choice(qui),choice(quoi),choice(comment)))

def mexit(msg: str = '') -> None:
    frame = sys._getframe(1)
    fcode = frame.f_code
#     fonction = fcode.co_name
    filename = Path(fcode.co_filename)#.name
    toclick = clickableLink(filename, frame.f_lineno)
    print("%s\nReprendre ici\n"%msg, toclick)#, file=sys.stderr)
    time.sleep(2)
    sys.exit()

def souligne(msg, c='-', shift=0):
    sl = shift*' '+len(msg)*c
    return shift*' '+msg+'\n'+sl

def sursouligne(msg, c='-', shift=0):
    sl = shift*' '+len(msg)*c
    return sl+'\n'+shift*' ' + msg+'\n'+sl

def trace(*args,**kargs):
    return _trace(sys.stdout, *args, **kargs)

def alert(*args,**kargs):
    return _trace(sys.stderr, *args, **kargs)

def _trace(output, *args,**kargs) :
    try : val = args[0] in (None,'')
    except Exception : val = False
    if val : msg = ''
    else:
        try : msg = args[0].__class__.__name__+'::'
        except AttributeError : msg = ''
    frame = sys._getframe(2)
    fcode = frame.f_code
    filename = Path(fcode.co_filename).name
    msg0 = '\n'+str(msg + fcode.co_name+' [%s, line %d] '%(filename, frame.f_lineno))+' : \n    '
    lmsg = [str(arg) for arg in args[1:]]+\
           [str(key)+' = '+str(value) for key,value in kargs.items()]
#     if output is sys.stdout :
    print(msg0 + ' ; '.join(lmsg), file=output)
    #     logger.info(msg0 + u' ; '.join(lmsg))
#     elif output is sys.stderr :
    #     logger.warning(msg0+u' ; '.join(lmsg))
#         print(msg0 + ' ; '.join(lmsg), file=sys.stderr)

def rstack(commentaire=''):
    _stack(sys.stderr, commentaire)
def stack(commentaire='') :
    _stack(sys.stdout, commentaire)

def _stack(output, commentaire=''):
    """Impression de la pile des appels sur output"""
#     print>>output,u"========================"
#     print>>output,u"=== Pile des appels ===="
    print(40*"=", file=output)
    print(10*"="+"   Pile des appels  "+10*"=", file=output)
    if commentaire :
        print(commentaire, file=output)
    k = 2
    while 1 :
        try :
            frame = sys._getframe(k)
            fcode = frame.f_code
            fonction = fcode.co_name
            filename = Path(fcode.co_filename)#.name
            toclick = clickableLink(filename, frame.f_lineno)#, fcode.co_name)
            print('    [%-20s] '%fonction + toclick, file=output)
            k += 1
#             filename = frame.f_code.co_filename).name
#             print>>output, u" ['%s':%d, (%s)] "%(filename, frame.f_lineno, frame.f_code.co_name)
        except :
            break
    print(10*"="+"      Fin pile      "+10*"=", file=output)
    print(40*"=", file=output)
    return

def clickableLink(filename, lineno=0):#, name):
    return str('File "%s", line %d'%(filename, lineno))#, name))

def _strace(*args,**kargs) :
    """_trace simplifié, sans le nom de la classe"""
    frame = sys._getframe(2)
    fcode = frame.f_code
    fonction = fcode.co_name
    filename = Path(fcode.co_filename)#.name
    if 0:#TEST_MODE :
        pass
#         toclick = "(module %s)"%name???
    else :
        toclick = clickableLink(filename, frame.f_lineno)
    output = args[0]
    args = args[1:]
    try :
        titre = sursouligne(kargs.pop('titre'),'=')
    except KeyError :
        titre = None
    try :
        paragraphe = souligne(kargs.pop('paragraphe'),'-',shift=4)
    except KeyError :
        paragraphe = None

    lmsg = [str(arg) for arg in args]+\
           [str(key)+" = "+str(value) for key,value in kargs.items()]
    msg = '[%s] '%fonction + toclick
    _, l, s = '    ', '\n', ' ; '#blanc, saut de ligne, separateur
    if titre and paragraphe :
        msg = msg + 2*l + titre + l + paragraphe
    elif paragraphe :#pas de titre
        msg = msg + 2*l + paragraphe
    elif titre :
        msg = msg + l + titre + l
    msg = msg + l + s.join(lmsg)
    if 0:#output in (logger.info, logger.warning):
        output(msg)
    else :
        print(msg, file=output)

def debug(*args,**kargs):
    """trace simplifié, stdout, identique à strace"""
    _strace(sys.stdout, *args, **kargs)
#     _strace(logger.info, *args, **kargs)

def rdebug(*args, **kargs):
    """trace simplifié, logger.warning, identique à salert"""
    _strace(sys.stderr, *args, **kargs)

def whoami(objet=None):
    if objet is None:
        msg=''
    else:
        try : msg=objet.__class__.__name__+'.'
        except AttributeError : msg=''
    return '%s::'%(msg+sys._getframe(1).f_code.co_name)#, id(object)

def smallStack(commentaire='', deep=10):
    _smallStack(sys.stdout, commentaire, deep)

def _smallStack(output, commentaire='', deep=1):
    u"""Impression de la pile des appels sur output, format compact"""
    deep = [2,2+deep]
    if commentaire :
        print(commentaire,file=output,)
#     frame = sys._getframe(2)
    k = deep[0]
    toprint = []
    while k<=deep[1] :
        try :
            filename = Path(sys._getframe(k).f_code.co_filename).name
            fonction = sys._getframe(k).f_code.co_name
            num = sys._getframe(k).f_lineno
            toprint.append(u"%s.%s(%d)"%(filename,fonction,num))
            k += 1
        except Exception as msg:
            print(msg)
            break
    toprint.reverse()
    print(8*u' '+u' >> '.join(toprint),file=output)#[1:])
    return

def className(obj:object,enluminure:str='<>'):
    if enluminure :
        return enluminure[0]+obj.__class__.__name__+enluminure[1]
    else :
        return obj.__class__.__name__

def my__str__(self):
    """str générique. """
    level=0
    toprint = ["class %s"%className(self)]+['    %20s : %s'%(key,value) for (key, value) in self.info(level)]
    return '\n'.join(toprint)

if __name__ == '__main__':
    debug()
