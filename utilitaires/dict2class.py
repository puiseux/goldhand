import copy
import pickle
from collections import namedtuple, OrderedDict
from pprint import pprint, pformat
from typing import Any
from collections.abc import Iterable

from path import Path

from debog import rdebug, debug, mexit

def makeKeysIdentifiers(obj:dict):
    """Modifie obj : transforme ses keys en identifier au sens isidentifier()==True de Python.
    - en remplaçant tous les caractères illicites par '_'
    - sauf le premier caractère : s'il est illicite, il est remplacé par f_
    Le premier caractère sert de test pour savoir si un attribut de Dict2
    N'est PAS recursif. Si obj['a**'] = b = {'w**':12}
    - la clé 'a**' de obj est modifiée en a__
    - la clé w** de b n'est pas modifié."""
    if not isinstance(obj, dict) :
        return obj
    keys = list(obj.keys())
    ikeys = []
    for s in keys :
        """faire de s un identificateur valide, et le retourner"""
        if not s : continue
        if isinstance(s, int) :
            s = '_%d'%s
        elif not s.isidentifier():
            for c in s:
                if not c.isalnum():
                    s = s.replace(c, '_')
            if s[0]=='_':
                s = 'f_'+s[1:]
            elif s[0].isdigit() :
                s = 'f_'+s
        ikeys.append(s)

    for so,si in zip(keys,ikeys) :
        obj[si] = obj.pop(so)
    return obj


def dict2class(name:str='MyNamedTuple', d=None, level:int=1, display:bool=True) :
    """Transformer récursivement un dictionnaire en class <name>, class(class(class...)).
    L'idée est de pouvoir écrire <name>.toto au lieu de d['toto'] ou
        <name>.a.b.c au lieu de d['a']['b']['c']
    Les (clé, value) du dictionnaire doivent donc être transformées en attribut de classe
    <name>.<clé> = value, ces clés, ainsi que le paramètre <name> doivent donc être des noms de
    variable valides, (les valeurs sont quelconques).
    la méthode str.isidentifier() est chargée de cette verification-transformation
    Si le nom <name> ou une clé contient un caractère c invalide, c est modifié en '_'.
    :param level: int,profondeur de la recursion
    :param d: dict, le dictionnaire à transformer
    :param name: str, nom de la class construite
    :param display: bool, écrire ou non la structure de la classe
    """
    if d is None:
        d = {}
    """Variables statiques"""
    MAX_LEVEL = level
    SHIFT = '\t'
    def identify(s:str) -> str:
        """faire de s un identificateur valide, et le retourner"""
        s0 = s
        if not s.isidentifier() :
            for c in s :
                if not c.isalnum():
                    s = s.replace(c,'_')
            # mexit(s0+s)
        return s

    def _dict2class(name:str, d:Any, level:int, display:bool):
        """ATTENTION
         - [effet de bord : d est modifié] => Obsolete, si d est un dict, on travaille sur une copie de d
         - les listes ne sont pas modifiées : une list de dict N'EST PAS transfromée en liste de classes.
         - idem pour tuple"""

        shift = (MAX_LEVEL-level)*SHIFT
        # name = identify(name)
        if level == 0 or not isinstance(d,dict):
            if display : print(shift + '<%s> : %s ' % (name, d.__class__.__name__))
            return d
        # print()
        if display : print(shift+'class <%s> =>'%name)
        dc = d.copy()
        del d
        makeKeysIdentifiers(dc)
        # debug(dc.keys())
        shift = shift + SHIFT
        ####### instanciation
        for key in list(dc.keys()) :
            value = dc[key]
            if isinstance(value, (dict, OrderedDict)) :
                instance1 = _dict2class(key, value, level=level-1, display=display)
                dc[key] = instance1
            elif display :
                print(shift+key+' : '+dc[key].__class__.__name__)
        try :
            NewClass = namedtuple(name, dc.keys())
            instance = NewClass(**dc)
        except ValueError as msg :
            instance = dc
        return instance

    return _dict2class(name, d, level, display)


def lDicts2lObjs(l: list, identify=True, scanlist=True):
    """transforme une liste en liste de Dict2Obj"""
    ll = []
    for i, x in enumerate(l):
        if isinstance(x, dict):
            ll.append(Dict2Obj(obj=x, identify=identify, scanlist=scanlist))
        else:
            ll.append(x)
    return ll


class Dict2Obj(object):
    SHIFT = 4*' '
    SHIFT = '\t'
    BULLETS = 3*'-+*#'
    _SEP = ':'
    """
    https://www.blog.pythonlibrary.org/2014/02/14/python-101-how-to-change-a-dict-into-a-class/
    Turns a dictionary into a class
    """

    # ----------------------------------------------------------------------
    def __init__(self, obj:dict or list[dict], name:str='', identify:bool=True, scanlist:bool=True, level:int=100):
        """
        :param obj: dict ou liste de dicts,  à transformer en instance(s) de classe (objet(s))
        # :param name: nom de la classe
        :param deep: profondeur de l'indentation lors d'un print. Utile seulement en interne.
            Ne pas utiliser à l'appel
            niveau 1.
            >>> Dict2Obj({'a*':1,'b':2,'c':{'A':10, 'B':20}})
            ... <Dict2Obj>
            ... 	+ a_ : 1
            ... 	+ b : 2
            ... 	+ c : <Dict2Obj>
            ... 		@ A : 10
            ... 		@ B : 20
        :param identify: transformer les clés qui ne sont pas des identificateurs en identificateurs
            au sens Python isidentifier(). par ex. 'quel bordel!' => "quel_bordel_"
        :param scanlist:
        """
        if not name :
            name="no_name"
        self.__name = name
        self.__scanlist = scanlist
        self._indent = 1
        self._level = level

        if level == 0 :
            setattr(self, name, obj)
        else :
            if isinstance(obj, dict):
                dobj = obj.copy()
                if identify :  makeKeysIdentifiers(dobj)
                for key, val in dobj.items():
                    if isinstance(val, dict) :
                        val = Dict2Obj(obj=val, name=key, identify=identify, scanlist=scanlist, level=self._level-1)
                    elif isinstance(val, list) and scanlist:
                        val = lDicts2lObjs(val)
                    else :
                        pass
                    setattr(self, key, val)
            elif isinstance(obj, list) :
                if scanlist: obj = lDicts2lObjs(obj)
                setattr(self, name, obj)
            else:
                raise TypeError ("obj (%s) devrait être une instance de dict"%type(obj))

    #----------------------------------------------------------------------
    def __repr__(self):
        """"""
        keys = [k for k in self.__dict__.keys() if k[0]!='_']
        d = {}
        for k in keys :  d.update({k:self.__dict__[k]})
        return "Dict2Obj(%r)" % d

    #----------------------------------------------------------------------
    def __oneLine(self, item:Any):
        if isinstance(item, (list, tuple)) :
            return [repr(v) for v in item]
        else :
            return item

    def __getitem__(self, item):
        """todo:faire hériter Dict2Obj de dict ?? """
        return getattr(self, item)

    def __str__(self):
        """todo:__str__ n'a pas de paramètre, on ne peut donc pas procéder comme dans self.structure.
        Donc on est obligé de fixer l'indentation à shift = self.__deep * self.SHIFT,
        qui ne vaut pas 0 sauf au premier niveau de la construction. Il faut trouver un moyen..."""
        shift = self._indent * self.SHIFT
        sep = self._SEP
        bullet = self.BULLETS[self._indent]
        info = ["<Dict2Obj>"]
        for key, val in  self.__dict__.items() :
            if key[0] == '_' : continue
            if isinstance(val, list) and self.__scanlist:
                s = []
                for k, item in enumerate(val) :
                    s.append(self.__oneLine(item))
                info.append(shift+"%s %s %s %s"%(bullet, key, sep, s))
            elif isinstance(val, Dict2Obj) :
                val._indent = 1 + self._indent
                info.append(shift+"%s %s %s %s"%(bullet, key, sep, val))
                val._indent = 1
            else :
                s = repr(val) if isinstance(val, str) else val
                info.append(shift+"%s %s %s %s"%(bullet, key, sep, s))
        return '\n'.join(info)
    # ----------------------------------------------------------------------
    def structure(self, indent=0, level=2):
        # indent = 1
        shift = indent * self.SHIFT
        shift1 = shift + self.SHIFT
        bullet = self.BULLETS[indent+1]
        sep = ':'
        info = ["<Dict2Obj> structure(level=%d)"%level]#(deep=%d)"%self.__deep]
        if level > 0 :
            for key, val in  self.__dict__.items() :
                if key[0] != '_' :
                    if isinstance(val, Dict2Obj):
                        info.append(shift1+"%s %s : %s"%(bullet, key, val.structure(1+indent, level-1)))
                    elif isinstance(val, (list,tuple)) and self.__scanlist and val :
                        vn = val[0].__class__.__name__
                        info.append(shift1+"%s %s %s %s = [%s,...], len = %d"%(bullet, key,sep,
                                                                    val.__class__.__name__, vn, len(val)))
                    else :
                        info.append(shift1+"%s %s %s %s"%(bullet, key, sep, val.__class__.__name__))

        return '\n'.join(info)

    def toSheet(self):
        """ecrire self sous forme tableur"""
        sep = Dict2Obj._SEP
        Dict2Obj._SEP='\t'
        print(self)
        Dict2Obj._SEP = sep

    @property
    def keys(self):
        return [k for k in self.__dict__.keys() if k[0]!='_']

    @property
    def dict(self):
        """retourne un dict de self.
        Normalement Dict2Obj(d).dict == d sauf certaines clés :
            - celles qui ne sont pas des identifiers
                + caractères non alnum() remplacés par '_'
                + le premier caractère est un chiffre -> ajout de 'f_' devant
            - celles qui commencent par _ :
        exemples :
            key='@:;x' -> newkey='___x'
            key='1x' -> newkey='f_1x'
            key='_x' -> newkey='f_x'
        """
        d = copy.deepcopy(self.__dict__)
        #Attention, si on écrit d = self.__dict__.copy(), (shallow copy), il y a des effets de bord
        keys = list(d.keys())
        for key in keys :
            value = d[key]
            if key[0] == '_' :
                d.pop(key)
            elif isinstance(value,Dict2Obj):
                d[key] = value.dict
            elif isinstance(value,list):
                for i, val in enumerate(value) :
                    if isinstance(val, Dict2Obj) :
                        value[i] = val.dict
                d[key] = value
        return d

if __name__ == "__main__":
    from pytictoc import TicToc
    ball_dict = {
        'B':[1,2],
        'A':[],
        "1size**": {"en": {"inches":8.0, "feet":"I dont know, f*** english system !!"},
                    "fr":"20,32 cm"},
        "genre":{'fr':['footballeur', 'rugbyman'],'12':['hooligans', 'gentlemen']},
        "team":[
            dict(nom='Zidane', role='meneur', num=10),
            dict(nom='Lloris', role='gardien', num=1),
            dict(nom='Ondra', role='climber', num=0.07)
                ],
        "color": "marron",
        "material": ["rubber","plastic"]
    }
    ball = Dict2Obj(obj=ball_dict, identify=True, scanlist=True)
    # print(ball)
    print(ball.structure())
    print(Dict2Obj([]))
    # mexit()

    if 1 :
        fpath = '/Users/puiseux/Google Drive/trading/crypto-data/trash/cginfos.pkl'
        with open(fpath, 'rb') as f :
            mushmush = pickle.load(f)
        fpath = Path(fpath.replace('.pkl','.pyd'))
        fpath.write_text(pformat(mushmush, width=250))

        makeKeysIdentifiers(mushmush)
        AAVE = mushmush['AAVE']
        oAAVE=Dict2Obj(AAVE)
        print(oAAVE)
        # for key, value in AAVE.items() :
        #     debug(titre=key)
        #     obj = Dict2Obj(value)
        mexit()
    if 1 :
        debug(keys=ball.keys)
        ball.toSheet()
        mexit()
    if 1:
        a = Dict2Obj(obj={'l':[{'a': 1}, {'b': 2}], 'c': {'A': 10, 'B': 20}})
        b = Dict2Obj(obj={'l': ['a', 'b'], 'c': {'A': 10, 'B': 20}})
        debug(a=a)
        s = 'a.structure(level=10)', a.structure(level=10)
        print(*s)
        print('a.dict =',a.dict)
        s1 = 'a.structure(level=1)', a.structure(level=1)
        print(*s1)
        print('s = s1 ??', s==s1)
        # debug(b=b)
        # print(b.structure(level=1))
        # print(b)
        # # print(Dict2Obj(obj={'l':[{'a': 1}, {'b': 2}], 'c': {'A': 10, 'B': 20}}).structure(1))
        # mexit()
        # mexit()
    if 1 :
        debug('ball_dict=')
        pprint(ball_dict)
        debug(ball=ball)
        debug('ball.dict=')
        pprint(ball.dict)
        debug('ball.structure(level=10)'), print(ball.structure(level=10))
        debug('ball.structure(level=1)'), print(ball.structure(level=1))
        mexit()
    if 1 :
        debug(ball)
        print('ball.color =',ball.color)
        print('ball.size__ =',ball.size__)
        print('ball.genre =',ball.genre)
        print('ball.team =',ball.team)
        mexit()
        debug('ball.structure(level=10)'), print(ball.structure(level=10))
        debug('ball.structure(level=1)'), print(ball.structure(level=1))
    if 0:
        t = TicToc()
        bdc = ball_dict.copy()
        t.tic()
        for i in range(10000) :
            ball = Dict2Obj(obj=ball_dict, identify=True)
        print('ball_dict==bdc ?', ball_dict==bdc)
        debug(ball)
        # mexit()
        print('ball.size___.en',ball.size___.en)
        print('ball.size___.fr',ball.size___.fr)
        t.toc(restart=True)
        # rdebug()
        for i in range(10000) :
            ball = dict2class('ball_dict',ball_dict,10,False)
        print(ball)
        print('ball.size___.en',ball.size___.en)
        print('ball.size___.fr',ball.size___.fr)

        t.toc()
        print("dict2class est beaucoup plus lent qur Dict2Obj car \n",
              "- je fais une copie du dict en entrée \n",
              "- je manipule les clés du dictionnaire en vérifiant qu'elles sont de identifiants de ",
              "variables (isidentifier() et identify())\n"
              "- J'utilise namedtuple() dont je ne connais pas les performances")
