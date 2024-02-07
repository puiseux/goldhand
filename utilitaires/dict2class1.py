from collections import UserDict, Mapping, namedtuple

# from UserDict import IterableUserDict

__author__ = 'github.com/hangtwenty'

from pprint import pprint
def condition(x) :
    """condition pour que """

def tupperwar(x:Mapping):
    """ Convert mappings to 'tupperwares' recursively.

    Lets you use dicts like they're JavaScript Object Literals (~=JSON)...
    It recursively turns mappings (dictionaries) into namedtuples.
    Thus, you can cheaply create an object whose attributes are accessible
    by dotted notation (all the way down).

    Use cases:

        * Fake objects (useful for dependency injection when you're making
         fakes/stubs that are simpler than proper mocks)

        * Storing data (like fixtures) in a structured way, in Python code
        (data whose initial definition reads nicely like JSON). You could do
        this with dictionaries, but namedtuples are immutable, and their
        dotted notation can be clearer in some contexts.

    Args:
        mapping: An object that might be a mapping. If it's a mapping, convert
        it (and all of its contents that are mappings) to namedtuples
        (called 'Tupperwares').

    Returns:
        A tupperware (a namedtuple (of namedtuples (of namedtuples (...)))).
        If argument is not a mapping, it just returns it (this enables the
        recursion).
        :param x:
    """
    if isinstance(x, Mapping):# and not isinstance(mapping, ProtectedDict):
        for key, value in x.items():
            x[key] = tupperwar(value)
        return namedtuple_from_mapping(x)
    if isinstance(x, list):
        ly = []
        for y in x : ly.append(tupperwar(y))
        return ly

    return x

def namedtuple_from_mapping(mapping, name="Tupperwar"):
    this_namedtuple_maker = namedtuple(name, mapping.keys())
    return this_namedtuple_maker(**mapping)

class ProtectedDict(UserDict):
    """ A class that exists just to tell `tupperware` not to eat it.

    `tupperware` eats all dicts you give it, recursively; but what if you
    actually want a dictionary in there? This will stop it. Just do
    ProtectedDict({...}) or ProtectedDict(kwarg=foo).
    """

def tupperware_from_kwargs(**kwargs):
    return tupperwar(kwargs)

if __name__ == '__main__':
    d = {'data': [{'first': '2022-01-02',
           'rank': 122,
           'slug': 'le2dcbpchi9',
           'symbol': '27id119bjryh'},
          {'first': '2022-01-02',
           'id': 6144,
           'rank': dict(a='a',b=2),
           'slug': 'hkiq9yguvo4',
           'symbol': 'u7kjkdcl32g'}],
         'status':dict(time=0, days=1)}
    td = tupperwar(d)
    pprint(td, width=100)
    pprint(td.data, width=100)
    print(td.data[0])
    print(td.data[0].first)

    print(dict(td))