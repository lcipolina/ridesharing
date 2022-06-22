import itertools
def flatten(lol):
    '''
    :inputs: list of list
    Converts a List of  (Lists of Lists) into a flat list
    '''
    if len(lol) == 1:
        return lol
    else:
        res = list(itertools.chain.from_iterable(lol))
        return res