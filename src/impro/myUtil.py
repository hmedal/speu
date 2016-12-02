'''
Created on Oct 3, 2013

@author: hmedal
'''
import itertools
    
def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return itertools.izip_longest(fillvalue=fillvalue, *args)

def grouper_asList(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    output = itertools.izip_longest(fillvalue=fillvalue, *args)
    return [list(i) for i in output]

def createSparseList(length, indexForOne):
    list = [0] * length
    list[indexForOne] = 1
    return list

if __name__ == "__main__":
    myList = [1,2,3,4,5,6,7,8,9]
    groups = grouper_asList(myList, 3)
    print [g for g in groups]
    print groups[0] + groups[1]