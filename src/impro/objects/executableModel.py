'''
Created on Aug 23, 2013

@author: hmedal
'''

import argparse
import logging

#paths
exprFilePath = None
dataFilePath = None

def parseExprParamsFilePath():
    global exprFilePath
    parser = argparse.ArgumentParser(description='Read a filename.')
    parser.add_argument('--exprfile', metavar='N', nargs='+', help='the experiment file for the experiment')
    parser.add_argument('--algType', metavar='N', nargs='+', help='the experiment file for the experiment')
    args = parser.parse_args()
    exprFilePath =  args.exprfile[0]
    algType =  args.algType[0]
    return exprFilePath, algType
