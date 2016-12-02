'''
Created on Aug 23, 2013

@author: hmedal
'''

import lxml.etree as etree
import numpy as np
from gurobipy import *
import itertools
from multiprocessing import Pool
import time
#from pulp import *
import argparse

class ImproDataset(object):
    '''
    classdocs
    '''
    numDemPts = 0
    capacities = 0
    pairsDistMatrix = None
    
    def readInDataset(self, path):
        #print "path: ", path
        #global numFacs, demPtWts, numDemPts, capacities, pairsDistMatrix
        d = etree.parse(open(path))
        #facility information
        
        self.facXVals = [float(i) for i in d.xpath('//fac/@x')]
        self.facYVals = [float(i) for i in d.xpath('//fac/@y')]
        self.coor = [[self.facXVals[i],self.facYVals[i]] for i in range(len(self.facXVals))]
        facNames = d.xpath('//fac/@name')
        
        self.numFacs = len(self.facXVals)
        self.facIDs = [int(i) for i in d.xpath('//fac/@id')]
        #demand point information
        demPtXVals = [float(i) for i in d.xpath('//demPt/@x')]
        demPtYVals = [float(i) for i in d.xpath('//demPt/@y')]
        demPtIDs = [int(i) for i in d.xpath('//demPt/@id')]
        demPtNames = d.xpath('//demPt/@name')
        self.demPtWts = [float(i) for i in d.xpath('//demPt/@wt')]
        self.numDemPts = len(demPtXVals)
        #sumDemand = sum(self.demPtWts)
        #capacities = [(sumDemand/((1-excess_capacity)*numFacs)) for i in range(numFacs)]
        #pairs info
        pairsDist = d.xpath('//pair/@dist')
        #pairsDistFloats = [float(value) for value in pairsDist]
        self.pairsDistMatrix = [[float(pairsDist[i*self.numFacs + j]) for j in range(self.numFacs)] for i in range(self.numDemPts)]
        #maxDist = max(pairsDistFloats)
        #print "numFacs: ", self.numFacs
        #pairsUtilityMatrix = [[utility(pairsDistMatrix[i][j]) for j in range(numFacs)] for i in range(numDemPts)]
        #for i in range(numDemPts):
        #    pairsUtilityMatrix[i].append(penaltyMultiplier * utility(maxDist))
        