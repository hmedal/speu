'''
Created on Oct 3, 2013

@author: hmedal
'''
import time
import logging

class LShapedAlg(object):
    '''
    classdocs
    '''
    time_limit = 0
    
    def __init__(self, timeLimit):
        self.time_limit = timeLimit
        
    def terminationCriteriaNotMet(self,lb, ub, iteration, runTime):
        if(runTime >= self.time_limit):
            return False
        return lb*(1+.001) < ub
    
    def bendersClassic(self, instance, scenariosSetArg, numBunches, firstStageSolver, secondStageSolver, addCutFn):
        global scenariosSet, gur_probChain_secondStageModel
        scenariosSet = scenariosSetArg
        numScenarios = len(scenariosSet)
        print "time limit: ", self.time_limit
        lb = 1
        ub = float('inf')
        iteration = 0
        runTime = 0
        startTime = time.time()
        bestAllocVarSoln = None
        while(self.terminationCriteriaNotMet(lb, ub, iteration, runTime)):
            print iteration, lb, ub, runTime
            masterLB, masterUB, allocVarSoln, runTime  = firstStageSolver()
            #print "firstStageSoln: ", allocVarSoln
            if(masterUB < ub): ub = masterUB
            totalObjVal, coefficientDualTerms, constantDualTerms = secondStageSolver(allocVarSoln)
            logging.info('totalObjVal' + str(totalObjVal))
            addCutFn(coefficientDualTerms, constantDualTerms, iteration, numBunches)
            #print "totalObjVal", totalObjVal
            if(totalObjVal > lb): 
                lb = totalObjVal
                bestAllocVarSoln = allocVarSoln
            iteration += 1
            runTime = time.time() - startTime
        runTime = time.time() - startTime
        return lb,ub, bestAllocVarSoln, runTime