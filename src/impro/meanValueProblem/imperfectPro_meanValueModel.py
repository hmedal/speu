   
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
from edu.msstate.hm568.impro import executableModel, imperfectPro_dataset,\
    imperfectPro_model, imperfectPro_problemInstance
from edu.msstate.hm568.impro.imperfectPro_problemInstance import Instance
import edu.msstate.hm568.impro.databaseUtil as dbUtil
import edu.msstate.hm568.impro.myUtil as myutil

writeToFile = False
includeHazards = True
debug = False
trace = True
solveProbChainSecondStageByInspection = False

def getCapAmtForLevel(fac, capLevel, instance):
    return (capLevel/float(instance.numCapLevels - 1)) * float(instance.capacities[fac])

class ImproMeanValueModel(object):
    '''
    classdocs
    '''
    
    counter = 0

    secondStageProblem = None
    
    #values
    amtOfPossibleImprovement = 0
    objWithoutPro = 0
    
    #dataset
    instance = None
    secondStageProblem = None
    
    #algorithm
    time_limit = 0
    g_numBunches = 0
    
    #parameters
    thetaValues = []
    
    #indices
    hazardScenarioIndices = []
    
    #model
    gurobiModel = None
    #gur_probChain_secondStageModel = None
    
    #constraints
    capacityConstraints = None
    
    #variables
    allocVars = None
    
    #other
    facilityProbabilitiesForAllocationLevels = None #[j][m][l][k] 
    
    #paths and flags
    exprFilePath = None
    dataFilePath = None
    hazardsScenarioDefPath = None
    hazardType = None
    algType = None
    databaseName = None
    g_secondStageAlg = None
    
    #flags
    reset_mode = True
    
    callback_counter = 0
    second_stage_counter = 0
    
    global_sampleScenarios = False
        
    def afterReadData(self):
        self.costOfAlloc= [[k for k in range(self.instance.numAllocLevels)] for j in range(self.instance.numFacs)]
        if(includeHazards):
            self.createFacilityProbabilitiesForAllocationLevels_withHazards()
    
    def setInstance(self, instance):
        self.instance = instance
        
    def setSecondStageProblem(self, secondStageProb):
        self.secondStageProblem = secondStageProb
        
    def createFacilityProbabilitiesForAllocationLevels_withHazards(self):
        self.facilityProbabilitiesForAllocationLevels = [[[[imperfectPro_model.getCapLevelProbabilityForAllocationLevelAndHazardLevel(j,l,k, m, self.instance, self.instance.numCapLevels) 
                                                  for k in range(self.instance.numAllocLevels)] for l in range(self.instance.numCapLevels)] 
                                                    for m in range(self.instance.numHazardLevels)] for j in range(self.instance.numFacs)]
    
    def getExpectedCapForFacGivenAllocLevel(self, fac, allocLevel):
        #print self.instance.scenarioHazardLevels
        expCapForHazardScenario = [sum([self.facilityProbabilitiesForAllocationLevels[fac][hazardScen[fac]][l][allocLevel]*getCapAmtForLevel(fac, l, self.instance) for l in range(self.instance.numCapLevels)]) for hazardScen in self.instance.scenarioHazardLevels]
        expCap = sum(a*b for a,b in zip(expCapForHazardScenario, self.instance.scenarioProbs))
        return expCap

    def createModelGurobi(self, showOutput = False):
        global gurobiModel,capacityConstraints, allocVars
        print "createModelGurobi"
        gurobiModel = gurobipy.Model("myLP")
        #print "GUROBI MODEL CREATED"
        #print [sum([OLD_getExpectedCapForFacGivenAllocLevel(j,k) for k in range(instance.numAllocLevels)]) for j in range(instance.numFacs)]
        try:
            # Create variables
            allocVars = [[gurobiModel.addVar(0,1,vtype = gurobipy.GRB.BINARY,name="y_"+str(j)+","+str(k)) for k in range(self.instance.numAllocLevels)] for j in range(self.instance.numFacs)]
            gurobiModel.update()        
            assignVars = [[gurobiModel.addVar(0,1,vtype=gurobipy.GRB.CONTINUOUS,name="x_"+str(i)+","+str(j)) for j in range(self.instance.numFacs+1)] for i in range(self.instance.numDemPts)]
            # Integrate new variables
            gurobiModel.update()
            # Set objective
            gurobiModel.setObjective(sum([self.instance.demPtWts[i]*self.instance.pairsUtilityMatrix[i][j]*assignVars[i][j] for j in range(self.instance.numFacs+1) for i in range(self.instance.numDemPts)]), gurobipy.GRB.MAXIMIZE)
            gurobiModel.update()
            # create capacity contraints
            capacityConstraints = [gurobiModel.addConstr(sum([self.instance.demPtWts[i]*assignVars[i][j] for i in range(self.instance.numDemPts)]) <= 
                                                         sum([self.getExpectedCapForFacGivenAllocLevel(j,k)*allocVars[j][k] for k in range(self.instance.numAllocLevels)]), 
                                                         "capacity_"+str(j)) for j in range(self.instance.numFacs)]
            gurobiModel.update()
            #create demand met constraints
            for i in range(self.instance.numDemPts):
                gurobiModel.addConstr(sum([assignVars[i][j] for j in range(self.instance.numFacs+1)]) == 1, "demand_met"+str(i))
            gurobiModel.update()
            # alloc levels constraint
            [gurobiModel.addConstr(sum([allocVars[j][k] for k in range(self.instance.numAllocLevels)]) == 1, "one alloc level_" + str(j)) for j in range(self.instance.numFacs)]
            gurobiModel.update()
            # instance.budget constraint
            gurobiModel.addConstr(sum([self.costOfAlloc[j][k] * allocVars[j][k] for k in range(self.instance.numAllocLevels) for j in range(self.instance.numFacs)]) <= self.instance.budget, "allocation instance.budget")
            gurobiModel.update()
            gurobiModel.setParam('OutputFlag', False)
            
        except gurobipy.GurobiError as e:
            print 'createModelGurobi: Gurobi Error reported' + str(e)
            #logging.error('Error reported')
            
    def getObjectiveValueForAllocationSolution(self, allocLevelsSoln, instance):
        #print "pValues for soln: ", [imperfectPro_model.getProbabilityFromAllocation(i, instance.numAllocLevels) for i in allocLevelsSoln]
        #capLevelsSamples = instance.createScenarios_CapLevels_withHazards()
        #print "samplesForActualCapLevels", samplesForActualCapLevels
        #capLevelsScenarios = [[sample[fac][1] for fac in range(instance.numFacs)] for sample in capLevelsSamples]
        capLevelsScenarios = instance.createScenarios_CapLevels()
        #print 'capLevelsScenarios', capLevelsScenarios
        #print "capLevelsScenarios[0]", capLevelsScenarios[0]
        #print "capLevelsScenarios len", len(capLevelsScenarios)
        localPoolVar = Pool()
        scenarioObjVals = localPoolVar.map(self.my_computeSecondStageUtility, capLevelsScenarios,1)
        #conditionalScenarioProbsActual = [imperfectPro_model.getProbabilityOfCapLevelScenario_AltNumCapLevels_withHazards([facInfo[1] for facInfo in scenario], allocLevelsSoln, 
        #                                                                                                      [facInfo[0] for facInfo in scenario], instance, 
        #                                                                       instance.numAllocLevelsActual, instance.numCapLevelsActual) for scenario in capLevelsSamples]
        #scenarioProbs = [instance.scenarioProbs[instance.hazardScenarioIndicesAtCapLevels[scenNumber]]*conditionalScenarioProbsActual[scenNumber] 
        #                 for scenNumber in range(len(conditionalScenarioProbsActual))]
        scenarioProbs = [imperfectPro_model.getProbabilityOfCapLevelScenConsideringHazards(scenario, allocLevelsSoln, instance) 
                                     for scenario in capLevelsScenarios]
        #print "scenarioProbs", scenarioProbs
        #print "scenarioProbs sum", sum(scenarioProbs)
        objToOriginalProb = sum([a*b for a,b, in zip(scenarioObjVals, scenarioProbs)])
        return objToOriginalProb

    def solveMeanValueProblem_BB(self):
        #print "solving benders master problem"
        startTime = time.time()
        if(writeToFile):
            gurobiModel.write("/home/hmedal/Documents/Temp/meanValueProblem.lp")
        gurobiModel.optimize()
        #print "benders master problem solved"
        runTime = time.time() - startTime
        #     if(global_sampleScenarios):
        #         cadinalityOfSampleSpace = instance.numCapLevels**instance.numFacs
        #         multiplier_to_compute_expectation = cadinalityOfSampleSpace/(numScenarios + 0.0)
        #     else:
        #         multiplier_to_compute_expectation = 1.0
        allocVarSoln = [[allocVars[j][k].getAttr("X") for k in range(self.instance.numAllocLevels)] for j in range(self.instance.numFacs)]
        allocLevelsSoln = imperfectPro_model.getAllocLevelsVectorFromBinaryVector(allocVarSoln, self.instance)
        lb = gurobiModel.objVal
        ub = gurobiModel.getAttr('ObjBound')
        return lb, ub, allocVarSoln, runTime
        #print "lb: ", lb, "ub: ", ub
        #print "allocVarSoln: ", allocLevelsSoln
        #objToOriginalProb = self.getObjectiveValueForAllocationSolution(allocLevelsSoln, self.instance)
        #print "objToOriginalProb", objToOriginalProb
        #tableName = "MeanValueImpro"
        #dataSetInfo = ['Daskin', self.hazardType, self.instance.numDemPts, self.instance.numFacs]
        #instanceInfo = [self.instance.numAllocLevels, self.instance.numCapLevels, self.instance.budget, self.instance.penaltyMultiplier, 
        #                self.instance.excess_capacity]
        #algParams = []
        #algOutput = [runTime, lb, ub]
        #solnOutput = [str(allocLevelsSoln), objToOriginalProb]
        #dbUtil.printResultsToDB(self.databaseName, tableName, dataSetInfo, self.instanceInfo, algParams, algOutput, solnOutput)
    
    def my_computeSecondStageUtility(self, capacityLevels):
        return self.secondStageProblem.computeSecondStageUtility(capacityLevels, self.instance.numCapLevels)