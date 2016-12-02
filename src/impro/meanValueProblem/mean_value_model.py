import lxml.etree as etree
import numpy as np
from scipy import stats
import gurobipy
import itertools
from multiprocessing import Pool, cpu_count
import time
from scipy.stats import t as tDist
import logging
from edu.msstate.hm568.impro import executableModel, imperfectPro_dataset,\
    imperfectPro_model, imperfectPro_problemInstance
from edu.msstate.hm568.impro.imperfectPro_problemInstance import Instance
import edu.msstate.hm568.impro.databaseUtil as dbUtil
import edu.msstate.hm568.impro.myUtil as myutil

counter = 0

secondStageProblem = None

#values
amtOfPossibleImprovement = 0
objWithoutPro = 0

#dataset
instance = None

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

writeToFile = False
includeHazards = True
debug = False
trace = True
solveProbChainSecondStageByInspection = False
    
def createFacilityProbabilitiesForAllocationLevels_withHazards():
    global facilityProbabilitiesForAllocationLevels
    facilityProbabilitiesForAllocationLevels = [[[[imperfectPro_model.getCapLevelProbabilityForAllocationLevelAndHazardLevel(j,l,k, m, instance, instance.numCapLevels) 
                                                  for k in range(instance.numAllocLevels)] for l in range(instance.numCapLevels)] 
                                                    for m in range(instance.numHazardLevels)] for j in range(instance.numFacs)]

def getCapAmtForLevel(fac, capLevel):
    return (capLevel/float(instance.numCapLevels - 1)) * float(instance.capacities[fac])
    
def getExpectedCapForFacGivenAllocLevel(fac, allocLevel):
    expCapForHazardScenario = [sum([facilityProbabilitiesForAllocationLevels[fac][hazardScen[fac]][l][allocLevel]*getCapAmtForLevel(fac, l) 
               for l in range(instance.numCapLevels)]) for hazardScen in instance.scenarioHazardLevels]
    expCap = sum(a*b for a,b in zip(expCapForHazardScenario, instance.scenarioProbs))
    return expCap
        
def createMeanValueProblem(showOutput = False):
    global gurobiModel,capacityConstraints, allocVars
    print "createModelGurobi"
    gurobiModel = gurobipy.Model("myLP")
    #print "GUROBI MODEL CREATED"
    #print [sum([OLD_getExpectedCapForFacGivenAllocLevel(j,k) for k in range(instance.numAllocLevels)]) for j in range(instance.numFacs)]
    try:
        # Create variables
        allocVars = [[gurobiModel.addVar(0,1,vtype = gurobipy.GRB.BINARY,name="y_"+str(j)+","+str(k)) for k in range(instance.numAllocLevels)] for j in range(instance.numFacs)]
        gurobiModel.update()        
        assignVars = [[gurobiModel.addVar(0,1,vtype=gurobipy.GRB.CONTINUOUS,name="x_"+str(i)+","+str(j)) for j in range(instance.numFacs+1)] for i in range(instance.numDemPts)]
        # Integrate new variables
        gurobiModel.update()
        # Set objective
        gurobiModel.setObjective(sum([instance.demPtWts[i]*instance.pairsUtilityMatrix[i][j]*assignVars[i][j] for j in range(instance.numFacs+1) for i in range(instance.numDemPts)]), gurobipy.GRB.MAXIMIZE)
        gurobiModel.update()
        # create capacity contraints
        capacityConstraints = [gurobiModel.addConstr(sum([instance.demPtWts[i]*assignVars[i][j] for i in range(instance.numDemPts)]) <= 
                                                     sum([getExpectedCapForFacGivenAllocLevel(j,k)*allocVars[j][k] for k in range(instance.numAllocLevels)]), 
                                                     "capacity_"+str(j)) for j in range(instance.numFacs)]
        gurobiModel.update()
        #create demand met constraints
        for i in range(instance.numDemPts):
            gurobiModel.addConstr(sum([assignVars[i][j] for j in range(instance.numFacs+1)]) == 1, "demand_met"+str(i))
        gurobiModel.update()
        # alloc levels constraint
        [gurobiModel.addConstr(sum([allocVars[j][k] for k in range(instance.numAllocLevels)]) == 1, "one alloc level_" + str(j)) for j in range(instance.numFacs)]
        gurobiModel.update()
        # instance.budget constraint
        gurobiModel.addConstr(sum([costOfAlloc[j][k] * allocVars[j][k] for k in range(instance.numAllocLevels) for j in range(instance.numFacs)]) <= instance.budget, "allocation instance.budget")
        gurobiModel.update()
        gurobiModel.setParam('OutputFlag', False)
        
    except gurobipy.GurobiError as e:
        print 'createModelGurobi: Gurobi Error reported' + str(e)
        #logging.error('Error reported')

def my_computeSecondStageUtility(capacityLevels):
    return secondStageProblem.computeSecondStageUtility(capacityLevels, instance.numCapLevels)

def getObjectiveValueForAllocationSolution(allocLevelsSoln):
    #print "pValues for soln: ", [imperfectPro_model.getProbabilityFromAllocation(i, instance.numAllocLevels) for i in allocLevelsSoln]
    #capLevelsSamples = instance.createScenarios_CapLevels_withHazards()
    #print "samplesForActualCapLevels", samplesForActualCapLevels
    #capLevelsScenarios = [[sample[fac][1] for fac in range(instance.numFacs)] for sample in capLevelsSamples]
    capLevelsScenarios = instance.createScenarios_CapLevels()
    #print 'capLevelsScenarios', capLevelsScenarios
    #print "capLevelsScenarios[0]", capLevelsScenarios[0]
    #print "capLevelsScenarios len", len(capLevelsScenarios)
    localPoolVar = Pool()
    scenarioObjVals = localPoolVar.map(my_computeSecondStageUtility, capLevelsScenarios,1)
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

def solveMeanValueProblem_BB():
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
    allocVarSoln = [[allocVars[j][k].getAttr("X") for k in range(instance.numAllocLevels)] for j in range(instance.numFacs)]
    allocLevelsSoln = imperfectPro_model.getAllocLevelsVectorFromBinaryVector(allocVarSoln, instance)
    lb = gurobiModel.objVal
    ub = gurobiModel.getAttr('ObjBound')
    lb, ub, allocVarSoln, runTime
    print "lb: ", lb, "ub: ", ub
    print "allocVarSoln: ", allocLevelsSoln
    objToOriginalProb = getObjectiveValueForAllocationSolution(allocLevelsSoln)
    print "objToOriginalProb", objToOriginalProb
    tableName = "MeanValueImpro"
    dataSetInfo = ['Daskin', hazardType, instance.numDemPts, instance.numFacs]
    instanceInfo = [instance.numAllocLevels, instance.numCapLevels, instance.budget, instance.penaltyMultiplier, instance.excess_capacity]
    algParams = []
    algOutput = [runTime, lb, ub]
    solnOutput = [str(allocLevelsSoln), objToOriginalProb]
    dbUtil.printResultsToDB(databaseName, tableName, dataSetInfo, instanceInfo, algParams, algOutput, solnOutput)
        
def afterReadData():
    global costOfAlloc
    costOfAlloc= [[k for k in range(instance.numAllocLevels)] for j in range(instance.numFacs)]
    if(includeHazards):
        createFacilityProbabilitiesForAllocationLevels_withHazards()

def readInExperimentData(path):
    global dataFilePath, hazardsScenarioDefPath, hazardType
    global bendersType, g_numBunches, numSAA_first_stage_probs, numSAA_first_stage_samples, numSAA_second_stage_samples, databaseName, time_limit
    global g_secondStageAlg
    d = etree.parse(open(path))
    dataFilePath = str(d.xpath('//dataset/path[1]/text()')[0])
    hazardsScenarioDefPath = str(d.xpath('//instance/hazardsPath[1]/text()')[0])
    hazardType = str(d.xpath('//instance/hazardType[1]/text()')[0])
    time_limit = float(d.xpath('//algorithm/timeLimit[1]/text()')[0])
    databaseName = str(d.xpath('//other/databaseName[1]/text()')[0])

def doPrelimStuff():
    global exprFilePath, algType, instance, secondStageProblem
    exprFilePath, algType = executableModel.parseExprParamsFilePath()
    readInExperimentData(exprFilePath)
    dataset = imperfectPro_dataset.ImproDataset()
    dataset.readInDataset(dataFilePath)
    instance = imperfectPro_problemInstance.Instance()
    instance.readInExperimentData(exprFilePath)
    #print "hazardsScenarioDefPath", hazardsScenarioDefPath
    instance.readInHazardsScenarioData(hazardsScenarioDefPath)
    instance.createInstance(dataset)
    afterReadData()
    secondStageProblem = imperfectPro_model.SecondStageProblem()
    secondStageProblem.setInstance(instance)
    secondStageProblem.createModelGurobi()
    createMeanValueProblem()
    
if __name__ == "__main__":
    if(debug):
        print "!!!WARNING: DEBUG MODE!!!"
    print "MEAN VALUE PROBLEM"
    doPrelimStuff()
    if(algType == 'bb'):
        solveMeanValueProblem_BB()
        
#createScenarios_withHazardsFromFile(instance.numCapLevels)
#generateRandomSetOfScenarios_withHazards_fromFile(10)
#instance.createScenarios_CapLevels_withHazards()
#instance.createScenarios_CapLevels_AlternateNumCapLevels_withHazards(5)