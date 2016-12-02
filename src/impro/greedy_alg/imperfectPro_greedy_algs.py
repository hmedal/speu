#!/usr/bin/python

import lxml.etree as etree
import numpy as np
#import gurobipy
import itertools
from multiprocessing import Pool
from scipy.stats import t as tDist
import time
import logging
from edu.msstate.hm568.impro import executableModel, imperfectPro_dataset,\
    imperfectPro_model, imperfectPro_problemInstance
import edu.msstate.hm568.impro.databaseUtil as dbUtil

counter = 0

secondStageProblem = None

#dataset
instance = None

#algorithm
#numSamplesExponent = 0
#numSamplesForFinalExponent = 0
#deltaExponent = 0
t = 0
numSamples = 0
numSampleBunches = 0
timeLimit = 0
#numSamplesForFinal = 0
#delta = 0

#parameters
thetaValues = []

#model
gurobiModel = None
capacityConstraints = None

#other

#paths
exprFilePath = None
dataFilePath = None
hazardsScenarioDefPath = None
algType = None
databaseName = None
hazardType = None

usePreGeneratedSamplesSet = False
samplesSet = None
includeHazards = False
includeHazardsHazardsForProbabilities = True

def my_computeSecondStageUtility(capacityLevels):
    return secondStageProblem.computeSecondStageUtility(capacityLevels, instance.numCapLevels)

def createSamples(sampling = True):
    global numSamples, delta, numSamplesForFinal, thetaValues, usePreGeneratedSamplesSet, samplesSet
    if(sampling):
        usePreGeneratedSamplesSet = False
    else:
        usePreGeneratedSamplesSet = True
        if(includeHazards):
            samplesSet = instance.createScenarios_CapLevels_withHazards()
            numSamples = len(samplesSet)
        else:
            samplesSet = instance.createScenarios_CapLevels()
            numSamples = len(samplesSet)

def computeMarginalDifference(capacityLevels, fac, normalSecondStageCost):
    capacityLevelsAdded = capacityLevels[:]
    capacityLevelsAdded[fac] += 1
    if(capacityLevelsAdded[fac] > (instance.numCapLevels-1)):
        diff = 0
    else:
        diff = my_computeSecondStageUtility(capacityLevelsAdded) - normalSecondStageCost
    return diff

def computeMarginalDifferenceTuple(array):
    return computeMarginalDifference(array[0], array[1][0], array[2])

def computeAverageMarginalDifference(allocationVector, fac, samples, secondStageCosts):
    meanDiff = np.mean([computeMarginalDifference(samples[i], fac, secondStageCosts[i]) for i in range(numSamples)])
    return meanDiff
    
def computeAverageMarginalDifferenceRevised(fac, samples, secondStageCosts, scenarioProbs = None):
    localPoolVar = Pool()
    tuples = [[samples[i], [fac], secondStageCosts[i]] for i in range(len(samples))]
    if(usePreGeneratedSamplesSet):
        output = localPoolVar.map(computeMarginalDifferenceTuple, tuples)
        meanDiff = sum(a*b for a,b in zip(output, scenarioProbs))
    else:
        meanDiff = np.mean(localPoolVar.map(computeMarginalDifferenceTuple, tuples))
    localPoolVar.close()
    localPoolVar.join()
    return meanDiff

def computeAllAverageMarginalDifferencesRevised(allocationVector):
    samples = None
    if(usePreGeneratedSamplesSet):
        #print "usePreGeneratedSamplesSet"
        samples = samplesSet
    else:
        if(includeHazards):
            #print "includeHazards", "!usePreGeneratedSamplesSet"
            samples = [imperfectPro_model.getRandomIndependentBinomials_withHazards(allocationVector, instance) for i in range(numSamples)]
            
        else:
            samples = [imperfectPro_model.getRandomIndependentBinomials(allocationVector, instance) for i in range(numSamples)]
    if(includeHazards):
        capLevelsSamples = instance.convertCapsAndHazardsScenarios_to_CapsScenarios(samples)
        hazardLevelsSamples = instance.convertCapsAndHazardsScenarios_to_HazardScenarios(samples)
    else:
        capLevelsSamples = samples
    #print "capLevelsSamples", capLevelsSamples
    secondStageCosts = [my_computeSecondStageUtility(sample) for sample in capLevelsSamples]
    if(usePreGeneratedSamplesSet):
        if(includeHazards):
            scenarioProbs = [imperfectPro_model.getProbabilityOfCapLevelScenario_withHazards(capLevelsSamples[scenIndex], allocationVector, hazardLevelsSamples[scenIndex], instance) for scenIndex in range(numSamples)]
        else:
            if(includeHazardsHazardsForProbabilities):
                scenarioProbs = [imperfectPro_model.getProbabilityOfCapLevelScenConsideringHazards(scenario, allocationVector, instance) 
                                 for scenario in samplesSet]
            else:
                scenarioProbs = [imperfectPro_model.getProbabilityOfCapLevelScenario(capLevelsSamples[scenIndex], allocationVector, instance) for scenIndex in range(numSamples)]
        return [(computeAverageMarginalDifferenceRevised(fac, capLevelsSamples, secondStageCosts, scenarioProbs), fac) for fac in range(instance.numFacs)]
    else:
        return [(computeAverageMarginalDifferenceRevised(fac, capLevelsSamples, secondStageCosts), fac) for fac in range(instance.numFacs)]

def getFacsWithBestMarginalGain(allocationVector, howManyToIncludeAtATime = 1):
    sortedList = sorted(computeAllAverageMarginalDifferencesRevised(allocationVector), reverse=True)[:int(howManyToIncludeAtATime)]
    return sortedList
        
def getFeasibleFacWithBestMarginalGain(allocationVector, howManyToIncludeAtATime = 1):
    sortedList = getFacsWithBestMarginalGain(allocationVector, instance.numFacs)
    for fac in sortedList:
        if(allocationVector[fac[1]] <= (instance.numAllocLevels - 2)):
            return fac[1]
    
def readInExperimentData(path):
    global numSamples, numSampleBunches, databaseName, dataFilePath, hazardsScenarioDefPath, timeLimit, hazardType
    d = etree.parse(open(path))
    dataFilePath = str(d.xpath('//dataset/path[1]/text()')[0])
    #numSamplesExponent = int(d.xpath('//algorithm/numSamplesExponent[1]/text()')[0])
    #numSamplesForFinalExponent = int(d.xpath('//algorithm/numSamplesForFinalExponent[1]/text()')[0])
    #deltaExponent = float(d.xpath('//algorithm/deltaExponent[1]/text()')[0])
    databaseName = str(d.xpath('//other/databaseName[1]/text()')[0])
    numSamples = int(d.xpath('//algorithm/numSamples[1]/text()')[0])
    timeLimit = float(d.xpath('//algorithm/timeLimit[1]/text()')[0])
    numSampleBunches = int(d.xpath('//algorithm/numSampleBunches[1]/text()')[0])
    hazardsScenarioDefPath = str(d.xpath('//instance/hazardsPath[1]/text()')[0])
    hazardType = str(d.xpath('//instance/hazardType[1]/text()')[0])

def doPrelimStuff():
    global exprFilePath, algType, instance, secondStageProblem
    global delta
    exprFilePath, algType = executableModel.parseExprParamsFilePath()
    readInExperimentData(exprFilePath)
    dataset = imperfectPro_dataset.ImproDataset()
    dataset.readInDataset(dataFilePath)
    instance = imperfectPro_problemInstance.Instance()
    instance.readInExperimentData(exprFilePath)
    instance.readInHazardsScenarioData(hazardsScenarioDefPath)
    instance.createInstance(dataset)
    print "created instance"
    n = instance.numFacs #see Kempe notes 4/22; figure out what n is for my problem later
    delta = 1.0/(n**2) #see Kempe notes 4/22
    #delta = 1/(instance.numAllocLevels - 1.0)  # for comparison with other algorithms
    secondStageProblem = imperfectPro_model.SecondStageProblem()
    secondStageProblem.setInstance(instance)
    secondStageProblem.createModelGurobi()
    print "end doPrelimStuff"
   
def computeAverageSecondStageUtilityRevised(allocationVector, numSamples):
    localPoolVar = Pool()
    print "samplesSet[0]", samplesSet[0]
    print "numSamples: ", len(samplesSet)
    if(usePreGeneratedSamplesSet):
        if(includeHazards):
            capLevelsSamples = instance.convertCapsAndHazardsScenarios_to_CapsScenarios(samplesSet)
            output = localPoolVar.map(my_computeSecondStageUtility, capLevelsSamples)
            print "output", output
            hazardLevelsSamples = instance.convertCapsAndHazardsScenarios_to_HazardScenarios(samplesSet)
            print "numSamples", numSamples
            scenarioProbs = [imperfectPro_model.getProbabilityOfCapLevelScenario_withHazards(capLevelsSamples[scenIndex], 
                                                                                             allocationVector, hazardLevelsSamples[scenIndex], instance) 
                                                                                                for scenIndex in range(numSamples)]
            #print "scenarioProbs", scenarioProbs
        else:
            output = localPoolVar.map(my_computeSecondStageUtility, samplesSet)
            if(includeHazardsHazardsForProbabilities):
                #print 'includeHazardsHazardsForProbabilities'
                #print 'samplesSet', samplesSet
                scenarioProbs = [imperfectPro_model.getProbabilityOfCapLevelScenConsideringHazards(scenario, allocationVector, instance) 
                                 for scenario in samplesSet]
            else:
                scenarioProbs = [imperfectPro_model.getProbabilityOfCapLevelScenario(scenario, allocationVector, instance) 
                                 for scenario in samplesSet]
            #print "scenarioProbs", scenarioProbs
        meanValue = sum(a*b for a,b in zip(output,scenarioProbs))
    else:
        if(includeHazards):
            samples = [imperfectPro_model.getRandomIndependentBinomials_withHazards(allocationVector, instance) for i in range(numSamples)]
            capLevelsSamples = instance.convertCapsAndHazardsScenarios_to_CapsScenarios(samples)
        else:
            samples = [imperfectPro_model.getRandomIndependentBinomials(allocationVector, instance) for i in range(numSamples)]
        meanValue = np.mean(localPoolVar.map(my_computeSecondStageUtility, capLevelsSamples))
    localPoolVar.close()
    localPoolVar.join()
    return meanValue

def DiscreteGreedy(budget, alpha = 0.05):
    allocationVector = [0 for fac in range(instance.numFacs)]
    global t
    logging.info("delta: " + str(delta))
    print "delta", delta
    startTime = time.time()
    while(t < budget):
        print "t= ", t
        fac = getFeasibleFacWithBestMarginalGain(allocationVector)
        allocationVector[fac] += delta
        t += delta
        elapsedTime = time.time() - startTime
        if(elapsedTime >= timeLimit):
            break
        print "allocationVector", allocationVector
    runTime = time.time() - startTime
    averages = []
    for index in range(numSampleBunches):
        averages.append(computeAverageSecondStageUtilityRevised(allocationVector, numSamples))
    print "averages", averages
    avgOfAverages = np.mean(averages)
    if(usePreGeneratedSamplesSet):
        hw = 0
    else:
        lbStdDev = np.std(averages)
        hw = tDist.ppf(1 - alpha/2.0, numSampleBunches - 1) * lbStdDev / np.sqrt(numSampleBunches)
    print "pValues: ", [imperfectPro_model.getProbabilityFromAllocation(i, instance.numAllocLevels) for i in allocationVector]
    print "greedy objective: " + str(avgOfAverages)
    print "average fractionality: ", np.mean([min(1 - val, val - 0) for val in allocationVector])
    print "std dev: ", np.std(allocationVector)
    print "non-zero: ", sum([1 for val in allocationVector if val > 0])
    return allocationVector, avgOfAverages, hw, runTime

def ContinuousGreedy(budget, alpha = 0.05):
    allocationVector = [0 for fac in range(instance.numFacs)]
    global t
    logging.info("delta: " + str(delta))
    print "delta", delta
    print "timeLimit", timeLimit
    startTime = time.time()
    while(t < (instance.numAllocLevels - 1)):
        print "t= ", t
        for fac in getFacsWithBestMarginalGain(allocationVector, budget):
            allocationVector[fac[1]] += delta
        t += delta
        elapsedTime = time.time() - startTime
        if(elapsedTime >= timeLimit):
            print "time elapsed"
            break
        print "allocationVector", allocationVector
    runTime = time.time() - startTime
    averages = []
    print "numSamples", numSamples
    print "numSampleBunches", numSampleBunches
    for index in range(numSampleBunches):
        averages.append(computeAverageSecondStageUtilityRevised(allocationVector, numSamples))
    print "averages", averages
    avgOfAverages = np.mean(averages)
    if(usePreGeneratedSamplesSet):
        hw = 0
    else:
        lbStdDev = np.std(averages)
        hw = tDist.ppf(1 - alpha/2.0, numSampleBunches - 1) * lbStdDev / np.sqrt(numSampleBunches)
    print "pValues: ", [imperfectPro_model.getProbabilityFromAllocation(i, instance.numAllocLevels) for i in allocationVector]
    print "Continuous greedy objective: " + str(avgOfAverages)
    print "average fractionality: ", np.mean([min(1 - val, val - 0) for val in allocationVector])
    print "std dev: ", np.std(allocationVector)
    print "non-zero: ", sum([1 for val in allocationVector if val > 0])
    return allocationVector, avgOfAverages, hw, runTime

def Enumeration(budget, alpha = 0.05):
    facsArray = [np.arange(0, budget+delta, delta) for fac in range(instance.numFacs)]
    possibleSolnVectors = itertools.product(*facsArray)
    logging.info("Number of possible solutions: " + str(len(np.arange(0,1+delta, delta))**instance.numFacs))
    bestValue = -np.Infinity
    bestVector = []
    startTime = time.time()
    for vector in possibleSolnVectors:
        #print instance.budget, vector, max(vector), instance.numAllocLevels, max(vector) <= instance.numAllocLevels
        if((sum(vector) == budget) & (max(vector) <= 1)):
            value = computeAverageSecondStageUtilityRevised(vector,numSamples)
            print vector, value
            if(value > bestValue):
                bestValue = value
                bestVector = vector
    runTime = time.time() - startTime
    averages = []
    for index in range(numSampleBunches):
        averages.append(computeAverageSecondStageUtilityRevised(bestVector, numSamples))
    avgOfAverages = np.mean(averages)
    if(usePreGeneratedSamplesSet):
        hw = 0
    else:
        lbStdDev = np.std(averages)
        hw = tDist.ppf(1 - alpha/2.0, numSampleBunches - 1) * lbStdDev / np.sqrt(numSampleBunches)
    print "pValues: ", [imperfectPro_model.getProbabilityFromAllocation(i, instance.numAllocLevels) for i in bestVector]
    print "Enumeration objective: ", str(bestValue), "vector: ", bestVector
    print "average fractionality: ", np.mean([min(1 - val, val - 0) for val in bestVector])
    print "variance: ", np.var(bestVector)
    return bestVector, bestValue, hw, runTime

def CompareContinuousWithEnumeration():
    return ContinuousGreedy()[1]/Enumeration()[1]

#def getContinuousGreedyAllocationFromContinuousAllocation(allocation, instance):
#    #print "getProbabilityFromContinuousAllocation"
#    return getProbabilityFromContinuousAllocation((instance.numAllocLevels - 1) * allocation)

#def getProbabilityFromContinuousAllocation(x, numAllocLevels):
    #print "getProbabilityFromContinuousAllocation"
    #return (1.0/numAllocLevels) * x + (1.0/numAllocLevels)

def runDiscreteGreedy(sampling = False):
    global delta
    #budget = instance.budget/(instance.numAllocLevels - 1.0)
    #print "budget", budget
    delta = 1
    allocLevelsSoln, bestObj, hw, runTime =  DiscreteGreedy(instance.budget)

    tableName = "GreedyImpro"
    dataSetInfo = ['Daskin', instance.numDemPts, instance.numFacs]
    instanceInfo = [instance.numAllocLevels, instance.numCapLevels, instance.budget, instance.penaltyMultiplier, instance.excess_capacity]
    algParams = [algType, numSamples]
    algOutput = [runTime, bestObj, hw]
    solnOutput = [str(allocLevelsSoln)]
    dbUtil.printResultsToDB(databaseName, tableName, dataSetInfo, instanceInfo, algParams, algOutput, solnOutput)
    
def runContinuousGreedy(sampling = False):
    global delta
    #budget = instance.budget/(instance.numAllocLevels - 1.0)
    delta = 1
    print "instance.budget", instance.budget
    print "numAllocLevels", instance.numAllocLevels
    budget = instance.budget/(instance.numAllocLevels - 1)
    print "budget'", budget
    allocLevelsSoln, bestObj, hw, runTime =  ContinuousGreedy(budget)

    tableName = "GreedyImpro"
    dataSetInfo = ['Daskin', hazardType, instance.numDemPts, instance.numFacs]
    instanceInfo = [instance.numAllocLevels, instance.numCapLevels, instance.budget, instance.penaltyMultiplier, instance.excess_capacity]
    algParams = [algType, numSamples]
    algOutput = [round(runTime,2), round(bestObj,2), hw]
    solnOutput = [str(allocLevelsSoln)]
    dbUtil.printResultsToDB(databaseName, tableName, dataSetInfo, instanceInfo, algParams, algOutput, solnOutput)
    
def runEnumeration(sampling = False):
    budget = instance.budget/(instance.numAllocLevels - 1.0)
    #print "budget", budget
    allocLevelsSoln, bestObj, hw, runTime =  Enumeration(budget)

    tableName = "GreedyImpro"
    dataSetInfo = ['Daskin', instance.numDemPts, instance.numFacs]
    instanceInfo = [instance.numAllocLevels, instance.numCapLevels, instance.budget, instance.penaltyMultiplier, instance.excess_capacity]
    algParams = [algType, numSamples]
    algOutput = [runTime, bestObj, hw]
    solnOutput = [str(allocLevelsSoln)]
    dbUtil.printResultsToDB(databaseName, tableName, dataSetInfo, instanceInfo, algParams, algOutput, solnOutput)

if __name__ == "__main__":
    print "GREEDY"
    doPrelimStuff()
    if(algType == 'continuous-deterministic'):
        print 'continuous-deterministic'
        createSamples(False)
        runContinuousGreedy()
    elif(algType == 'continuous-sampling'):
        print 'continuous-sampling'
        createSamples(True)
        if(numSamples == 0):
            raise NameError('numSamples == 0')
        runContinuousGreedy()
    elif(algType == 'discrete-deterministic'):
        print 'discrete-deterministic'
        createSamples(False)
        runDiscreteGreedy()
    elif(algType == 'discrete-sampling'):
        print 'discrete-sampling'
        createSamples(True)
        if(numSamples == 0):
            raise NameError('numSamples == 0')
        runDiscreteGreedy()
    elif(algType == 'enum-deterministic'):
        print 'enum-deterministic'
        createSamples(False)
        runEnumeration()
    elif(algType == 'enum-sampling'):
        print 'enum-deterministic'
        createSamples(True)
        if(numSamples == 0):
            raise NameError('numSamples == 0')
        runEnumeration()
    else:
        print "invalid algType:", algType