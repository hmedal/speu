import xml.etree.cElementTree as ET
import os
import numpy as np
import itertools

#paths and strings
dataPath = None
ending = None
exprFilePath = None
exeStatement = None
runFilesPath = None
runFilesPathName = None
localExprFilePath = None
databaseName = None
#hazardsFileEnding = "_conditional"

#names and statements
cluster = None
moduleStatement = None
timeString = None
myNumThreads = 0

#constants
numThreadsLocal = 8
numThreadsHPC = 16
alg_time_limit = 3600.0

exeMap = {'greedy' : "greedy_alg/imperfectPro_greedy_algs.py", 'probabilityChain' : "probabilityChain/probability_chain_model.py",
          'randomDraw' : "randomDraw/random_draw_model.py", 'meanVal' : "meanValueProblem/mean_value_model.py"}

hazardFileEndingsMap = {'conditional' : "_conditional", 'allExposed' : "_allFullyExposedAlways", 'halfExposed' : "_halfExposedAlways"}

jobFilesList = []

def setQueueInfo(debug):
    global numThreadsHPC
    global cluster, timeString
    if(debug):
        cluster = 'debug12core' # there is no debug for 16 core
        timeString = '30:00'
        numThreadsHPC = 12
    else:
        cluster = 'tiny16core'
        timeString = '2:00:00'
        
def setPathsAndNumThreads(local, exeMapKey):
    global dataPath, ending, exprFilePath, exeStatement, runFilesPath, localExprFilePath, runFilesPathName
    global moduleStatement, myNumThreads, databaseName
    if(local):
        moduleStatement = ''
        exeStatement = '/home/hmedal/Documents/2_msu/research_manager/code/ide/eclipse/Impro_submodular/edu/msstate/hm568/impro/' + exeMap[exeMapKey]
        dataPath = '/home/hmedal/Documents/2_msu/research_manager/data/'
        ending = '_local'
        localExprFilePath ='/home/hmedal/Documents/2_msu/1_MSU_Projects/facilityProtection_CSF/exprFiles' +ending
        exprFilePath = localExprFilePath
        runFilesPath = '/home/hmedal/Documents/2_msu/1_MSU_Projects/facilityProtection_CSF/runFiles' + ending
        runFilesPathName = '/home/hmedal/Documents/2_msu/1_MSU_Projects/facilityProtection_CSF/runFiles' + ending
        myNumThreads = numThreadsLocal
        databaseName = "/home/hmedal/Documents/2_msu/1_MSU_Projects/facilityProtection_CSF/expr_output/impro-results_local.db"
    else:
        moduleStatement = "module load mkl/13.1.0 python/2.7.5 gurobi/5.5.0"
        exeStatement = '~/code/src/python/Impro_submodular/edu/msstate/hm568/impro/' + exeMap[exeMapKey]
        dataPath = '/home/hmedal/data/'
        ending = ''
        exprFilePath ='/home/hmedal/exprFiles/imperfectPro'
        localExprFilePath ='/home/hmedal/Documents/2_msu/1_MSU_Projects/facilityProtection_CSF/exprFiles'
        runFilesPath = '/home/hmedal/Documents/2_msu/1_MSU_Projects/facilityProtection_CSF/runFiles' + ending
        runFilesPathName = '/home/hmedal/runFiles/imperfectPro' + ending
        myNumThreads = numThreadsHPC
        databaseName = "/home/hmedal/outputFiles/imperfectPro/impro-results.db"
        
def getDatasetPart(baseline, name, hazardType, numFacs, numDemPts):
    if(baseline):
        return '_base'
    else:
        return 'Daskin' + str(numDemPts) + '_FacPro_p' + str(numFacs) + '_' + hazardType
    
def getInstancePart(baseline, numAllocLevels, numCapLevels, numAllocLevelsActual, numCapLevelsActual, budget):
    if(baseline):
        return '_base'
    else:
        return '_k' + str(numAllocLevels) + '_l' + str(numCapLevels) + '_b' + str(round(budget,2))
    
def getInstancePartMinor(baseline, penaltyMultiplier = 2, excess_capacity = 0.3):
    if(baseline):
        return '_base'
    else:
        return '_pen' + str(penaltyMultiplier) + '_cap' + str(excess_capacity)

def getAlgorithmPart_Greedy(baseline, algType, numSamples, numSampleBunches):
    if(baseline):
        return '_base'
    else:
        return '_type-' + algType + '_s' + str(numSamples) + '_sb' + str(numSampleBunches)

def getAlgorithmPart_SAA(baseline, algType, bendersType, numBunches, secondStageAlg, numSAA_first_stage_probs, numSAA_first_stage_samples, numSAA_second_stage_probs):
    if(baseline):
        return '_base'
    else:
        return '_type-' + algType + '_btype-' + bendersType + '_bunch-' + str(numBunches) + '_ssAlg-' + str(secondStageAlg) + '_m' + str(numSAA_first_stage_probs) + '_n' + str(numSAA_first_stage_samples) + '_n' + str(numSAA_second_stage_probs)
    
def getAlgorithmPart_MeanValue(baseline, algType):
    if(baseline):
        return '_base'
    else:
        return '_type-' + algType

def createExprAndJobFiles_Greedy(algType, paramsTuple, flagsTuple, modelName, numLevels=3):
    index = 0
    datasetBaseline = flagsTuple[index]
    index += 1
    instanceBaseline = flagsTuple[index]
    index += 1
    instanceBaselineMinor = flagsTuple[index]
    index += 1
    algorithmBaseline = flagsTuple[index]
    index = 0
    hazardType = paramsTuple[index]
    index += 1
    numFacs = paramsTuple[index]
    index += 1
    numDemPts = paramsTuple[index]
    index += 1
    numAllocLevels = paramsTuple[index]
    index += 1
    numCapLevels = paramsTuple[index]
    index += 1
    numAllocLevelsActualArray = paramsTuple[index]
    index += 1
    numCapLevelsActualArray = paramsTuple[index]
    index += 1
    budgetMultiplier = paramsTuple[index]
    index += 1
    penaltyMultiplier = paramsTuple[index]
    index += 1
    excess_capacity = paramsTuple[index]
    index += 1
    #numThreads = paramsTuple[index]
    #index += 1
    numSamples = paramsTuple[index]
    index += 1
    numSampleBunches = paramsTuple[index]
    index += 1
    #numSamplesForFinalExponent = paramsTuple[index]
    #index += 1
    #deltaExponent = paramsTuple[index]
    
    root = ET.Element("experimentData")
    
    dataset = ET.SubElement(root, "dataset")
    ET.SubElement(dataset, "path").text = dataPath + 'facLoc/Daskin/Daskin' + str(numDemPts) + '_FacPro_p' + str(numFacs)  + '.xml'
    ET.SubElement(dataset, "name").text = 'd' + str(numDemPts)
    ET.SubElement(dataset, "numFacs").text = str(numFacs)
    
    instance = ET.SubElement(root, "instance")
    ET.SubElement(instance, "numAllocLevels").text = str(numAllocLevels)
    ET.SubElement(instance, "numAllocLevelsActual").text = str(numAllocLevelsActualArray)
    ET.SubElement(instance, "numCapLevels").text = str(numCapLevels)
    ET.SubElement(instance, "numCapLevelsActual").text = str(numCapLevelsActualArray)
    ET.SubElement(instance, "budgetMultiplier").text = str(budgetMultiplier)
    ET.SubElement(instance, "penaltyMultiplier").text = str(penaltyMultiplier)
    ET.SubElement(instance, "excess_capacity").text = str(excess_capacity)
    ET.SubElement(instance, "hazardsPath").text = dataPath + 'facLoc/Daskin/Hazards/hazardsDef_custom_facs' +str(numFacs) + '_levels' +str(numLevels) + hazardFileEndingsMap[hazardType] + '.xml'
    ET.SubElement(instance, "hazardType").text = str(hazardType)
    algorithm = ET.SubElement(root, "algorithm")
    ET.SubElement(algorithm, "numSamples").text = str(numSamples)
    ET.SubElement(algorithm, "numSampleBunches").text = str(numSampleBunches)
    ET.SubElement(algorithm, "timeLimit").text = str(alg_time_limit)
    #ET.SubElement(algorithm, "numSamplesForFinalExponent").text = str(numSamplesForFinalExponent)
    #ET.SubElement(algorithm, "deltaExponent").text = str(deltaExponent)
    ET.SubElement(algorithm, "numThreads").text = str(myNumThreads)
    
    otherSubElement = ET.SubElement(root, "other")
    ET.SubElement(otherSubElement, "databaseName").text = databaseName
    
    tree = ET.ElementTree(root)
    headString = modelName + "_" + getDatasetPart(datasetBaseline, 'Daskin', hazardType, numFacs, numDemPts) + getInstancePart(instanceBaseline, numAllocLevels, numCapLevels, numAllocLevelsActualArray, numCapLevelsActualArray, budgetMultiplier)  + \
        getInstancePartMinor(instanceBaselineMinor) + getAlgorithmPart_Greedy(algorithmBaseline, algType, numSamples, numSampleBunches)
    exprFileName = exprFilePath + '/' + headString + ending + '.xml'
    exprFile = localExprFilePath + '/' + headString  + ending + '.xml'
    tree.write(exprFile)
    createJobFile(algType, exprFileName, headString + "_Job", cluster, runFilesPath + '/' + headString +'_Job.pbs')
    jobFilesList.append(runFilesPathName + '/' + headString +'_Job.pbs')
    
def createExprAndJobFiles_RandomDrawAndProbabilityChain_SAA(algType, paramsTuple, flagsTuple, modelName, numLevels=3):
    index = 0
    datasetBaseline = flagsTuple[index]
    index += 1
    instanceBaseline = flagsTuple[index]
    index += 1
    instanceBaselineMinor = flagsTuple[index]
    index += 1
    algorithmBaseline = flagsTuple[index]
    index = 0
    hazardType = paramsTuple[index]
    index += 1
    numFacs = paramsTuple[index]
    index += 1
    numDemPts = paramsTuple[index]
    index += 1
    numAllocLevels = paramsTuple[index]
    index += 1
    numCapLevels = paramsTuple[index]
    index += 1
    numAllocLevelsActualArray = paramsTuple[index]
    index += 1
    numCapLevelsActualArray = paramsTuple[index]
    index += 1
    budgetMultiplier = paramsTuple[index]
    index += 1
    penaltyMultiplier = paramsTuple[index]
    index += 1
    excess_capacity = paramsTuple[index]
    index += 1
    bendersType = paramsTuple[index]
    index += 1
    numBunches = paramsTuple[index]
    index += 1
    secondStageAlg = paramsTuple[index]
    index += 1
    numSAA_first_stage_probs = paramsTuple[index]
    index += 1
    numSAA_first_stage_samples = paramsTuple[index]
    index += 1
    numSAA_second_stage_probs = paramsTuple[index]
    
    root = ET.Element("experimentData")
    
    dataset = ET.SubElement(root, "dataset")
    ET.SubElement(dataset, "path").text = dataPath + 'facLoc/Daskin/Daskin' + str(numDemPts) + '_FacPro_p' + str(numFacs)  + '.xml'
    ET.SubElement(dataset, "name").text = 'd' + str(numDemPts)
    ET.SubElement(dataset, "numFacs").text = str(numFacs)
    
    instance = ET.SubElement(root, "instance")
    ET.SubElement(instance, "numAllocLevels").text = str(numAllocLevels)
    ET.SubElement(instance, "numAllocLevelsActual").text = str(numAllocLevelsActualArray)
    ET.SubElement(instance, "numCapLevels").text = str(numCapLevels)
    ET.SubElement(instance, "numCapLevelsActual").text = str(numCapLevelsActualArray)
    ET.SubElement(instance, "budgetMultiplier").text = str(budgetMultiplier)
    ET.SubElement(instance, "penaltyMultiplier").text = str(penaltyMultiplier)
    ET.SubElement(instance, "excess_capacity").text = str(excess_capacity)
    ET.SubElement(instance, "hazardsPath").text = dataPath + 'facLoc/Daskin/Hazards/hazardsDef_custom_facs' +str(numFacs) + '_levels' +str(numLevels) + hazardFileEndingsMap[hazardType] + '.xml'
    ET.SubElement(instance, "hazardType").text = str(hazardType)
    algorithm = ET.SubElement(root, "algorithm")
    ET.SubElement(algorithm, "timeLimit").text = str(alg_time_limit)
    ET.SubElement(algorithm, "bendersType").text = str(bendersType)
    ET.SubElement(algorithm, "numBunches").text = str(numBunches)
    ET.SubElement(algorithm, "secondStageAlg").text = str(secondStageAlg)
    ET.SubElement(algorithm, "numSAA_first_stage_probs").text = str(numSAA_first_stage_probs)
    ET.SubElement(algorithm, "numSAA_first_stage_samples").text = str(numSAA_first_stage_samples)
    ET.SubElement(algorithm, "numSAA_second_stage_probs").text = str(numSAA_second_stage_probs)
    
    otherSubElement = ET.SubElement(root, "other")
    ET.SubElement(otherSubElement, "databaseName").text = databaseName
    
    ET.SubElement(algorithm, "numThreads").text = str(myNumThreads)
    
    tree = ET.ElementTree(root)
    headString = modelName + "_" + getDatasetPart(datasetBaseline, 'Daskin', hazardType, numFacs, numDemPts) + getInstancePart(instanceBaseline, numAllocLevels, numCapLevels, numAllocLevelsActualArray, numCapLevelsActualArray, budgetMultiplier)  + \
        getInstancePartMinor(instanceBaselineMinor) + getAlgorithmPart_SAA(algorithmBaseline, algType, bendersType, numBunches, secondStageAlg, numSAA_first_stage_probs, numSAA_first_stage_samples, numSAA_second_stage_probs)
    exprFileName = exprFilePath + '/' + headString + ending + '.xml'
    exprFile = localExprFilePath + '/' + headString  + ending + '.xml'
    tree.write(exprFile)
    createJobFile(algType, exprFileName, headString + "_Job", cluster, runFilesPath + '/' + headString +'_Job.pbs')
    jobFilesList.append(runFilesPathName + '/' + headString +'_Job.pbs')

def createExprAndJobFiles_MeanValue(algType, paramsTuple, flagsTuple, modelName,numLevels=3):
    index = 0
    datasetBaseline = flagsTuple[index]
    index += 1
    instanceBaseline = flagsTuple[index]
    index += 1
    instanceBaselineMinor = flagsTuple[index]
    index += 1
    algorithmBaseline = flagsTuple[index]
    index = 0
    hazardType = paramsTuple[index]
    index += 1
    numFacs = paramsTuple[index]
    index += 1
    numDemPts = paramsTuple[index]
    index += 1
    numAllocLevels = paramsTuple[index]
    index += 1
    numCapLevels = paramsTuple[index]
    index += 1
    numAllocLevelsActualArray = paramsTuple[index]
    index += 1
    numCapLevelsActualArray = paramsTuple[index]
    index += 1
    budgetMultiplier = paramsTuple[index]
    index += 1
    penaltyMultiplier = paramsTuple[index]
    index += 1
    excess_capacity = paramsTuple[index]
    index += 1
    
    root = ET.Element("experimentData")
    
    dataset = ET.SubElement(root, "dataset")
    ET.SubElement(dataset, "path").text = dataPath + 'facLoc/Daskin/Daskin' + str(numDemPts) + '_FacPro_p' + str(numFacs)  + '.xml'
    ET.SubElement(dataset, "name").text = 'd' + str(numDemPts)
    ET.SubElement(dataset, "numFacs").text = str(numFacs)
    
    instance = ET.SubElement(root, "instance")
    ET.SubElement(instance, "numAllocLevels").text = str(numAllocLevels)
    ET.SubElement(instance, "numAllocLevelsActual").text = str(numAllocLevelsActualArray)
    ET.SubElement(instance, "numCapLevels").text = str(numCapLevels)
    ET.SubElement(instance, "numCapLevelsActual").text = str(numCapLevelsActualArray)
    ET.SubElement(instance, "budgetMultiplier").text = str(budgetMultiplier)
    ET.SubElement(instance, "penaltyMultiplier").text = str(penaltyMultiplier)
    ET.SubElement(instance, "excess_capacity").text = str(excess_capacity)
    ET.SubElement(instance, "hazardsPath").text = dataPath + 'facLoc/Daskin/Hazards/hazardsDef_custom_facs' +str(numFacs) + '_levels' +str(numLevels) + hazardFileEndingsMap[hazardType] + '.xml'
    ET.SubElement(instance, "hazardType").text = str(hazardType)
    algorithm = ET.SubElement(root, "algorithm")
    ET.SubElement(algorithm, "timeLimit").text = str(alg_time_limit)
    #ET.SubElement(algorithm, "numSamplesForFinalExponent").text = str(numSamplesForFinalExponent)
    #ET.SubElement(algorithm, "deltaExponent").text = str(deltaExponent)
    ET.SubElement(algorithm, "numThreads").text = str(myNumThreads)
    
    otherSubElement = ET.SubElement(root, "other")
    ET.SubElement(otherSubElement, "databaseName").text = databaseName
    
    tree = ET.ElementTree(root)
    headString = modelName + "_" + getDatasetPart(datasetBaseline, 'Daskin', hazardType, numFacs, numDemPts) + getInstancePart(instanceBaseline, numAllocLevels, numCapLevels, numAllocLevelsActualArray, numCapLevelsActualArray, budgetMultiplier)  + \
        getInstancePartMinor(instanceBaselineMinor) + getAlgorithmPart_MeanValue(algorithmBaseline, algType)
    exprFileName = exprFilePath + '/' + headString + ending + '.xml'
    exprFile = localExprFilePath + '/' + headString  + ending + '.xml'
    tree.write(exprFile)
    createJobFile(algType, exprFileName, headString + "_Job", cluster, runFilesPath + '/' + headString +'_Job.pbs')
    jobFilesList.append(runFilesPathName + '/' + headString +'_Job.pbs')
    
def createJobFile(algType, exprFile, exprName, cluster, outputFile):
    global jobFilesList
    f = open(outputFile, 'w')
    myStr = "#PBS -N " + exprName + "\n"
    myStr += "#PBS -q " + cluster + "\n"
    myStr += "\n"
    myStr += "#PBS -j oe\n"
    myStr += "\n"
    myStr += "#PBS -o " + exprName +".$PBS_JOBID\n"
    myStr += "#PBS -l nodes=1:ppn=" + str(numThreadsHPC) + "\n"
    myStr += "#PBS -l walltime=" + timeString + "\n"
    myStr += "\n"
    myStr += "cd $PBS_O_WORKDIR\n"
    myStr += moduleStatement + "\n"
    myStr += "/home/hmedal/scripts/gurobi.sh " + exeStatement + " --algType " + algType + " --exprfile " + exprFile + " > " + exprFile.replace("xml","log").replace("exprFiles","outputFiles")
    
    f.write(myStr)
    
def createRunScript(scriptFile, exprFiles):
    print "createRunScript"
    f = open(scriptFile, 'w')
    myStr = "#!/bin/sh\n"
    myStr += ". ~/.bashrc"
    myStr += "\n"
    for exprFile in exprFiles:
        print exprFile
        myStr += "/share/apps/torque/bin/qsub " + exprFile + "\n"
    f.write(myStr)
    print "scriptFile", scriptFile

def createFilesForJob_Greedy(algType, tuple, flagsTuple, modelName):
    global jobFilesList
    createExprAndJobFiles_Greedy(algType, tuple, flagsTuple, modelName)

def createFilesForJob_RandomDrawAndProbabilityChain_SAA(algType, tuple, flagsTuple, modelName):
    global jobFilesList
    createExprAndJobFiles_RandomDrawAndProbabilityChain_SAA(algType, tuple, flagsTuple, modelName)
    
def createFilesForJob_ProbabilityChain(algType, tuple, flagsTuple, modelName):
    global jobFilesList
    createExprAndJobFiles_RandomDrawAndProbabilityChain_SAA(algType, tuple, flagsTuple, modelName)
    
def createFilesForJob_MeanValue(algType, tuple, flagsTuple, modelName):
    global jobFilesList
    createExprAndJobFiles_MeanValue(algType, tuple, flagsTuple, modelName)
    
def createFilesForJob_Batch_Greedy(algType, arrays, flagsTuple, modelName, scenario):
    cartProd = itertools.product(*arrays)
    for array in cartProd:
        createFilesForJob_Greedy(algType, array, flagsTuple, modelName)
    createRunScript(runFilesPath + '/scripts/' + modelName + '_' + algType + '_' + scenario + '_script.sh', jobFilesList)
    print "SCRIPT CREATED"
    
def createFilesForJob_Batch_RandomDrawAndProbabilityChain_SAA(algType, arrays, flagsTuple, modelName, scenario):
    print arrays
    cartProd = itertools.product(*arrays)
    for array in cartProd:
        createFilesForJob_RandomDrawAndProbabilityChain_SAA(algType, array, flagsTuple, modelName)
    createRunScript(runFilesPath + '/scripts/' + modelName + '_' + algType + '_' + scenario + '_script.sh', jobFilesList)
    print "SCRIPT CREATED"
    
def createFilesForJob_Batch_MeanValue(algType, arrays, flagsTuple, modelName, scenario):
    print arrays
    cartProd = itertools.product(*arrays)
    for array in cartProd:
        createFilesForJob_MeanValue(algType, array, flagsTuple, modelName)
    createRunScript(runFilesPath + '/scripts/' + modelName + '_' + algType + '_' + scenario + '_script.sh', jobFilesList)
    print "SCRIPT CREATED"
    
def OLD_getModelParamsArray(param, variation):
    if(variation == 'small'):
        if(param == 'numFacs'):
            return [3, 4]
        elif(param == 'numAllocLevelsModeledAndActual'):
            return [[3, 4], [4, 3]]
        elif(param == 'numCapLevelsModeledAndActual'):
            return [[3, 4],  [4, 3]]
    if(variation == 'single'):
        if(param == 'numFacs'):
            return [3, 4, 5, 6]
        elif(param == 'numDemPts'):
            return [49]
        elif(param == 'numAllocLevelsModeledAndActual'):
            return [[3, 3]]
        elif(param == 'numCapLevelsModeledAndActual'):
            return [[3, 3]]
        elif(param == 'budgetMultiplier'):
            return [1.0/3]
    if(variation == 'base'):
        if(param == 'numFacs'):
            return [3, 6]
        elif(param == 'numDemPts'):
            return [49, 150]
        elif(param == 'numAllocLevelsModeledAndActual'):
            return [[3, 5], [5, 3]]
        elif(param == 'numCapLevelsModeledAndActual'):
            return [[3, 5],  [5, 3]]
        elif(param == 'budgetMultiplier'):
            return [1.0/3, 2.0/3]
        elif(param == 'excess_capacity'):
            return [0.2]
    elif(variation == 'problemSize'):
        if(param == 'numFacs'):
            return [9, 12]
        elif(param == 'numDemPts'):
            return [150]
        elif(param == 'numAllocLevelsModeledAndActual'):
            return [[3, 4], [4, 3]]
        elif(param == 'numCapLevelsModeledAndActual'):
            return [[3, 4],  [4, 3]]
    elif(variation == 'large'):
        if(param == 'numAllocLevelsModeledAndActual'):
            return [[6, 6], [8, 8]]
    elif(variation == 'additional'):
        if(param == 'numAllocLevelsModeledAndActual'):
            return [[2, 3], [4, 3], [6, 3]]
        elif(param == 'numCapLevelsModeledAndActual'):
            return [[2, 3], [4, 3], [6, 3]]
        elif(param == 'excess_capacity'):
            return [0.1, 0.3]
    if(variation == 'runTimePaper'):
        if(param == 'numFacs'):
            return [6, 9]
        elif(param == 'numDemPts'):
            return [49,88]
        elif(param == 'numAllocLevelsModeledAndActual'):
            return [[3, 4], [4, 3]]
        elif(param == 'numCapLevelsModeledAndActual'):
            return [[3, 4],  [4, 3]]
        elif(param == 'budgetMultiplier'):
            return [1.0/3, 2.0/3]
    if(variation == 'sensitivityPaper49-6'):
        if(param == 'numFacs'):
            return [6]
        elif(param == 'numDemPts'):
            return [49]
        elif(param == 'numAllocLevelsModeledAndActual'):
            return [[2, 2], [2, 3], [2, 4], [3, 2], [3, 3], [3, 4], [3, 2], [3, 3], [3, 4]]
        elif(param == 'numCapLevelsModeledAndActual'):
            return [[2, 2], [2, 3], [2, 4], [3, 2], [3, 3], [3, 4], [3, 2], [3, 3], [3, 4]]
        elif(param == 'budgetMultiplier'):
            return [1.0/3, 2.0/3]
    if(variation == 'sensitivityPaper49-9'):
        if(param == 'numFacs'):
            return [9]
        elif(param == 'numDemPts'):
            return [49]
        elif(param == 'numAllocLevelsModeledAndActual'):
            return [[2, 2], [2, 3], [2, 4], [3, 2], [3, 3], [3, 4], [3, 2], [3, 3], [3, 4]]
        elif(param == 'numCapLevelsModeledAndActual'):
            return [[2, 2], [2, 3], [2, 4], [3, 2], [3, 3], [3, 4], [3, 2], [3, 3], [3, 4]]
        elif(param == 'budgetMultiplier'):
            return [1.0/3, 2.0/3]
    if(variation == 'sensitivityPaper'):
        if(param == 'numFacs'):
            return [4, 8]
        elif(param == 'numDemPts'):
            return [49, 88]
        elif(param == 'numAllocLevelsModeledAndActual'):
            return [[2, 2], [3, 3], [4, 4]]
        elif(param == 'numCapLevelsModeledAndActual'):
            return [[2, 2], [3, 3], [4, 4]]
        elif(param == 'budgetMultiplier'):
            return [0.25, 0.75]
        
def getModelParamsArray(param, variation):
    if(variation == 'small'):
        if(param == 'numFacs'):
            return [3, 4]
        elif(param == 'numAllocLevelsModeledAndActual'):
            return [[3, 4], [4, 3]]
        elif(param == 'numCapLevelsModeledAndActual'):
            return [[3, 4],  [4, 3]]
    if(variation == 'single'):
        if(param == 'numFacs'):
            return [3, 4, 5, 6]
        elif(param == 'numDemPts'):
            return [49]
        elif(param == 'numAllocLevelsModeledAndActual'):
            return [[3, 3]]
        elif(param == 'numCapLevelsModeledAndActual'):
            return [[3, 3]]
        elif(param == 'budgetMultiplier'):
            return [1.0/3]
    if(variation == 'base'):
        if(param == 'numFacs'):
            return [3, 6]
        elif(param == 'numDemPts'):
            return [49, 150]
        elif(param == 'numAllocLevelsModeledAndActual'):
            return [[3, 5], [5, 3]]
        elif(param == 'numCapLevelsModeledAndActual'):
            return [[3, 5],  [5, 3]]
        elif(param == 'budgetMultiplier'):
            return [1.0/3, 2.0/3]
        elif(param == 'excess_capacity'):
            return [0.1]
        elif(param == 'penaltyMultiplier'):
            return [2.0]
    elif(variation == 'problemSize'):
        if(param == 'numFacs'):
            return [9, 12]
        elif(param == 'numDemPts'):
            return [150]
        elif(param == 'numAllocLevelsModeledAndActual'):
            return [[3, 4], [4, 3]]
        elif(param == 'numCapLevelsModeledAndActual'):
            return [[3, 4],  [4, 3]]
    elif(variation == 'large'):
        if(param == 'numAllocLevelsModeledAndActual'):
            return [[6, 6], [8, 8]]
    elif(variation == 'additional'):
        if(param == 'numAllocLevelsModeledAndActual'):
            return [[2, 3], [4, 3], [6, 3]]
        elif(param == 'numCapLevelsModeledAndActual'):
            return [[2, 3], [4, 3], [6, 3]]
        elif(param == 'excess_capacity'):
            return [0.1, 0.3]
    if(variation == 'runTimePaperOLD'):
        if(param == 'numFacs'):
            return [6, 9]
        elif(param == 'numDemPts'):
            return [49,88]
        elif(param == 'numAllocLevelsModeledAndActual'):
            return [[3, 4], [4, 3]]
        elif(param == 'numCapLevelsModeledAndActual'):
            return [[3, 4],  [4, 3]]
        elif(param == 'budgetMultiplier'):
            return [1.0/3, 2.0/3]
    if(variation == 'runTimePaper'):
        if(param == 'hazardType'):
            return ['conditional', 'allExposed', 'halfExposed']
        elif(param == 'numFacs'):
            return [6, 9]
        elif(param == 'numDemPts'):
            return [49,88]
        elif(param == 'numAllocLevels'):
            return [3, 4]
        elif(param == 'numCapLevels'):
            return [3, 4]
        elif(param == 'numAllocLevelsActual'):
            return [[2]]
        elif(param == 'numCapLevelsActual'):
            return [[2]]
        elif(param == 'budgetMultiplier'):
            return [1.0/3, 2.0/3]
        elif(param == 'penaltyMultiplier'):
            return [2.0]
    if(variation == 'runTimePaperOLD2'):
        if(param == 'hazardType'):
            return ['allExposed', 'halfExposed']
        elif(param == 'numFacs'):
            return [6, 9]
        elif(param == 'numDemPts'):
            return [49,88]
        elif(param == 'numAllocLevels'):
            return [3, 4]
        elif(param == 'numCapLevels'):
            return [3, 4]
        elif(param == 'numAllocLevelsActual'):
            return [[2]]
        elif(param == 'numCapLevelsActual'):
            return [[2]]
        elif(param == 'budgetMultiplier'):
            return [1.0/3, 2.0/3]
        elif(param == 'penaltyMultiplier'):
            return [2.0]
    if(variation == 'sensitivityPaper'):
        if(param == 'hazardType'):
            return ['conditional', 'allExposed', 'halfExposed']
        elif(param == 'numFacs'):
            return [4, 8]
        elif(param == 'numDemPts'):
            return [49, 88]
        elif(param == 'numAllocLevels'):
            return [2, 3, 4]
        elif(param == 'numCapLevels'):
            return [2, 3, 4]
        elif(param == 'numAllocLevelsActual'):
            return [[2, 3, 4]]
        elif(param == 'numCapLevelsActual'):
            return [[2, 3, 4]]
        elif(param == 'budgetMultiplier'):
            return [0.25, 0.75]
        elif(param == 'penaltyMultiplier'):
            return [2.0]
    if(variation == 'sensitivityPaperOLD2'):
        if(param == 'hazardType'):
            return ['allExposed', 'halfExposed']
        elif(param == 'numFacs'):
            return [4, 8]
        elif(param == 'numDemPts'):
            return [49, 88]
        elif(param == 'numAllocLevels'):
            return [2, 3, 4]
        elif(param == 'numCapLevels'):
            return [2, 3, 4]
        elif(param == 'numAllocLevelsActual'):
            return [[2, 3, 4]]
        elif(param == 'numCapLevelsActual'):
            return [[2, 3, 4]]
        elif(param == 'budgetMultiplier'):
            return [0.25, 0.75]
        elif(param == 'penaltyMultiplier'):
            return [2.0]
    if(variation == 'sensitivityPaperOLD'):
        if(param == 'numFacs'):
            return [6]
        elif(param == 'numDemPts'):
            return [49, 88]
        elif(param == 'numAllocLevels'):
            return [2, 3, 4]
        elif(param == 'numCapLevels'):
            return [2, 3, 4]
        elif(param == 'numAllocLevelsActual'):
            return [[2, 3, 4]]
        elif(param == 'numCapLevelsActual'):
            return [[2, 3, 4]]
        elif(param == 'budgetMultiplier'):
            return [1.0/3, 2.0/3]
    
def createFiles_ProblemSize_Greedy(modelName, algType, scenario):
    global jobFilesList
    numSamplesArray = [0]
    numSampleBunchesArray = [1]
    if(scenario == 'excess_cap'):
        numFacsArray = getModelParamsArray('numFacs', 'base')
        numDemPtsArray = getModelParamsArray('numDemPts', 'base')
        numAllocLevelsArray = getModelParamsArray('numAllocLevelsModeledAndActual', 'base')
        numCapLevelsArray = getModelParamsArray('numCapLevelsModeledAndActual', 'base')
        budgetArray = getModelParamsArray('budgetMultiplier', 'base')
        excess_capacityArray = getModelParamsArray('excess_capacity', 'additional')
    elif(scenario == 'single'):
        numFacsArray = getModelParamsArray('numFacs', 'single')
        numDemPtsArray = getModelParamsArray('numDemPts', 'single')
        numAllocLevelsArray = getModelParamsArray('numAllocLevelsModeledAndActual', 'single')
        numCapLevelsArray = getModelParamsArray('numCapLevelsModeledAndActual', 'single')
        budgetArray = getModelParamsArray('budgetMultiplier', 'single')
        excess_capacityArray = getModelParamsArray('excess_capacity', 'base')
    elif(scenario == 'continuousGreedyInitial'):
        numFacsArray = getModelParamsArray('numFacs', 'base')
        numDemPtsArray = getModelParamsArray('numDemPts', 'base')
        numAllocLevelsArray = getModelParamsArray('numAllocLevelsModeledAndActual', 'large')
        numCapLevelsArray = getModelParamsArray('numCapLevelsModeledAndActual', 'base')
        budgetArray = getModelParamsArray('budgetMultiplier', 'base')
        excess_capacityArray = getModelParamsArray('excess_capacity', 'base')
    elif(scenario == 'greedyInitial'):
        numFacsArray = getModelParamsArray('numFacs', 'base')
        numDemPtsArray = getModelParamsArray('numDemPts', 'base')
        numAllocLevelsArray = getModelParamsArray('numAllocLevelsModeledAndActual', 'base')
        numCapLevelsArray = getModelParamsArray('numCapLevelsModeledAndActual', 'base')
        budgetArray = getModelParamsArray('budgetMultiplier', 'base')
        excess_capacityArray = getModelParamsArray('excess_capacity', 'base')
    elif(scenario == 'greedySamplingTest'):
        numFacsArray = getModelParamsArray('numFacs', 'single')
        numDemPtsArray = getModelParamsArray('numDemPts', 'single')
        numAllocLevelsArray = getModelParamsArray('numAllocLevelsModeledAndActual', 'single')
        numCapLevelsArray = getModelParamsArray('numCapLevelsModeledAndActual', 'single')
        budgetArray = getModelParamsArray('budgetMultiplier', 'single')
        excess_capacityArray = getModelParamsArray('excess_capacity', 'base')
        numSamplesArray = [1000]
        numSampleBunchesArray = [10]
    elif(scenario == 'greedyInitialSampling'):
        numFacsArray = getModelParamsArray('numFacs', 'base')
        numDemPtsArray = getModelParamsArray('numDemPts', 'base')
        numAllocLevelsArray = getModelParamsArray('numAllocLevelsModeledAndActual', 'large')
        numCapLevelsArray = getModelParamsArray('numCapLevelsModeledAndActual', 'base')
        budgetArray = getModelParamsArray('budgetMultiplier', 'base')
        excess_capacityArray = getModelParamsArray('excess_capacity', 'base')
        numSamplesArray = [1000]
        numSampleBunchesArray = [10]
    elif(scenario == 'greedyLarger'):
        numFacsArray = getModelParamsArray('numFacs', 'problemSize')
        numDemPtsArray = getModelParamsArray('numDemPts', 'problemSize')
        numAllocLevelsArray = getModelParamsArray('numAllocLevelsModeledAndActual', 'problemSize')
        numCapLevelsArray = getModelParamsArray('numCapLevelsModeledAndActual', 'problemSize')
        budgetArray = getModelParamsArray('budgetMultiplier', 'base')
        excess_capacityArray = getModelParamsArray('excess_capacity', 'base')
    elif(scenario == 'greedyLargerSampling'):
        numFacsArray = getModelParamsArray('numFacs', 'problemSize')
        numDemPtsArray = getModelParamsArray('numDemPts', 'problemSize')
        numAllocLevelsArray = getModelParamsArray('numAllocLevelsModeledAndActual', 'problemSize')
        numCapLevelsArray = getModelParamsArray('numCapLevelsModeledAndActual', 'problemSize')
        budgetArray = getModelParamsArray('budgetMultiplier', 'base')
        excess_capacityArray = getModelParamsArray('excess_capacity', 'base')
        numSamplesArray = [1000, 5000]
        numSampleBunchesArray = [10, 20]
    elif(scenario == 'runTimePaper'):
        hazardTypeArray = getModelParamsArray('hazardType', 'runTimePaper')
        numFacsArray = getModelParamsArray('numFacs', 'runTimePaper')
        numDemPtsArray = getModelParamsArray('numDemPts', 'runTimePaper')
        numAllocLevelsArray = getModelParamsArray('numAllocLevels', 'runTimePaper')
        numCapLevelsArray = getModelParamsArray('numCapLevels', 'runTimePaper')
        numAllocLevelsActualArray = getModelParamsArray('numAllocLevelsActual', 'runTimePaper')
        numCapLevelsActualArray = getModelParamsArray('numCapLevelsActual', 'runTimePaper')
        budgetArray = getModelParamsArray('budgetMultiplier', 'runTimePaper')
        excess_capacityArray = getModelParamsArray('excess_capacity', 'base')
        penaltyMultiplierArray = getModelParamsArray('penaltyMultiplier', 'base')
    elif(scenario == 'sensitivityPaper'):
        hazardTypeArray = getModelParamsArray('hazardType', 'sensitivityPaper')
        numFacsArray = getModelParamsArray('numFacs', 'sensitivityPaper')
        numDemPtsArray = getModelParamsArray('numDemPts', 'sensitivityPaper')
        numAllocLevelsArray = getModelParamsArray('numAllocLevels', 'sensitivityPaper')
        numCapLevelsArray = getModelParamsArray('numCapLevels', 'sensitivityPaper')
        numAllocLevelsActualArray = getModelParamsArray('numAllocLevelsActual', 'sensitivityPaper')
        numCapLevelsActualArray = getModelParamsArray('numCapLevelsActual', 'sensitivityPaper')
        budgetArray = getModelParamsArray('budgetMultiplier', 'sensitivityPaper')
        excess_capacityArray = getModelParamsArray('excess_capacity', 'base')
        penaltyMultiplierArray = getModelParamsArray('penaltyMultiplier', 'base')
    arrays = [hazardTypeArray, numFacsArray, numDemPtsArray, numAllocLevelsArray, numCapLevelsArray, numAllocLevelsActualArray, numCapLevelsActualArray, budgetArray, penaltyMultiplierArray, excess_capacityArray, 
              numSamplesArray, numSampleBunchesArray]
    flagsTuple = [False, False, True, False]
    createFilesForJob_Batch_Greedy(algType, arrays, flagsTuple, modelName, scenario)
    jobFilesList = []
    
def createFiles_ProblemSize_RandomDrawAndProbabilityChain_SAA(modelName, algType, scenario):
    global jobFilesList
    numBunches = [10] # this seems to be the best
    bendersType = ['callback'] # this seems to be the best
    secondStageAlg = ['gurobi']
    penaltyMultiplierArray = [5.0]
    numSAA_first_stage_probs = [10]
    numSAA_first_stage_samples = [100, 1000]
    numSAA_second_stage_samples = [10000]
    if(scenario == 'testSecondStageAlg'):
        numFacsArray = getModelParamsArray('numFacs', 'base')
        numDemPtsArray = getModelParamsArray('numDemPts', 'base')
        numAllocLevelsArray = getModelParamsArray('numAllocLevelsModeledAndActual', 'base')
        numCapLevelsArray = getModelParamsArray('numCapLevelsModeledAndActual', 'base')
        budgetArray = getModelParamsArray('budgetMultiplier', 'base')
        excess_capacityArray = getModelParamsArray('excess_capacity', 'base')
        bendersType = ['classic', 'callback']
        numBunches = [1, 5]
    elif(scenario == 'single'):
        numFacsArray = getModelParamsArray('numFacs', 'single')
        numDemPtsArray = getModelParamsArray('numDemPts', 'single')
        numAllocLevelsArray = getModelParamsArray('numAllocLevelsModeledAndActual', 'single')
        numCapLevelsArray = getModelParamsArray('numCapLevelsModeledAndActual', 'single')
        budgetArray = getModelParamsArray('budgetMultiplier', 'single')
        excess_capacityArray = getModelParamsArray('excess_capacity', 'base')
        bendersType = ['classic', 'callback']
        numBunches = [1,5,10]
    elif(scenario == 'runTime'):
        numFacsArray = getModelParamsArray('numFacs', 'single')
        numDemPtsArray = getModelParamsArray('numDemPts', 'single')
        numAllocLevelsArray = getModelParamsArray('numAllocLevelsModeledAndActual', 'single')
        numCapLevelsArray = getModelParamsArray('numCapLevelsModeledAndActual', 'single')
        budgetArray = getModelParamsArray('budgetMultiplier', 'single')
        excess_capacityArray = getModelParamsArray('excess_capacity', 'base')
        bendersType = ['classic', 'callback']
        numBunches = [1,5,10]
    elif(scenario == 'runTimePaper'):
        hazardTypeArray = getModelParamsArray('hazardType', 'runTimePaper')
        numFacsArray = getModelParamsArray('numFacs', 'runTimePaper')
        numDemPtsArray = getModelParamsArray('numDemPts', 'runTimePaper')
        numAllocLevelsArray = getModelParamsArray('numAllocLevels', 'runTimePaper')
        numCapLevelsArray = getModelParamsArray('numCapLevels', 'runTimePaper')
        numAllocLevelsActualArray = getModelParamsArray('numAllocLevelsActual', 'runTimePaper')
        numCapLevelsActualArray = getModelParamsArray('numCapLevelsActual', 'runTimePaper')
        budgetArray = getModelParamsArray('budgetMultiplier', 'runTimePaper')
        excess_capacityArray = getModelParamsArray('excess_capacity', 'base')
        penaltyMultiplierArray = getModelParamsArray('penaltyMultiplier', 'base')
        bendersType = ['classic']
        numBunches = [1]
        numSAA_first_stage_samples = [0]
    elif(scenario == 'sensitivityPaper'):
        hazardTypeArray = getModelParamsArray('hazardType', 'sensitivityPaper')
        numFacsArray = getModelParamsArray('numFacs', 'sensitivityPaper')
        numDemPtsArray = getModelParamsArray('numDemPts', 'sensitivityPaper')
        numAllocLevelsArray = getModelParamsArray('numAllocLevels', 'sensitivityPaper')
        numCapLevelsArray = getModelParamsArray('numCapLevels', 'sensitivityPaper')
        numAllocLevelsActualArray = getModelParamsArray('numAllocLevelsActual', 'sensitivityPaper')
        numCapLevelsActualArray = getModelParamsArray('numCapLevelsActual', 'sensitivityPaper')
        budgetArray = getModelParamsArray('budgetMultiplier', 'sensitivityPaper')
        penaltyMultiplierArray = getModelParamsArray('penaltyMultiplier', 'base')
        excess_capacityArray = getModelParamsArray('excess_capacity', 'base')
        bendersType = ['classic']
        numBunches = [1]
        numSAA_first_stage_samples = [0]
    elif(scenario == 'sensitivityPaper2'):
        numFacsArray = getModelParamsArray('numFacs', 'sensitivityPaper2')
        numDemPtsArray = getModelParamsArray('numDemPts', 'sensitivityPaper2')
        numAllocLevelsArray = getModelParamsArray('numAllocLevels', 'sensitivityPaper2')
        numCapLevelsArray = getModelParamsArray('numCapLevels', 'sensitivityPaper2')
        numAllocLevelsActualArray = getModelParamsArray('numAllocLevelsActual', 'sensitivityPaper2')
        numCapLevelsActualArray = getModelParamsArray('numCapLevelsActual', 'sensitivityPaper2')
        budgetArray = getModelParamsArray('budgetMultiplier', 'sensitivityPaper2')
        excess_capacityArray = getModelParamsArray('excess_capacity', 'base')
        bendersType = ['classic']
        numBunches = [1]
        numSAA_first_stage_samples = [0]
    elif(scenario == 'comparisonRD'):
        numFacsArray = getModelParamsArray('numFacs', 'small')
        numDemPtsArray = getModelParamsArray('numDemPts', 'base')
        numAllocLevelsArray = getModelParamsArray('numAllocLevelsModeledAndActual', 'small')
        numCapLevelsArray = getModelParamsArray('numCapLevelsModeledAndActual', 'small')
        budgetArray = getModelParamsArray('budgetMultiplier', 'base')
        excess_capacityArray = getModelParamsArray('excess_capacity', 'base')
        bendersType = ['classic', 'callback']
        numBunches = [1, 5, 10]
    elif(scenario == 'probChainCallback'):
        numFacsArray = getModelParamsArray('numFacs', 'base')
        numDemPtsArray = getModelParamsArray('numDemPts', 'base')
        numAllocLevelsArray = getModelParamsArray('numAllocLevelsModeledAndActual', 'base')
        numCapLevelsArray = getModelParamsArray('numCapLevelsModeledAndActual', 'base')
        budgetArray = getModelParamsArray('budgetMultiplier', 'base')
        excess_capacityArray = getModelParamsArray('excess_capacity', 'base')
        bendersType = ['callback']
        numBunches = [15, 20, 25]
    elif(scenario == 'excess_cap'):
        numFacsArray = getModelParamsArray('numFacs', 'base')
        numDemPtsArray = getModelParamsArray('numDemPts', 'base')
        numAllocLevelsArray = getModelParamsArray('numAllocLevelsModeledAndActual', 'base')
        numCapLevelsArray = getModelParamsArray('numCapLevelsModeledAndActual', 'base')
        budgetArray = getModelParamsArray('budgetMultiplier', 'base')
        excess_capacityArray = getModelParamsArray('excess_capacity', 'additional')
        bendersType = ['classic']
        numBunches = [1]
    elif(scenario == 'probChainCallbackSAA'):
        numFacsArray = getModelParamsArray('numFacs', 'problemSize')
        numDemPtsArray = getModelParamsArray('numDemPts', 'problemSize')
        numAllocLevelsArray = getModelParamsArray('numAllocLevelsModeledAndActual', 'problemSize')
        numCapLevelsArray = getModelParamsArray('numCapLevelsModeledAndActual', 'problemSize')
        budgetArray = getModelParamsArray('budgetMultiplier', 'base')
        excess_capacityArray = getModelParamsArray('excess_capacity', 'base')
        numSAA_first_stage_probs = [10, 20]
        numSAA_first_stage_samples = [1000, 5000]
        numSAA_second_stage_samples = [10000, 20000]
    arrays = [hazardTypeArray, numFacsArray, numDemPtsArray, numAllocLevelsArray, numCapLevelsArray, numAllocLevelsActualArray, numCapLevelsActualArray, budgetArray, penaltyMultiplierArray, excess_capacityArray, 
              bendersType, numBunches, secondStageAlg, numSAA_first_stage_probs, numSAA_first_stage_samples, numSAA_second_stage_samples]
    flagsTuple = [False, False, True, False]
    createFilesForJob_Batch_RandomDrawAndProbabilityChain_SAA(algType, arrays, flagsTuple, modelName, scenario)
    jobFilesList = []
    
def createFiles_ProblemSize_MeanValue(modelName, algType, scenario):
    global jobFilesList
    if(scenario == 'vss'):
        hazardTypeArray = getModelParamsArray('hazardType', 'runTimePaper')
        numFacsArray = getModelParamsArray('numFacs', 'runTimePaper')
        numDemPtsArray = getModelParamsArray('numDemPts', 'runTimePaper')
        numAllocLevelsArray = getModelParamsArray('numAllocLevels', 'runTimePaper')
        numCapLevelsArray = getModelParamsArray('numCapLevels', 'runTimePaper')
        numAllocLevelsActualArray = getModelParamsArray('numAllocLevelsActual', 'runTimePaper')
        numCapLevelsActualArray = getModelParamsArray('numCapLevelsActual', 'runTimePaper')
        budgetArray = getModelParamsArray('budgetMultiplier', 'runTimePaper')
        excess_capacityArray = getModelParamsArray('excess_capacity', 'base')
        penaltyMultiplierArray = getModelParamsArray('penaltyMultiplier', 'base')
    arrays = [hazardTypeArray, numFacsArray, numDemPtsArray, numAllocLevelsArray, numCapLevelsArray, numAllocLevelsActualArray, numCapLevelsActualArray, 
              budgetArray, penaltyMultiplierArray, excess_capacityArray]
    flagsTuple = [False, False, True, False]
    createFilesForJob_Batch_MeanValue(algType, arrays, flagsTuple, modelName, scenario)
    jobFilesList = []

def clearOld(clearGreedy, clearRandomDraw, clearProbabilityChain, clearMeanValueProblem):
    if(clearGreedy):
        bashCommand1 = "rm /home/hmedal/Documents/2_msu/1_MSU_Projects/facilityProtection_CSF/runFiles/greedy*.pbs"
        bashCommand2 = "rm /home/hmedal/Documents/2_msu/1_MSU_Projects/facilityProtection_CSF/runFiles/scripts/greedy*.sh"
        bashCommand3 = 'rm /home/hmedal/Documents/2_msu/1_MSU_Projects/facilityProtection_CSF/exprFiles/greedy*.xml'
        os.system(bashCommand1 + '; '+ bashCommand2 + ';' + bashCommand3)
    if(clearRandomDraw):
        bashCommand1 = 'rm /home/hmedal/Documents/2_msu/1_MSU_Projects/facilityProtection_CSF/runFiles/randomDraw*.pbs'
        bashCommand2 = 'rm /home/hmedal/Documents/2_msu/1_MSU_Projects/facilityProtection_CSF/runFiles/scripts/randomDraw*.sh'
        bashCommand3 = 'rm /home/hmedal/Documents/2_msu/1_MSU_Projects/facilityProtection_CSF/exprFiles/randomDraw*.xml'
        os.system(bashCommand1 + '; '+ bashCommand2 + ';' + bashCommand3)
    if(clearProbabilityChain):
        bashCommand1 = 'rm /home/hmedal/Documents/2_msu/1_MSU_Projects/facilityProtection_CSF/runFiles/probabilityChain*.pbs'
        bashCommand2 = 'rm /home/hmedal/Documents/2_msu/1_MSU_Projects/facilityProtection_CSF/runFiles/scripts/probabilityChain*.sh'
        bashCommand3 = 'rm /home/hmedal/Documents/2_msu/1_MSU_Projects/facilityProtection_CSF/exprFiles/probabilityChain*.xml'
        os.system(bashCommand1 + '; '+ bashCommand2 + ';' + bashCommand3)
    if(clearMeanValueProblem):
        bashCommand1 = 'rm /home/hmedal/Documents/2_msu/1_MSU_Projects/facilityProtection_CSF/runFiles/meanVal*.pbs'
        bashCommand2 = 'rm /home/hmedal/Documents/2_msu/1_MSU_Projects/facilityProtection_CSF/runFiles/scripts/meanVal*.sh'
        bashCommand3 = 'rm /home/hmedal/Documents/2_msu/1_MSU_Projects/facilityProtection_CSF/exprFiles/meanVal*.xml'
        os.system(bashCommand1 + '; '+ bashCommand2 + ';' + bashCommand3)
        
clearOld(True, True, True, True)

#myModelAndAlgNames = [[['probabilityChain'],['lshaped']],[['greedy'],['continuous-deterministic']]]
myModelAndAlgNames = [[['probabilityChain'],['lshaped']]]
#myModelAndAlgNames = [[['greedy'],['continuous-deterministic']]]
#myModelAndAlgNames = [[['meanVal'],['bb']]]

#scenarios = ['runTimePaper']
scenarios = ['sensitivityPaper']
#scenarios = ['vss']

debug = False
setQueueInfo(debug)
for myModelAndAlgName, scenario in itertools.product(myModelAndAlgNames, scenarios):
    print myModelAndAlgName
    myModelName = myModelAndAlgName[0][0]
    myAlgType = myModelAndAlgName[1][0]
    print myAlgType, myModelName, scenario
    setPathsAndNumThreads(True, myModelName)
    if(myModelName is 'greedy'):
        createFiles_ProblemSize_Greedy(myModelName, myAlgType, scenario)
    elif(myModelName is 'probabilityChain'):
        createFiles_ProblemSize_RandomDrawAndProbabilityChain_SAA(myModelName, myAlgType, scenario)
    else:
        createFiles_ProblemSize_MeanValue(myModelName, myAlgType, scenario)
    setPathsAndNumThreads(False, myModelName)
    if(myModelName is 'greedy'):
        createFiles_ProblemSize_Greedy(myModelName, myAlgType, scenario)
    elif(myModelName is 'probabilityChain'):
        createFiles_ProblemSize_RandomDrawAndProbabilityChain_SAA(myModelName, myAlgType, scenario)
    else:
        createFiles_ProblemSize_MeanValue(myModelName, myAlgType, scenario)
    print "files created"
#os.system("~/scripts/crImperfectProPython.sh")