import lxml.etree as etree
import numpy as np
from scipy import stats
import gurobipy
import itertools
from multiprocessing import Pool, cpu_count
import time
from scipy.stats import t as tDist
import logging
from src.impro.objects import executableModel
import src.impro.dat.imperfectPro_dataset as imperfectPro_dataset
from src.impro import imperfectPro_model, imperfectPro_problemInstance
import src.impro.meanValueProblem.imperfectPro_meanValueModel as imperfectPro_meanValueModel
import src.impro.io.databaseUtil as dbUtil
import src.impro.myUtil as myutil

counter = 0

secondStageProblem = None
meanValueProblem = None

#values
amtOfPossibleImprovement = 0
objWithoutPro = 0
objValueWithFullProtection = 0
hazardType = None

#dataset
instance = None

#algorithm
numSAA_first_stage_probs = 0
numSAA_first_stage_samples = 0
numSAA_second_stage_samples = 0
time_limit = 0
g_numBunches = 0

#parameters
thetaValues = []

#indices
hazardScenarioIndices = []

#model
gurobiProbabilityChainModel = None
#gur_probChain_secondStageModel = None

#constraints
calcVarsSumConstraints = None
flowBalanceConstraints = None
probChain_VUB_constraints = None

#variables
calcVars = None
allocVars = None
thetaVars = None

#rhs
rhsIndicatorCoefficientTerm = None
rhsIndicatorConstantTerm = None
#other
numScenarios = 0
facilityProbabilitiesForAllocationLevels = None #[j][m][l][k] 
scenarioObjectiveValues = None

#paths and flags
exprFilePath = None
dataFilePath = None
hazardsScenarioDefPath = None
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
useOneProc = False
trace = True
solveProbChainSecondStageByInspection = False

###########################
#### JAVA CODE
###########################
# public double[][] getBasisFromNonNegativePrimals(double[] primalSoln){
#     ArrayList<Integer> basisColumns=new ArrayList<Integer>();
#     for(int i=0;i<primalSoln.length;i++){
#         if(primalSoln[i]>0)
#             basisColumns.add(i);
#     }
#     double[][] basis=new double[arrayA.length][basisColumns.size()];
#     int counter;
#     for(int row=0;row<arrayA.length;row++){
#         counter=0;
#         for(int col=0;col<basisColumns.size();col++){
#             basis[row][col]=arrayA[row][basisColumns.get(counter++)];
#         }
#     }
#     return basis;
# }
# 
# public double[] getBasisCostsFromNonNegativePrimals(double[] primalSoln){
#     ArrayList<Integer> basisColumns=new ArrayList<Integer>();
#     for(int i=0;i<primalSoln.length;i++){
#         if(primalSoln[i]>0)
#             basisColumns.add(i);
#     }
#     double[] basisCosts=new double[basisColumns.size()];
#     int counter=0;
#     for(int col=0;col<basisColumns.size();col++){
#         basisCosts[col]=c[basisColumns.get(counter++)];
#     }
#     return basisCosts;
# }
    
# public double[] solveForDualMultipliers(double[] primalSoln){
#     Matrix basis = new Matrix(getBasisFromNonNegativePrimals(primalSoln));
#     //System.out.println("basis:\n "+MatrixUtil.matrixToStringByRow(basis.getArray()));
#     double[] basisCosts=getBasisCostsFromNonNegativePrimals(primalSoln);
#     Matrix basisCostsMatrix = new Matrix(basisCosts,basisCosts.length);
#     //System.out.println("basisCostsMatrix:\n"+MatrixUtil.matrixToStringByRow(basisCostsMatrix.getArray()));
#     Matrix dualMultipliersMatrix=(basisCostsMatrix.transpose()).times(basis.inverse());
#     return dualMultipliersMatrix.getColumnPackedCopy();
# }
# public double[] getCalVarsSolutionFromFirstStageSoln(){
#     double[] calcVarsSoln=new double[numLocs*numAllocLevels*2];
#     int counter=0;
#     for(int level=0;level<numLocs;level++){
#         for(int allocLevel=0;allocLevel<numAllocLevels;allocLevel++){
#             calcVarsSoln[counter]=scenarioObjectiveValues[scenarioIndex]*
#             getCumulativeFailureProbability(level)*firstStageVector[level][allocLevel];
#             //System.out.println(scenarioObjectiveValues[scenarioIndex]
#             //                              +" "+getCumulativeFailureProbability(level)+" "+firstStageVector[level][allocLevel]+" "+calcVarsSoln[counter]);
#             counter++;
#         }
#     }
#     for(int level=0;level<numLocs;level++){
#         for(int allocLevel=0;allocLevel<numAllocLevels;allocLevel++){
#             calcVarsSoln[counter]=scenarioObjectiveValues[scenarioIndex]*
#             (getCumulativeMaxProbability(level)-getCumulativeFailureProbability(level))*firstStageVector[level][allocLevel];
#             counter++;
#         }
#     }
#     return calcVarsSoln;
# }

# def getProbabilityForFacilityState(fac, firstStageVector):
#     mySum=0
#     for allocLevel in range(instance.numAllocLevels):
#         mySum += facilityProbabilitiesForAllocationLevels[fac][allocLevel]*firstStageVector[fac][allocLevel]
#     return mySum
#     
# def getCumFailProb(level):
#         coef=1.0
#         for index in range(level):
#             coef *= getProbabilityForFacilityState(index)
#         return coef
# 
# def getCumMaxProb(level):
#         coef=1.0
#         for index in range(level):
#             coef *= getMaxFacProbOverAllAllocLevels(index)
#         return coef
    
# def getMaxFacProbOverAllAllocLevels(fac):
#     max = -float('inf')
#     for allocLevel in range(instance.numAllocLevels):
#         if(facilityProbabilitiesForAllocationLevels[fac][allocLevel]>max):
#             max = facilityProbabilitiesForAllocationLevels[fac][allocLevel]
#     return max
# 
# def getCalVarsSolutionFromFirstStageSoln(firstStageVector, scenarioIndex):
#     calcVarsSoln = []
#     counter=0
#     for level in range(instance.numFacs):
#         for allocLevel in range(instance.numAllocLevels):
#             #check if I should multiply by scenObjValue calcVarsSoln[counter] = scenarioObjectiveValues[scenarioIndex] * getCumFailProb(level)*firstStageVector[level][allocLevel]
#             counter += 1
#     for l in range(instance.numFacs):
#         for allocLevel in range(instance.numAllocLevels):
#            calcVarsSoln[counter] = scenarioObjectiveValues[scenarioIndex] * (getCumMaxProb(l)-getCumFailProb(l)) * firstStageVector[l][allocLevel]
#            counter += 1
#     return calcVarsSoln;
# 
# def getBasisFromNonNegativePrimals(aMatrix, primalSoln):
#     basisColumns = []
#     for i in range(len(primalSoln)):
#         if(primalSoln[i] > 0):
#             basisColumns.append(i)
#     basis = []
#     counter = 0
#     for row in range(len(aMatrix)):
#         counter = 0
#         for col in range(len(basisColumns)):
#             basis[row][col] = aMatrix[row][basisColumns[counter]]
#             counter = counter + 1
#     return basis
# 
# def getBasisCostsFromNonNegativePrimals(costsVector, primalSoln):
#     basisColumns = []
#     for i in range(len(primalSoln)):
#         if(primalSoln[i] > 0):
#             basisColumns.append(i)
#     basisCosts = []
#     counter = 0
#     for col in range(len(basisColumns)):
#         basisCosts[col] = costsVector[basisColumns[counter]]
#         counter = counter + 1
#     return basisCosts

# def solveForDualMultipliers(primalSoln):
#     basis = numpy.array(getBasisFromNonNegativePrimals(primalSoln))
#     basisCosts=getBasisCostsFromNonNegativePrimals(primalSoln)
#     basisCostsMatrix = numpy.array(basisCosts)
#     dualMultipliersMatrix = basisCostsMatrix.T * scipy.linalg.inv(basis)
#     return dualMultipliersMatrix

def my_computeSecondStageUtility(capacityLevels):
    return secondStageProblem.computeSecondStageUtility(capacityLevels, instance.numCapLevels)

def my_computeSecondStageUtility_AltCapLevels(capacityLevels, myNumCapLevels):
    return secondStageProblem.computeSecondStageUtility(capacityLevels, myNumCapLevels)

def my_computeSecondStageUtilityAndGetSolution(capacityLevels):
    return secondStageProblem.computeSecondStageUtilityAndSolution(capacityLevels, instance.numCapLevels)
    
def createFacilityProbabilitiesForAllocationLevels():
    global facilityProbabilitiesForAllocationLevels
    facilityProbabilitiesForAllocationLevels = [[[imperfectPro_model.getCapLevelProbabilityForAllocationLevel(j,l,k, instance, instance.numCapLevels) 
                                                  for k in range(instance.numAllocLevels)] for l in range(instance.numCapLevels)] 
                                                    for j in range(instance.numFacs)]
    
def createFacilityProbabilitiesForAllocationLevels_withHazards():
    global facilityProbabilitiesForAllocationLevels
    facilityProbabilitiesForAllocationLevels = [[[[imperfectPro_model.getCapLevelProbabilityForAllocationLevelAndHazardLevel(j,l,k, m, instance, instance.numCapLevels) 
                                                  for k in range(instance.numAllocLevels)] for l in range(instance.numCapLevels)] 
                                                    for m in range(instance.numHazardLevels)] for j in range(instance.numFacs)]
    
def createScenarios():
    cartProd = itertools.product(range(instance.numCapLevels), repeat = instance.numFacs)
    cartProdList = [list(i) for i in cartProd]
    localScenariosSet = [[myutil.createSparseList(instance.numCapLevels, capLevel) for capLevel in sublist] for sublist in cartProdList]
    global numScenarios
    numScenarios = len(localScenariosSet)
    return localScenariosSet
        
def createScenarios_withHazards(myNumCapLevels, myNumHazardLevels):
    cartProd = itertools.product(range(myNumCapLevels), repeat = instance.numFacs*myNumHazardLevels)
    cartProdList = [list(i) for i in cartProd]
    #print "cartProdList: ", cartProdList
    cartProdList_divided = [myutil.grouper_asList(list(i), myNumHazardLevels) for i in cartProdList]
    #print "cartProdList_divided: ", cartProdList_divided
    #localScenariosSet = [[myutil.createSparseList(myNumCapLevels, capLevel) for capLevel in sublist] for sublist in cartProdList]
    localScenariosSet_withHazards = []
    for sublist in cartProdList_divided:
        localScenariosSet_withHazards.append([[myutil.createSparseList(myNumCapLevels, capLevel) for capLevel in subsublist] for subsublist in sublist])
    #print "localScenariosSet_withHazards: ", localScenariosSet_withHazards
    global numScenarios
    numScenarios = len(localScenariosSet_withHazards)
    return localScenariosSet_withHazards

def createScenarios_withHazardsFromFile():
    if(trace):
        print "createScenarios_withHazardsFromFile"
    allScenarios = []
    hazardScenarioIndex = 0
    for scenario in instance.scenarioHazardLevels:
        cartProd = itertools.product(range(instance.numCapLevels), repeat = instance.numFacs)
        cartProdList = [list(i) for i in cartProd]
        for myTuple in cartProdList:
            allScenarios.append([[scenario[facIndex], myutil.createSparseList(instance.numCapLevels, myTuple[facIndex])] 
                              for facIndex in range(instance.numFacs)])
            hazardScenarioIndices.append(hazardScenarioIndex)
        hazardScenarioIndex += 1
    #print "allScenarios: ", allScenarios
    #print "hazardScenarioIndices: ", hazardScenarioIndices
    global numScenarios
    numScenarios = len(allScenarios)
    print "instance.numCapLevels", instance.numCapLevels
    print "numScenarios: createScenarios_withHazardsFromFile", numScenarios
    print "allScenarios[0]", allScenarios[0]
    return allScenarios

def computScenarioObjectiveValues_SERIES(scenariosSet):
    if(trace):
        print "computScenarioObjectiveValues"
    #print "scenariosSet: ", str(scenariosSet)
    #print "computScenarioObjectiveValues AFTER LOGGING"
    scenarioIndicatorsConvertedToCapLevels = [[sum([a*b for a,b in zip(scenario[j],range(instance.numCapLevels))]) for j in range(instance.numFacs)] for scenario in scenariosSet]
    if(trace):
        print "scenarioIndicatorsConvertedToCapLevels: " + str(len(scenarioIndicatorsConvertedToCapLevels))
    output = [my_computeSecondStageUtility(capLevelsVector)[1] for capLevelsVector in scenarioIndicatorsConvertedToCapLevels]
    if(trace):
        print "finished computing"
    return output

def computScenarioObjectiveValues(scenariosSet):
    if(trace):
        print "computScenarioObjectiveValues"
    #print "scenariosSet: ", str(scenariosSet)
    #print "computScenarioObjectiveValues AFTER LOGGING"
    scenarioIndicatorsConvertedToCapLevels = [[sum([a*b for a,b in zip(scenario[j],range(instance.numCapLevels))]) for j in range(instance.numFacs)] for scenario in scenariosSet]
    if(trace):
        print "scenarioIndicatorsConvertedToCapLevels: " + str(len(scenarioIndicatorsConvertedToCapLevels))
        #print "numThreads", numThreads
    if(debug):
        localPoolVar = Pool(1)
    else:
        print "set pool 12"
        localPoolVar = Pool()
        print "pool set"
    if(trace):
        print "check 1"
    if(trace):
        print "numProcs:", localPoolVar._processes
    if(trace):
        print "check 2"
    output = localPoolVar.map(my_computeSecondStageUtility, scenarioIndicatorsConvertedToCapLevels,1)
    #print "scenObjVals", output
    localPoolVar.close()
    localPoolVar.join()
    #print 'computScenarioObjectiveValues finished'
    #[my_computeSecondStageUtility(scenario) for scenario in scenarioIndicatorsConvertedToCapLevels]
    return output

def computScenarioObjectiveValues_withHazards(scenariosSet):
    if(trace):
        print "computScenarioObjectiveValues_withHazards"
    capsIndicatorsScenarios = instance.convertCapsAndHazardsScenarios_to_CapsScenarios(scenariosSet)
    #print "computScenarioObjectiveValues_withHazards - capsIndicatorsScenarios", capsIndicatorsScenarios
    logging.info("scenariosSet: " + str(capsIndicatorsScenarios))
    #print "computScenarioObjectiveValues AFTER LOGGING"
    return computScenarioObjectiveValues(capsIndicatorsScenarios)

def createProbabilityChainDiscreteEquivalentModel(scenariosSet):
    global gurobiProbabilityChainModel,allocVars, calcVars, scenarioObjectiveValues
    scenarioObjectiveValues = computScenarioObjectiveValues(scenariosSet)
    gurobiProbabilityChainModel = gurobipy.Model("DiscreteEquivalent- Probability Chain")
    try:
        # Create first stage variables and constraints
        allocVars = [[gurobiProbabilityChainModel.addVar(0,1,vtype = gurobipy.GRB.BINARY,name="y_"+str(j)+","+str(k)) for k in range(instance.numAllocLevels)] for j in range(instance.numFacs)]
        calcVars = [[[gurobiProbabilityChainModel.addVar(0,gurobipy.GRB.INFINITY,vtype = gurobipy.GRB.CONTINUOUS,name="w_"+str(r)+","+str(k)+","+str(scenNumber)) for k in range(instance.numAllocLevels)]
                     for r in range(instance.numFacs)] for scenNumber in range(len(scenariosSet))]
        gurobiProbabilityChainModel.update()
        gurobiProbabilityChainModel.setObjective(sum([ calcVars[scenNumber][instance.numFacs-1][k] for k in range(instance.numAllocLevels) for scenNumber in range(len(scenariosSet))]), gurobipy.GRB.MAXIMIZE)
        #calcVarsSum == 1
        [gurobiProbabilityChainModel.addConstr(sum([calcVars[scenNumber][0][k] for k in range(instance.numAllocLevels)]) == scenarioObjectiveValues[scenNumber], 
                                         "calcVarsSum == 1 s"+str(scenNumber)) for scenNumber in range(len(scenariosSet))]
        #flow balance constraints
        [gurobiProbabilityChainModel.addConstr(sum([facilityProbabilitiesForAllocationLevels[r][l][k] * scenariosSet[scenNumber][r][l] * calcVars[scenNumber][r-1][k] 
                                              for l in range(instance.numCapLevels) for k in range(instance.numAllocLevels)]) 
                                         == 
            sum([calcVars[scenNumber][r][k] for k in range(instance.numAllocLevels)]), 
            "flowBalance_r"+str(r)+" s"+str(scenNumber))  
            for r in range(1,instance.numFacs) for scenNumber in range(len(scenariosSet))]
        #calcVar VUB constraints
        [[gurobiProbabilityChainModel.addConstr(calcVars[scenNumber][r][k] <= scenarioObjectiveValues[scenNumber] * allocVars[r][k], "calcVar <= allocVar r"+str(r) + "_k"+str(k) 
                                          + " s" + str(scenNumber)) for k in range(instance.numAllocLevels)] for r in range(instance.numFacs) for scenNumber in range(len(scenariosSet))]
        # alloc levels constraint
        [gurobiProbabilityChainModel.addConstr(sum([allocVars[j][k] for k in range(instance.numAllocLevels)]) == 1, "one alloc level_j" + str(j)) for j in range(instance.numFacs)]
        #instance.budget constraint
        gurobiProbabilityChainModel.addConstr(sum([costOfAlloc[j][k] * allocVars[j][k] for k in range(instance.numAllocLevels) for j in range(instance.numFacs)]) <= instance.budget, "allocation instance.budget")
        gurobiProbabilityChainModel.update()
        gurobiProbabilityChainModel.setParam('OutputFlag', True )
    except gurobipy.GurobiError as e:
        print 'createProbabilityChainDiscreteEquivalentModel: Gurobi Error reported' + e
            
def solveProbabilityChainDeterministicEquivalentModel():
    gurobiProbabilityChainModel.optimize()
    allocVarSoln = [[allocVars[j][k].getAttr("X") for k in range(instance.numAllocLevels)] for j in range(instance.numFacs)]
    return allocVarSoln
    
def createProbabilityChainSecondStageModel_GRB():
    #global gur_probChain_secondStageModel
    global calcVars
    global calcVarsSumConstraints, flowBalanceConstraints, probChain_VUB_constraints
    local_gur_probChain_secondStageModel = gurobipy.Model("myLP")
    try:
        #create variables
        calcVars = [[local_gur_probChain_secondStageModel.addVar(0,gurobipy.GRB.INFINITY,vtype = gurobipy.GRB.CONTINUOUS,name="w_"+str(r)+","+str(k)) 
                      for k in range(instance.numAllocLevels)]
                     for r in range(instance.numFacs)]
        local_gur_probChain_secondStageModel.update()
        #create objective
        local_gur_probChain_secondStageModel.setObjective(sum([ calcVars[instance.numFacs-1][k] for k in range(instance.numAllocLevels)]), gurobipy.GRB.MAXIMIZE)
        local_gur_probChain_secondStageModel.update()
        #calcVarsSum == 1
        calcVarsSumConstraints = [local_gur_probChain_secondStageModel.addConstr(sum([calcVars[0][k] for k in range(instance.numAllocLevels)]) == 1, 
                                         "calcVarsSum == 1")]
        #flow balance constraints
        flowBalanceConstraints = [local_gur_probChain_secondStageModel.addConstr(sum([facilityProbabilitiesForAllocationLevels[r][l][k] * 
                                                                                       calcVars[r-1][k] 
                                              for l in range(instance.numCapLevels) for k in range(instance.numAllocLevels)]) 
                                         == 
            sum([calcVars[r][k] for k in range(instance.numAllocLevels)]), 
            "flowBalance_r"+str(r)) for r in range(1,instance.numFacs)]
        local_gur_probChain_secondStageModel.update()
        #calcVar VUB constraints
        probChain_VUB_constraints = [[local_gur_probChain_secondStageModel.addConstr(calcVars[r][k] <= 0, 
                                                                             "calcVar <= allocVar r"+str(r) + "_k"+str(k)) 
            for k in range(instance.numAllocLevels)] for r in range(instance.numFacs)]
        local_gur_probChain_secondStageModel.update()
        local_gur_probChain_secondStageModel.setParam('OutputFlag', False )
    except gurobipy.GurobiError as e:
        print 'createProbabilityChainSecondStageModel_GRB: Gurobi Error reported' + e
    return local_gur_probChain_secondStageModel

def createProbabilityChainSecondStageModel_GRB_withHazards():
    #global gur_probChain_secondStageModel
    global calcVars
    global calcVarsSumConstraints, flowBalanceConstraints, probChain_VUB_constraints
    local_gur_probChain_secondStageModel = gurobipy.Model("myLP")
    try:
        #create variables
        calcVars = [[local_gur_probChain_secondStageModel.addVar(0,gurobipy.GRB.INFINITY,vtype = gurobipy.GRB.CONTINUOUS,name="w_"+str(r)+","+str(k)) 
                      for k in range(instance.numAllocLevels)]
                     for r in range(instance.numFacs)]
        local_gur_probChain_secondStageModel.update()
        #create objective
        local_gur_probChain_secondStageModel.setObjective(sum([ calcVars[instance.numFacs-1][k] for k in range(instance.numAllocLevels)]), gurobipy.GRB.MAXIMIZE)
        local_gur_probChain_secondStageModel.update()
        #calcVarsSum == 1
        calcVarsSumConstraints = [local_gur_probChain_secondStageModel.addConstr(sum([calcVars[0][k] for k in range(instance.numAllocLevels)]) == 1, 
                                         "calcVarsSum == 1")]
        #flow balance constraints
        flowBalanceConstraints = [local_gur_probChain_secondStageModel.addConstr(sum([facilityProbabilitiesForAllocationLevels[r][0][l][k] * 
                                                                                       calcVars[r-1][k] 
                                              for l in range(instance.numCapLevels) for k in range(instance.numAllocLevels)]) 
                                         == 
            sum([calcVars[r][k] for k in range(instance.numAllocLevels)]), 
            "flowBalance_r"+str(r)) for r in range(1,instance.numFacs)]
        local_gur_probChain_secondStageModel.update()
        #calcVar VUB constraints
        probChain_VUB_constraints = [[local_gur_probChain_secondStageModel.addConstr(calcVars[r][k] <= 0, 
                                                                             "calcVar <= allocVar r"+str(r) + "_k"+str(k)) 
            for k in range(instance.numAllocLevels)] for r in range(instance.numFacs)]
        local_gur_probChain_secondStageModel.update()
        local_gur_probChain_secondStageModel.setParam('OutputFlag', False )
    except gurobipy.GurobiError as e:
        print 'createProbabilityChainSecondStageModel_GRB: Gurobi Error reported' + e
    return local_gur_probChain_secondStageModel

def createProbabilityChainSecondStageModel_GRB_ForAllScenarios(numScenarios):
    global gur_probChain_secondStageModels
    gur_probChain_secondStageModels = []
    for scen in range(numScenarios):
        gur_probChain_secondStageModels.append(createProbabilityChainSecondStageModel_GRB())
        
def createProbabilityChainSecondStageModel_GRB_ForAllScenarios_withHazards(numScenarios):
    if(trace):
        print "createProbabilityChainSecondStageModel_GRB_ForAllScenarios_withHazards", numScenarios
    global gur_probChain_secondStageModels
    gur_probChain_secondStageModels = []
    for scen in range(numScenarios):
        #print "createProbabilityChainSecondStageModel_GRB_withHazards", scen
        gur_probChain_secondStageModels.append(createProbabilityChainSecondStageModel_GRB_withHazards())
    
def calculateFacilityStateProbabilities(scen):
        return [[sum(facilityProbabilitiesForAllocationLevels[j][l][k] * scenariosSet[scen][j][l] 
                     for l in range(instance.numCapLevels)) 
                                               for k in range(instance.numAllocLevels)] for j in range(instance.numFacs)]
        
def calculateFacilityStateProbabilities_withHazards(scen):
    local_array = []
    for j in range(instance.numFacs):
        local_array.append([])
        #print "j", j, instance.numAllocLevels
        #print [scenariosSet[scen][j][1] for l in range(instance.numCapLevels)]
        #print "facilityProbabilitiesForAllocationLevels[j]", facilityProbabilitiesForAllocationLevels[j]
        for k in range(instance.numAllocLevels):
            #print "j", j, k
            hazardLevelForFac = scenariosSet[scen][j][0]
            #print "hazardLevelForFac", hazardLevelForFac
            #for l in range(instance.numCapLevels):
                #print "scenariosSet[scen][j][1][l]", l, scenariosSet[scen][j][1][l]
                #print "facilityProbabilitiesForAllocationLevels", facilityProbabilitiesForAllocationLevels[j][hazardLevelForFac][l][k]
            value = sum([facilityProbabilitiesForAllocationLevels[j][hazardLevelForFac][l][k] 
                                        * scenariosSet[scen][j][1][l] for l in range(instance.numCapLevels)])
            #print "value", value
            local_array[j].append(value)
    #print "local_array", local_array
    return local_array
    
def getCalcVarsSumConstraint(model):
    return model.getConstrs()[0]

def getCalcVars(model):
    myVarsFlat = model.getVars()
    return [[myVarsFlat[instance.numAllocLevels*r + k] for k in range(instance.numAllocLevels)] for r in range(instance.numFacs)]

def getProbChain_VUB_constraints(model):
    myConstraintsFlat = model.getConstrs()[1 + (instance.numFacs-1):]
    return [[myConstraintsFlat[instance.numAllocLevels*r + k] for k in range(instance.numAllocLevels)] for r in range(instance.numFacs)]

def resetProbabilityChainSecondStageModel(scenNumber, firstStageSoln, model):
    #print "resetProbabilityChainSecondStageModel", hazardScenarioIndices
    #print "reset", "len(scenarioObjectiveValues)", len(scenarioObjectiveValues)
    hazardScenario = hazardScenarioIndices[scenNumber]
    getCalcVarsSumConstraint(model).setAttr("rhs", scenarioObjectiveValues[scenNumber])
    #print "after get constraints"
    if(includeHazards):
        #print "include hazards"
        facilityStateProbabilities = calculateFacilityStateProbabilities_withHazards(scenNumber)
    else:
        facilityStateProbabilities = calculateFacilityStateProbabilities(scenNumber)
    #print "facilityStateProbabilities", facilityStateProbabilities
    myCalcVars = getCalcVars(model)
    for k in range(instance.numAllocLevels):
        myCalcVars[instance.numFacs - 1][k].setAttr("Obj", float(facilityStateProbabilities[instance.numFacs - 1][k] * instance.scenarioProbs[hazardScenario]))
    for r in range(1,instance.numFacs):
        for k in range(instance.numAllocLevels):
            model.chgCoeff(flowBalanceConstraints[r-1], calcVars[r-1][k], facilityStateProbabilities[r-1][k])
    myProbChain_VUB_constraints = getProbChain_VUB_constraints(model)
    #print "myProbChain_VUB_constraints", myProbChain_VUB_constraints
    for j in range(instance.numFacs):
        for k in range(instance.numAllocLevels):
            #print myProbChain_VUB_constraints[j][k], firstStageSoln[j][k]
            myProbChain_VUB_constraints[j][k].setAttr("rhs", scenarioObjectiveValues[scenNumber] * firstStageSoln[j][k])
    #print "before update"
    model.update()

def solveProbabilityChainSecondStageModel(scenNumber, firstStageSoln):
    #print "firstStageSoln", firstStageSoln, scenNumber
    #global second_stage_counter
    if(reset_mode):
        local_model = gur_probChain_secondStageModel
    else:
        local_model = gur_probChain_secondStageModels[scenNumber]
    try:
        #print "before reset"
        resetProbabilityChainSecondStageModel(scenNumber, firstStageSoln, local_model)
        #second_stage_counter += 1
        #print "before optimize"
        local_model.optimize()
        #print "optimized"
        calcVarsSumDuals = [local_model.getConstrs()[0].getAttr("Pi")]
        vubDuals = [constraint.getAttr("Pi") for constraint in local_model.getConstrs()[(1 + (instance.numFacs-1)):]]
        #print "before return", local_model.objVal, [calcVarsSumDuals, vubDuals]
        return local_model.objVal, [calcVarsSumDuals, vubDuals]
    except gurobipy.GurobiError as e:
        print "model status: ", local_model.getAttr(gurobipy.GRB.Attr.Status)
        print 'solveProbabilityChainSecondStageModel: Gurobi Error reported: ' + str(e)

def solveProbabilityChainSecondStageModelTuple(array):
    #print "solveProbabilityChainSecondStageModelTuple", array
    return solveProbabilityChainSecondStageModel(array[0], array[1])

def getScenariosOutput(firstStageSoln, usePool):
    tuples = [[scenarioIndex, firstStageSoln] for scenarioIndex in range(len(scenariosSet))]
    #print "firstStageSoln", firstStageSoln
    #print "scenariosSet", scenariosSet
    #print "tuples", tuples
    if(usePool):
        if(useOneProc):
            po = Pool(1)
        else:
            po = Pool()
        #res = po.map_async(solveProbabilityChainSecondStageModelTuple,tuples)
        output = po.map_async(solveProbabilityChainSecondStageModelTuple, tuples).get()
        #print "output", output
        #output = res.get()
        #localPoolVar = Pool(processes=numThreads)
        #output = localPoolVar.map(solveProbabilityChainSecondStageModelTuple, tuples)
        #localPoolVar.terminate()
        po.close()
        po.terminate()
        po.join()
    else:
        #print "no tuple"
        output = [solveProbabilityChainSecondStageModelTuple(tuple) for tuple in tuples]
    return output

def solveProbabilityChainSecondStageModelForAllScenarios(firstStageSoln, usePool=True):
    #print "facilityProbabilitiesForAllocationLevels", facilityProbabilitiesForAllocationLevels
    #for facIndex in range(instance.numFacs):
    #    print facilityProbabilitiesForAllocationLevels[facIndex]    
    output = getScenariosOutput(firstStageSoln, usePool)
    #print "objVals:", [list[0] for list in output]
    #print "objValAvg:", np.mean([list[0] for list in output])
    #scenarioProbs = [output[scenarioIndex][0]/scenarioObjectiveValues[scenarioIndex] for scenarioIndex in range(len(scenariosSet))]
    #print "scenObjValAvg:", np.mean([scenarioObjectiveValues[scenarioIndex] for scenarioIndex in range(len(scenariosSet))])
    #print "scenarioProbs", scenarioProbs
    #print "sumScenarioProbs", sum(scenarioProbs), sum(scenarioProbs)*0.243
    totalObjVal = sum([list[0] for list in output])
    calcVarsSumDuals = [list[1][0] for list in output]
    vubDualsFlat = [list[1][1]  for list in output]
    vubDuals = [[[vubDualsFlat[scenarioIndex][j*instance.numAllocLevels + k] for k in range(instance.numAllocLevels)] for j in range(instance.numFacs)] for scenarioIndex in range(len(scenariosSet))]
    constantDualSums = [sum([scenarioObjectiveValues[scenarioIndex] *  calcVarsSumDuals[scenarioIndex][0]]) for scenarioIndex in range(len(scenariosSet))]
    coefficientDualTerms = [[[scenarioObjectiveValues[s] * vubDuals[s][j][k] for k in range(instance.numAllocLevels)] for j in range(instance.numFacs)] for s in range(len(scenariosSet))]
    
    coefficientDualTermsSum = sum([sum([firstStageSoln[j][k]*coefficientDualTerms[s][j][k] for k in range(instance.numAllocLevels) for j in range(instance.numFacs)]) for s in range(len(scenariosSet))])
    #print "sum(coefficientDualTerms)", coefficientDualTermsSum
    #print "totalDuals", sum(constantDualSums) + coefficientDualTermsSum
    #print "totalObjVals solveProbabilityChainSecondStageModelForAllScenarios", totalObjVal
    return totalObjVal, coefficientDualTerms, constantDualSums

def createProbabilityChainMasterProblemModel_SingleCut_GRB(initialUpperBound, showOutput = False):
    global gurobiProbabilityChainModel, allocVars, thetaVars
    gurobiProbabilityChainModel = gurobipy.Model("master problem single cut")
    try:
        allocVars = [[gurobiProbabilityChainModel.addVar(0,1,vtype = gurobipy.GRB.BINARY,name="y_"+str(j)+","+str(k)) for k in range(instance.numAllocLevels)] for j in range(instance.numFacs)]
        thetaVars = [gurobiProbabilityChainModel.addVar(0, initialUpperBound,vtype = gurobipy.GRB.CONTINUOUS,name="theta")]
        gurobiProbabilityChainModel.update()
        gurobiProbabilityChainModel.setObjective(thetaVars[0], gurobipy.GRB.MAXIMIZE)

        gurobiProbabilityChainModel.update()
        # alloc levels constraint
        [gurobiProbabilityChainModel.addConstr(sum([allocVars[j][k] for k in range(instance.numAllocLevels)]) == 1, "one alloc level_" + str(j)) for j in range(instance.numFacs)]
        gurobiProbabilityChainModel.update()
        # instance.budget constraint
        gurobiProbabilityChainModel.addConstr(sum([costOfAlloc[j][k] * allocVars[j][k] for k in range(instance.numAllocLevels) for j in range(instance.numFacs)]) <= instance.budget, "allocation instance.budget")
        gurobiProbabilityChainModel.update()
        gurobiProbabilityChainModel.setParam('OutputFlag', showOutput)
    except gurobipy.GurobiError as e:
        print 'createProbabilityChainMasterProblemModel_SingleCut_GRB: Gurobi Error reported ' + e

def createProbChainMasterProblemModel_MultiCut_GRB(initialUpperBound, numBunches, showOutput = False):
    global gurobiProbabilityChainModel, allocVars, thetaVars
    if(trace):
        print "CREATE PROB CHAIN MASTER PROBLEM MULTICUT"
    numScenarios = len(scenariosSet)
    bunchSize = numScenarios/numBunches
    #print "numBunches", numBunches
    remainder = numScenarios%numBunches
    #print "numBunches", numBunches
    #print "remainder", remainder
    gurobiProbabilityChainModel = gurobipy.Model("master problem single cut")
    try:
        allocVars = [[gurobiProbabilityChainModel.addVar(0,1,vtype = gurobipy.GRB.BINARY,name="y_"+str(j)+","+str(k)) for k in range(instance.numAllocLevels)] for j in range(instance.numFacs)]
        thetaVars = [gurobiProbabilityChainModel.addVar(0, initialUpperBound,vtype = gurobipy.GRB.CONTINUOUS,name="theta_"+str(scenNumber)) for scenNumber in range(numBunches)]
        if(remainder > 0):
            thetaVars.append(gurobiProbabilityChainModel.addVar(0, initialUpperBound,vtype = gurobipy.GRB.CONTINUOUS,name="theta_"+str(numBunches)))
        
        gurobiProbabilityChainModel.update()
        #print "thetaVars", thetaVars
        gurobiProbabilityChainModel.setObjective(sum(thetaVars), gurobipy.GRB.MAXIMIZE)

        gurobiProbabilityChainModel.update()
        # alloc levels constraint
        [gurobiProbabilityChainModel.addConstr(sum([allocVars[j][k] for k in range(instance.numAllocLevels)]) == 1, "one alloc level_" + str(j)) for j in range(instance.numFacs)]
        gurobiProbabilityChainModel.update()
        # instance.budget constraint
        print "instance.budget", instance.budget
        gurobiProbabilityChainModel.addConstr(sum([costOfAlloc[j][k] * allocVars[j][k] for k in range(instance.numAllocLevels) for j in range(instance.numFacs)]) <= instance.budget, "allocation instance.budget")
        gurobiProbabilityChainModel.update()
        gurobiProbabilityChainModel.setParam('OutputFlag', showOutput)
        #print "time_limit", time_limit
        gurobiProbabilityChainModel.setParam('TimeLimit', time_limit)
    except gurobipy.GurobiError as e:
        print 'createProbabilityChainMasterProblemModel_SingleCut_GRB: Gurobi Error reported ' + e
    
def solveBendersMasterProblem(callback = False):
    #print "solving benders master problem"
    startTime = time.time()
    if(writeToFile):
        gurobiProbabilityChainModel.write("/home/hmedal/Documents/Temp/masterProb.lp")
    if(callback):
        gurobiProbabilityChainModel.optimize(bendersCutCallback)
    else:
        gurobiProbabilityChainModel.optimize()
    #print "benders master problem solved"
    runTime = time.time() - startTime
    #     if(global_sampleScenarios):
    #         cadinalityOfSampleSpace = instance.numCapLevels**instance.numFacs
    #         multiplier_to_compute_expectation = cadinalityOfSampleSpace/(numScenarios + 0.0)
    #     else:
    #         multiplier_to_compute_expectation = 1.0
    allocVarSoln = [[allocVars[j][k].getAttr("X") for k in range(instance.numAllocLevels)] for j in range(instance.numFacs)]
    #print "masterProb-objVal", gurobiProbabilityChainModel.objVal
    lb = gurobiProbabilityChainModel.objVal
    ub = gurobiProbabilityChainModel.getAttr('ObjBound')
    return lb, ub, allocVarSoln, runTime
    
def bendersPrelimStuff(scenariosSetArg):
    global scenarioObjectiveValues
    #print 'bendersPrelimStuff-scenariosSetArg'
    if(includeHazards):
        if(not reset_mode):
            createProbabilityChainSecondStageModel_GRB_ForAllScenarios_withHazards(len(scenariosSetArg))
        scenarioObjectiveValues = computScenarioObjectiveValues_withHazards(scenariosSetArg)
    else:
        if(not reset_mode):
            createProbabilityChainSecondStageModel_GRB_ForAllScenarios(len(scenariosSetArg))
        scenarioObjectiveValues = computScenarioObjectiveValues(scenariosSetArg)
    #print "scenarioObjectiveValues", scenarioObjectiveValues

def getObjectiveValueForAllocationSolution(allocLevelsSoln, myNumAllocLevels, myNumCapLevels):
    print "k", myNumAllocLevels, "l", myNumCapLevels
    #print "pValues for soln: ", [imperfectPro_model.getProbabilityFromAllocation(i, instance.numAllocLevels) for i in allocLevelsSoln]
    #capLevelsSamples = instance.createScenarios_CapLevels_withHazards()
    #print "samplesForActualCapLevels", samplesForActualCapLevels
    #capLevelsScenarios = [[sample[fac][1] for fac in range(instance.numFacs)] for sample in capLevelsSamples]
    capLevelsScenarios = instance.createScenarios_CapLevels_AlternateNumCapLevels(myNumCapLevels)
    #print 'capLevelsScenarios', capLevelsScenarios
    #print "capLevelsScenarios[0]", capLevelsScenarios[0]
    #print "capLevelsScenarios len", len(capLevelsScenarios)
    #localPoolVar = Pool()
    scenarioObjVals = [my_computeSecondStageUtility_AltCapLevels(scenario, myNumCapLevels) for scenario in capLevelsScenarios]
    #conditionalScenarioProbsActual = [imperfectPro_model.getProbabilityOfCapLevelScenario_AltNumCapLevels_withHazards([facInfo[1] for facInfo in scenario], allocLevelsSoln, 
    #                                                                                                      [facInfo[0] for facInfo in scenario], instance, 
    #                                                                       instance.numAllocLevelsActual, instance.numCapLevelsActual) for scenario in capLevelsSamples]
    #scenarioProbs = [instance.scenarioProbs[instance.hazardScenarioIndicesAtCapLevels[scenNumber]]*conditionalScenarioProbsActual[scenNumber] 
    #                 for scenNumber in range(len(conditionalScenarioProbsActual))]
    scenarioProbs = [imperfectPro_model.getProbabilityOfCapLevelScenConsideringHazards_AltNumCapLevels(scenario, allocLevelsSoln, instance,
                                                                                                       myNumAllocLevels, myNumCapLevels) 
                                 for scenario in capLevelsScenarios]
    #print "scenarioObjVals", scenarioObjVals
    #print "scenarioProbs", scenarioProbs
    #print "scenarioProbs sum", sum(scenarioProbs)
    objToOriginalProb = sum([a*b for a,b, in zip(scenarioObjVals, scenarioProbs)])
    return objToOriginalProb

# def OLD_getObjectiveValueForAllocationSolution(allocLevelsSoln):
#     #print "pValues for soln: ", [imperfectPro_model.getProbabilityFromAllocation(i, instance.numAllocLevels) for i in allocLevelsSoln]
#     #capLevelsSamples = instance.createScenarios_CapLevels_withHazards()
#     #print "samplesForActualCapLevels", samplesForActualCapLevels
#     #capLevelsScenarios = [[sample[fac][1] for fac in range(instance.numFacs)] for sample in capLevelsSamples]
#     capLevelsScenarios = instance.createScenarios_CapLevels()
#     #print 'capLevelsScenarios', capLevelsScenarios
#     #print "capLevelsScenarios[0]", capLevelsScenarios[0]
#     #print "capLevelsScenarios len", len(capLevelsScenarios)
#     localPoolVar = Pool()
#     scenarioObjVals = localPoolVar.map(my_computeSecondStageUtility, capLevelsScenarios,1)
#     #conditionalScenarioProbsActual = [imperfectPro_model.getProbabilityOfCapLevelScenario_AltNumCapLevels_withHazards([facInfo[1] for facInfo in scenario], allocLevelsSoln, 
#     #                                                                                                      [facInfo[0] for facInfo in scenario], instance, 
#     #                                                                       instance.numAllocLevelsActual, instance.numCapLevelsActual) for scenario in capLevelsSamples]
#     #scenarioProbs = [instance.scenarioProbs[instance.hazardScenarioIndicesAtCapLevels[scenNumber]]*conditionalScenarioProbsActual[scenNumber] 
#     #                 for scenNumber in range(len(conditionalScenarioProbsActual))]
#     scenarioProbs = [imperfectPro_model.getProbabilityOfCapLevelScenConsideringHazards(scenario, allocLevelsSoln, instance) 
#                                  for scenario in capLevelsScenarios]
#     #print "scenarioProbs", scenarioProbs
#     #print "scenarioProbs sum", sum(scenarioProbs)
#     objToOriginalProb = sum([a*b for a,b, in zip(scenarioObjVals, scenarioProbs)])
#     return objToOriginalProb
    
def getObjectiveValueWithoutProtection():
    return getObjectiveValueForAllocationSolution([0]*instance.numFacs, instance.numAllocLevels, instance.numCapLevels)

def getInitialUpperBound():
    global objValueWithFullProtection
    soln, objValue = my_computeSecondStageUtilityAndGetSolution([instance.numCapLevels-1]*instance.numFacs)
    #print "soln", soln
    #print "objValue", objValue
    objValueWithFullProtection = objValue
    meanValueObjective = meanValueProblem.solveMeanValueProblem_BB()[0]
    print "meanValueObjective: ", meanValueObjective
    return min(objValueWithFullProtection, meanValueObjective)

def bendersClassic_ProbabilityChainModel(scenariosSetArg, numBunches, firstStageAlg = 'gurobi', secondStageAlg='gurobi'):
    global scenariosSet, gur_probChain_secondStageModel, amtOfPossibleImprovement, objWithoutPro
    if(trace):
        print "RUNNING CLASSIC MODE"
    #print "scenariosSetArg", scenariosSetArg
    scenariosSet = scenariosSetArg
    numScenarios = len(scenariosSet)
    runTime = 0
    bendersPrelimStuff(scenariosSet)
    initialUB = getInitialUpperBound()
    objWithoutPro = getObjectiveValueWithoutProtection()
    amtOfPossibleImprovement = initialUB - objWithoutPro
    print "amtOfPossibleImprovement", amtOfPossibleImprovement
    if(firstStageAlg == 'gurobi'):
        createProbChainMasterProblemModel_MultiCut_GRB(initialUB, numBunches)
    if(secondStageAlg == 'gurobi'):
        if(includeHazards):
            gur_probChain_secondStageModel = createProbabilityChainSecondStageModel_GRB_withHazards()
        else:
            gur_probChain_secondStageModel = createProbabilityChainSecondStageModel_GRB()
    lb = 1
    ub = float('inf')
    iteration = 0
    startTime = time.time()
    bestAllocVarSoln = None
    if(global_sampleScenarios):
        cadinalityOfSampleSpace = instance.numCapLevels**instance.numFacs
        #print "cadinalityOfSampleSpace", cadinalityOfSampleSpace
        multiplier_to_compute_expectation = cadinalityOfSampleSpace/(numScenarios + 0.0)
    else:
        multiplier_to_compute_expectation = 1.0
    #print "multiplier_to_compute_expectation", multiplier_to_compute_expectation
    allocVarSoln = []
    while(terminationCriteriaNotMet(lb, ub, iteration, runTime)):
        print iteration, lb, ub, runTime, imperfectPro_model.getAllocLevelsVectorFromBinaryVector(allocVarSoln, instance)
        masterLB, masterUB, allocVarSoln, runTime  = solveBendersMasterProblem()
        #print "firstStageSoln: ", allocVarSoln
        if(masterUB < ub): 
            ub = masterUB
        totalObjVal, coefficientDualTerms, constantDualTerms = solveProbabilityChainSecondStageModelForAllScenarios(allocVarSoln)
        logging.info('totalObjVal' + str(totalObjVal))
        addDualBasedCutToMaster_Multicut(coefficientDualTerms, constantDualTerms, iteration, g_numBunches)
        #print "totalObjVal", totalObjVal
        if(totalObjVal * multiplier_to_compute_expectation > lb): 
            lb = totalObjVal * multiplier_to_compute_expectation
            bestAllocVarSoln = allocVarSoln
        iteration += 1
        runTime = time.time() - startTime
    #print "finished", lb, ub
    runTime = time.time() - startTime
    return lb,ub, bestAllocVarSoln, runTime

# def bendersClassic_ProbabilityChainModel_withHazards(scenariosSetArg, numBunches, firstStageAlg = 'gurobi', secondStageAlg='gurobi'):
#     global scenariosSet, gur_probChain_secondStageModel
#     scenariosSet = scenariosSetArg
#     numScenarios = len(scenariosSet)
#     bendersPrelimStuff(scenariosSetArg)
#     if(firstStageAlg == 'gurobi'):
#         createRandomDrawMasterProblemModel_MultiCut_GRB(getInitialUpperBound(), numBunches)
#     if(secondStageAlg == 'gurobi'):
#         gur_probChain_secondStageModel = createProbabilityChainSecondStageModel_GRB()
#     lb = 1
#     ub = float('inf')
#     iteration = 0
#     runTime = 0
#     startTime = time.time()
#     bestAllocVarSoln = None
#     if(global_sampleScenarios):
#         cardinalityOfSampleSpace = (instance.numCapLevels**instance.numFacs) * instance.numHazardScenarios
#         #print "cadinalityOfSampleSpace", cadinalityOfSampleSpace
#         multiplier_to_compute_expectation = cardinalityOfSampleSpace/(numScenarios + 0.0)
#     else:
#         multiplier_to_compute_expectation = 1.0
#     #print "multiplier_to_compute_expectation", multiplier_to_compute_expectation
#     while(terminationCriteriaNotMet(lb, ub, iteration, runTime)):
#         print iteration, lb, ub, runTime
#         masterLB, masterUB, allocVarSoln, runTime  = solveBendersMasterProblem()
#         #print "firstStageSoln: ", allocVarSoln
#         if(masterUB < ub): ub = masterUB
#         totalObjVal, coefficientDualTerms, constantDualTerms = solveProbabilityChainSecondStageModelForAllScenarios(allocVarSoln)
#         logging.info('totalObjVal' + str(totalObjVal))
#         addDualBasedCutToMaster_Multicut(coefficientDualTerms, constantDualTerms, iteration, g_numBunches)
#         #print "totalObjVal", totalObjVal
#         if(totalObjVal * multiplier_to_compute_expectation > lb): 
#             lb = totalObjVal * multiplier_to_compute_expectation
#             bestAllocVarSoln = allocVarSoln
#         iteration += 1
#         runTime = time.time() - startTime
#     runTime = time.time() - startTime
#     return lb,ub, bestAllocVarSoln, runTime
    
def bendersCallback_ProbabilityChainModel(scenariosSetArg, numBunches = 1, firstStageAlg = 'gurobi', secondStageAlg='gurobi'):
    global scenariosSet, scenarioObjectiveValues, gur_probChain_secondStageModel
    reset_mode = False
    scenariosSet = scenariosSetArg
    bendersPrelimStuff(scenariosSetArg)
    if(firstStageAlg == 'gurobi'):    
        createProbChainMasterProblemModel_MultiCut_GRB(getInitialUpperBound(), numBunches, True)
    if(secondStageAlg == 'gurobi'):
        if(includeHazards):
            gur_probChain_secondStageModel = createProbabilityChainSecondStageModel_GRB_withHazards()
        else:
            gur_probChain_secondStageModel = createProbabilityChainSecondStageModel_GRB()
    gurobiProbabilityChainModel._thetaVars = thetaVars
    gurobiProbabilityChainModel._allocVars = [allocVars[j][k] for k in range(instance.numAllocLevels) for j in range(instance.numFacs)]
    gurobiProbabilityChainModel.params.LazyConstraints = 1
    return solveBendersMasterProblem(True)

# def bendersCallback_ProbabilityChainModel_withHazards(scenariosSetArg, numBunches = 1, firstStageAlg = 'gurobi', secondStageAlg='gurobi'):
#     global scenariosSet, scenarioObjectiveValues, gur_probChain_secondStageModel
#     reset_mode = False
#     scenariosSet = scenariosSetArg
#     bendersPrelimStuff(scenariosSetArg)
#     if(firstStageAlg == 'gurobi'):    
#         createRandomDrawMasterProblemModel_MultiCut_GRB(getInitialUpperBound(), numBunches, True)
#     if(secondStageAlg == 'gurobi'):
#         gur_probChain_secondStageModel = createProbabilityChainSecondStageModel_GRB_withHazards()
#     gurobiProbabilityChainModel._thetaVars = thetaVars
#     gurobiProbabilityChainModel._allocVars = [allocVars[j][k] for k in range(instance.numAllocLevels) for j in range(instance.numFacs)]
#     gurobiProbabilityChainModel.params.LazyConstraints = 1
#     return solveBendersMasterProblem(True)
    
def bendersCutCallback(model, where):
    if where == gurobipy.GRB.callback.MIPSOL:
        try:
            #print "add cut"
            allocVarSolnFlat = model.cbGetSolution(model._allocVars)
            allocVarsNested = [[model._allocVars[instance.numFacs*k + j] for k in range(instance.numAllocLevels)] for j in range(instance.numFacs)]
            allocVarSolnNested = [[allocVarSolnFlat[instance.numFacs*k + j] for k in range(instance.numAllocLevels)] for j in range(instance.numFacs)]
            totalObjVal, coefficientDualTerms, constantDualTerms = solveProbabilityChainSecondStageModelForAllScenarios(allocVarSolnNested, False)
            expressions = getConstraintExpressionsForBendersCut(coefficientDualTerms, constantDualTerms, 0, g_numBunches, allocVarsNested, model._thetaVars)
            for bunch in range(len(expressions)):
                model.cbLazy(expressions[bunch])
        except gurobipy.GurobiError as e:
            print e.message
        
def terminationCriteriaNotMet(lb, ub, iteration, runTime):
    if(runTime >= time_limit):
        return False
    if(lb*(1+.0001) < ub):
        return True
    else:
        print "within 0.0001"
        return False

def getConstraintExpressionsForBendersCut(multiplier, coefficientDualTermsAllScenarios, constantDualTerms, iteration, numBunches, allocVarsArg, thetaVarsArg):
    expressions = []
    numScenarios = len(constantDualTerms)
    #print "numBunches", numBunches
    bunchSize = numScenarios/numBunches
    #print "numBunches", numBunches
    remainder = numScenarios%numBunches
    
    for bunch in range(numBunches):
        rangeForBunch = range(bunch * bunchSize, (bunch + 1) * bunchSize)
        coefficientDualTermsSumForBunch = [[multiplier * sum([coefficientDualTermsAllScenarios[scenarioIndex][j][k] for scenarioIndex in rangeForBunch]) for k in range(instance.numAllocLevels)] 
                                           for j in range(instance.numFacs)]
        constantDualsSumForBunch = multiplier * sum(constantDualTerms[bunch * bunchSize:(bunch + 1) * bunchSize])
        expressions.append(sum([allocVarsArg[j][k] * coefficientDualTermsSumForBunch[j][k] for k in range(instance.numAllocLevels) for j in range(instance.numFacs)]) + constantDualsSumForBunch 
                                        >= thetaVarsArg[bunch])
        #print "theta", bunch, thetaVarsArg[bunch]
    if(remainder > 0):
        bunch = numBunches
        rangeForLastBunch = range(numBunches * bunchSize, numScenarios)
        coefficientDualTermsSumForBunch = [[multiplier * sum([coefficientDualTermsAllScenarios[scenarioIndex][j][k] for scenarioIndex in rangeForLastBunch]) for k in range(instance.numAllocLevels)] 
                                           for j in range(instance.numFacs)]
        #print "coefficientDualTermsSumForBunch", coefficientDualTermsSumForBunch
        constantDualsSumForBunch = multiplier * sum(constantDualTerms[numBunches * bunchSize : numScenarios])
        expressions.append(sum([allocVarsArg[j][k] * coefficientDualTermsSumForBunch[j][k] for k in range(instance.numAllocLevels) for j in range(instance.numFacs)]) + constantDualsSumForBunch 
                                        >= thetaVarsArg[bunch])
        #print "thetaRemainder", bunch, thetaVarsArg[bunch]
    #for expr in expressions:
    #    print "expr", expr
    return expressions

def addDualBasedCutToMaster_Multicut(coefficientDualTermsAllScenarios, constantDualTerms, iteration, numBunches):
    #if(debug):
    #    print "addDualBasedCutToMaster_Multicut", constantDualTerms
    if(global_sampleScenarios):
        cadinalityOfSampleSpace = instance.numCapLevels**instance.numFacs
        multiplier_to_compute_expectation = cadinalityOfSampleSpace/(numScenarios + 0.0)
    else:
        multiplier_to_compute_expectation = 1.0
    expressions = getConstraintExpressionsForBendersCut(multiplier_to_compute_expectation, 
                                                        coefficientDualTermsAllScenarios, constantDualTerms, iteration, numBunches, allocVars, thetaVars)
    for bunch in range(len(expressions)):
        #print "cut", expressions[bunch]
        gurobiProbabilityChainModel.addConstr(expressions[bunch], "cut " + str(bunch) + "_" + str(iteration))
    gurobiProbabilityChainModel.update()
    if(writeToFile):
        gurobiProbabilityChainModel.write("/home/hmedal/Documents/Temp/gurobiProbabilityChainModel_afterMultiCut.lp")
        
def afterReadData():
    global numSamples, delta, numSamplesForFinal, thetaValues
    numRVectorRealizations = instance.numFacs**instance.numCapLevels
    logging.info(numRVectorRealizations)
    thetaValues = np.random.uniform(1,5,instance.numFacs)
    #if(numSamples > numRVectorRealizations):
    #    print "WARNING: more samples than possible scenarios"
    #instance information
    global costOfAlloc
    costOfAlloc= [[k for k in range(instance.numAllocLevels)] for j in range(instance.numFacs)]
    if(includeHazards):
        createFacilityProbabilitiesForAllocationLevels_withHazards()
    else:
        createFacilityProbabilitiesForAllocationLevels()

def generateRandomSetOfScenarios(count):
    localScenariosSet = [[myutil.createSparseList(instance.numCapLevels,np.random.randint(instance.numCapLevels)) for j in range(instance.numFacs)] for i in range(count)]
    global numScenarios
    numScenarios = len(localScenariosSet)
    return localScenariosSet

# def OLD_generateRandomSetOfScenarios_withHazards_fromFile(countForEachHazardScenario):
#     allScenarios = []
#     for scenario in instance.scenarioHazardLevels:
#         randomSet = [[np.random.randint(instance.numCapLevels) for j in range(instance.numFacs)] for i in range(countForEachHazardScenario)]
#         allScenarios.append([[[scenario[instance.facIDs[facIndex]], myutil.createSparseList(instance.numCapLevels, tuple[facIndex])] 
#                               for facIndex in range(instance.numFacs)] for tuple in randomSet])
#     global numScenarios
#     numScenarios = len(allScenarios)
#     print "allScenarios_random: ", allScenarios
#     return allScenarios

def generateRandomSetOfScenarios_withHazards_fromFile(totalCount):
    allScenarios = []
    hazardScenarioIndex = 0
    for scenario in instance.scenarioHazardLevels:
        countForHazardScenario = int(totalCount * instance.scenarioProbs[hazardScenarioIndex])
        #print "countForHazardScenario", countForHazardScenario
        capScenariosForHazardScenario = generateRandomSetOfScenarios(countForHazardScenario)
        for scen in capScenariosForHazardScenario:
            #for fac in range(instance.numFacs):
            #    print fac, scen[fac], instance.facIDs, instance.facIDs[fac]
            #    print instance.scenarioHazardLevels[hazardScenarioIndex]
            #    print instance.scenarioHazardLevels[hazardScenarioIndex][fac]
            allScenarios.append([[instance.scenarioHazardLevels[hazardScenarioIndex][fac], scen[fac]] for fac in range(instance.numFacs)])
            hazardScenarioIndices.append(hazardScenarioIndex)
        hazardScenarioIndex += 1
    global numScenarios
    numScenarios = len(allScenarios)
    #print "allScenarios_random: ", allScenarios
    return allScenarios

def generateRandomScenariosFromAllocVector(allocationSoln, count):
    #sum([a*b for a,b in zip(allocationSoln,range(instance.numAllocLevels))])
    allocLevelsSet = [sum([int(a*b) for a,b in zip(list,range(instance.numAllocLevels))]) for list in allocationSoln]
    probabilities = [imperfectPro_model.getProbabilityFromAllocation(allocLevelsSet[fac], instance.numAllocLevels) for fac in allocLevelsSet]
    return [[np.random.binomial(instance.numCapLevels - 1, probabilities[fac]) for fac in range(instance.numFacs)] for i in range(count)]

def generateRandomScenariosFromAllocVector_withHazards_fromFile(allocationSoln, totalCount):
    allScenarios = []
    allocLevelsSet = [sum([int(a*b) for a,b in zip(list,range(instance.numAllocLevels))]) for list in allocationSoln]
    for scenarioIndex in range(len(instance.scenarioHazardLevels)):
        countForHazardScenario = int(totalCount * instance.scenarioProbs[scenarioIndex])
        probabilities = [imperfectPro_model.getProbabilityFromAllocationAndHazardLevel(allocLevelsSet[fac], instance.numAllocLevels, 
                                instance.scenarioHazardLevels[scenarioIndex][fac], instance.numHazardLevels) 
                         for fac in allocLevelsSet]
        capScenariosForHazardScenario = [[np.random.binomial(instance.numCapLevels - 1, probabilities[fac]) 
                                          for fac in range(instance.numFacs)] for i in range(countForHazardScenario)]
        for scen in capScenariosForHazardScenario:
            allScenarios.append(scen)
    return allScenarios

def saa(numMasterProblems, numScenariosInMaster, numSecondStageSamples, alpha = 0.05, bendersType='classic', numBunches = 1, firstStageAlg = 'gurobi', secondStageAlg='gurobi'):
    #prelim stuff
    bendersFn = None
    if(bendersType == 'classic'):
        bendersFn = bendersClassic_ProbabilityChainModel
    elif(bendersType == 'callback'):
        bendersFn = bendersCallback_ProbabilityChainModel
    startTime = time.time()
    if(includeHazards):
        firstRoundResults = [bendersFn(generateRandomSetOfScenarios_withHazards_fromFile(numScenariosInMaster), 
                                       numBunches, firstStageAlg, g_secondStageAlg) for i in range(numMasterProblems)]
    else:
        firstRoundResults = [bendersFn(generateRandomSetOfScenarios(numScenariosInMaster), 
                                       numBunches, firstStageAlg, g_secondStageAlg) for i in range(numMasterProblems)]
    firstRoundSolutions = [result[2] for result in firstRoundResults]
    #print "firstRoundResults", firstRoundResults
    bestFirstRoundSolution = sorted([(tuple[1], tuple[2]) for tuple in firstRoundResults], reverse=True)[:1][0][1]
    #print "best solution: ", bestFirstRoundSolution
    firstRoundObjectives = [list[1] for list in firstRoundResults]
    avgUpperBound = np.mean(firstRoundObjectives)
    #print "avgUB_", avgUpperBound
    ubStdDev = np.std(firstRoundObjectives)
    #print "ubStdDev", ubStdDev
    ubHW = tDist.ppf(1 - alpha/2.0, numMasterProblems - 1) * ubStdDev / np.sqrt(numMasterProblems)
    estimateOfAcualObjective = []
    for soln in firstRoundSolutions:
        po = Pool()
        if(includeHazards):
            randScenarios = generateRandomScenariosFromAllocVector_withHazards_fromFile(soln, numSecondStageSamples)
        else:
            randScenarios = generateRandomScenariosFromAllocVector(soln, numSecondStageSamples)
        res = po.map_async(my_computeSecondStageUtility,[scenario for scenario in randScenarios])
        secondRoundObjectives = res.get()
        estimateOfAcualObjective.append(np.mean(secondRoundObjectives))
        po.close()
        po.join()
    runTime = time.time() - startTime
    avgLowerBound = np.mean(estimateOfAcualObjective)
    #print "avgLB", avgLowerBound
    lbStdDev = np.std(estimateOfAcualObjective)
    #print "lbStdDev", lbStdDev
    lbHW = tDist.ppf(1 - alpha/2.0, numMasterProblems - 1) * lbStdDev / np.sqrt(numMasterProblems)
    #lbLCL = avgLowerBound - lbHW
    #lbUCL = avgLowerBound + lbHW
    #print "avgLB", avgLowerBoundg
    #print "avgUB", avgUpperBound
    return bestFirstRoundSolution, [avgLowerBound, lbHW], [avgUpperBound, ubHW], runTime

def readInExperimentData(path):
    global dataFilePath, hazardsScenarioDefPath, hazardType
    global bendersType, g_numBunches, numSAA_first_stage_probs, numSAA_first_stage_samples, numSAA_second_stage_samples, databaseName, time_limit
    global g_secondStageAlg
    d = etree.parse(open(path))
    dataFilePath = str(d.xpath('//dataset/path[1]/text()')[0])
    hazardsScenarioDefPath = str(d.xpath('//instance/hazardsPath[1]/text()')[0])
    hazardType = str(d.xpath('//instance/hazardType[1]/text()')[0])
    time_limit = float(d.xpath('//algorithm/timeLimit[1]/text()')[0])
    bendersType = str(d.xpath('//algorithm/bendersType[1]/text()')[0])
    g_numBunches = int(d.xpath('//algorithm/numBunches[1]/text()')[0])
    g_secondStageAlg = str(d.xpath('//algorithm/secondStageAlg[1]/text()')[0])
    numSAA_first_stage_probs = int(d.xpath('//algorithm/numSAA_first_stage_probs[1]/text()')[0])
    numSAA_first_stage_samples = int(d.xpath('//algorithm/numSAA_first_stage_samples[1]/text()')[0])
    numSAA_second_stage_samples = int(d.xpath('//algorithm/numSAA_second_stage_probs[1]/text()')[0])
    databaseName = str(d.xpath('//other/databaseName[1]/text()')[0])

def doPrelimStuff():
    global exprFilePath, algType, instance, secondStageProblem, meanValueProblem
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
    meanValueProblem = imperfectPro_meanValueModel.ImproMeanValueModel()
    secondStageProblem.setInstance(instance)
    secondStageProblem.createModelGurobi()
    meanValueProblem.setInstance(instance)
    meanValueProblem.afterReadData()
    meanValueProblem.createModelGurobi()

def runLShaped(sampleScenarios = False, numSamples = 0):
    if(trace):
        print "RUNNING L SHAPED"
    global global_sampleScenarios
    global_sampleScenarios = sampleScenarios
    if(global_sampleScenarios):
        if(includeHazards):
            scenarios = generateRandomSetOfScenarios_withHazards_fromFile(numSamples)
        else:
            scenarios = generateRandomSetOfScenarios(numSamples)
    else:
        if(includeHazards):
            scenarios = createScenarios_withHazardsFromFile()
        else:
            scenarios = createScenarios(instance.numCapLevels)
    #bendersPrelimStuff(scenarios)
    if(bendersType == 'classic'):
        bendersFn = bendersClassic_ProbabilityChainModel
    elif(bendersType == 'callback'):
        bendersFn = bendersCallback_ProbabilityChainModel
    #print "numBunches", numBunches
    print "instance.budget", instance.budget
    lb,ub, bestAllocVarSoln, runTime = bendersFn(scenarios, g_numBunches, 'gurobi', g_secondStageAlg)
    print "runTime", runTime
    print "LB: ", lb, "UB: ", ub
    print "objective improvement ratio", (lb-objWithoutPro)/amtOfPossibleImprovement
    print bestAllocVarSoln
    allocLevelsSoln = imperfectPro_model.getAllocLevelsVectorFromBinaryVector(bestAllocVarSoln, instance)
    print "alloc Levels: ", instance.numAllocLevels, " capLevels: ", instance.numCapLevels
    print "solution: ", allocLevelsSoln
    objForSolution = getObjectiveValueForAllocationSolution(allocLevelsSoln, instance.numAllocLevels, instance.numCapLevels)
    print "obj for solution: ", objForSolution
    pVals = [imperfectPro_model.getProbabilityFromAllocation(i, instance.numAllocLevels) for i in allocLevelsSoln]
    print "pValues: ", pVals
    
    for indexAlloc in range(len(instance.numAllocLevelsActual)):
        for indexCap in range(len(instance.numCapLevelsActual)):
            print "actualNumAllocLevels", instance.numAllocLevelsActual[indexAlloc]
            print "actualNumCapLevels", instance.numCapLevelsActual[indexCap]
            budgetActual = instance.numFacs*(instance.numAllocLevelsActual[indexAlloc]-1)*instance.budgetMultiplier
            solnTrans = [i * (budgetActual+0.0)/instance.budget for i in allocLevelsSoln]
            print "solutionTrans", solnTrans
            print "pValuesTrans: ", [imperfectPro_model.getProbabilityFromAllocation(i, instance.numAllocLevelsActual[indexAlloc]) for i in solnTrans]
            #samplesForActualCapLevels = instance.createScenarios_CapLevels_AlternateNumCapLevels_withHazards(instance.numCapLevelsActual[indexCap])
            #print "samplesForActualCapLevels", samplesForActualCapLevels
            #capLevelsOnly = [[sample[fac][1] for fac in range(instance.numFacs)] for sample in samplesForActualCapLevels]
            #localPoolVar = Pool()
            #scenarioObjValsActual = localPoolVar.map(my_computeSecondStageUtility, capLevelsOnly,1)
            #scenarioObjValsActual = [my_computeSecondStageUtility(scenario) for scenario in capLevelsOnly]
            #conditionalScenarioProbsActual = [imperfectPro_model.getProbabilityOfCapLevelScenario_AltNumCapLevels_withHazards([facInfo[1] for facInfo in scenario], solutionTrans, 
            #                                                                                                       [facInfo[0] for facInfo in scenario], instance, 
            #                                                                        instance.numAllocLevelsActual[indexAlloc], instance.numCapLevelsActual[indexCap]) for scenario in samplesForActualCapLevels]
            #scenarioProbsActual = [instance.scenarioProbs[instance.hazardScenarioIndicesAtCapLevels[scenNumber]]*conditionalScenarioProbsActual[scenNumber] for scenNumber in range(len(conditionalScenarioProbsActual))] 
            #print "scenarioProbsActual", scenarioProbsActual, sum(scenarioProbsActual)
            #objToOriginalProb = sum([a*b for a,b, in zip(scenarioObjValsActual, scenarioProbsActual)])
            #objToOriginalProb = -1
            #print "objToOriginalProb", objToOriginalProb
            objToOriginalProb = getObjectiveValueForAllocationSolution(solnTrans, instance.numAllocLevelsActual[indexAlloc], instance.numCapLevelsActual[indexCap])
            print "objToOriginalProb_REVISED", objToOriginalProb
            
            tableName = "ProbChainImproLShaped"
            dataSetInfo = ['Daskin', hazardType, instance.numDemPts, instance.numFacs]
            print "actualNumCapLevels", instance.numCapLevelsActual[indexCap]
            instanceInfo = [instance.numAllocLevels, instance.numAllocLevelsActual[indexAlloc], instance.numCapLevels, instance.numCapLevelsActual[indexCap], instance.budget, instance.penaltyMultiplier, instance.excess_capacity]
            algParams = [bendersType, g_numBunches]
            algOutput = [runTime, lb, ub]
            solnOutput = [str(allocLevelsSoln), str(solnTrans), objToOriginalProb, objWithoutPro, objValueWithFullProtection]
            dbUtil.printResultsToDB(databaseName, tableName, dataSetInfo, instanceInfo, algParams, algOutput, solnOutput)
    
def runSAA():
    global global_sampleScenarios
    global_sampleScenarios = True
    alpha = 0.05
    timeStart = time.time()
    bestAllocVarSoln, lbInfo, ubInfo, runTime = saa(numSAA_first_stage_probs, numSAA_first_stage_samples, numSAA_second_stage_samples, alpha, bendersType, g_numBunches)
    print "totalTime", time.time() - timeStart
    print "LB: ", lbInfo[0], "+/-", lbInfo[1]
    print "UB: ", ubInfo[0], "+/-", ubInfo[1]
    print "bestAllocVarSoln", bestAllocVarSoln
    allocLevelsSoln = imperfectPro_model.getAllocLevelsVectorFromBinaryVector(bestAllocVarSoln, instance)
    print "solution: ", allocLevelsSoln
    print "pValues: ", [imperfectPro_model.getProbabilityFromAllocation(i, instance.numAllocLevels) for i in allocLevelsSoln]
    solutionTrans = [i * (instance.numAllocLevelsActual/(instance.numAllocLevels + 0.0)) for i in allocLevelsSoln]
    print "solutionTrans", solutionTrans
    samplesForActualCapLevels = instance.createScenarios_CapLevels_AlternateNumCapLevels(instance.numCapLevelsActual)
    #print "samplesForActualCapLevels", samplesForActualCapLevels
    scenarioObjValsActual = [my_computeSecondStageUtility(scenario) for scenario in samplesForActualCapLevels]
    scenarioProbsActual = [imperfectPro_model.getProbabilityOfCapLevelScenario_AltNumCapLevels(scenario, solutionTrans, instance, instance.numAllocLevelsActual, instance.numCapLevelsActual) for scenario in samplesForActualCapLevels]
    #print "scenarioObjValsActual", scenarioObjValsActual
    #print "scenarioProbsActual", scenarioProbsActual
    objToOriginalProb = sum([a*b for a,b, in zip(scenarioObjValsActual, scenarioProbsActual)])
    print "objToOriginalProb", objToOriginalProb
    
    tableName = "ProbChainImproSAA"
    dataSetInfo = ['Daskin', instance.numDemPts, instance.numFacs]
    instanceInfo = [instance.numAllocLevels, instance.numAllocLevelsActual, instance.numCapLevels, instance.numCapLevelsActual, instance.budget, instance.penaltyMultiplier, instance.excess_capacity]
    algParams = [bendersType, g_numBunches, numSAA_first_stage_probs, numSAA_first_stage_samples, numSAA_second_stage_samples, alpha]
    algOutput = [runTime, lbInfo[0], lbInfo[1], ubInfo[0], ubInfo[1]]
    solnOutput = [str(allocLevelsSoln), str(solutionTrans), objToOriginalProb]
    dbUtil.printResultsToDB(databaseName, tableName, dataSetInfo, instanceInfo, algParams, algOutput, solnOutput)
    
if __name__ == "__main__":
    if(debug):
        print "!!!WARNING: DEBUG MODE!!!"
    print "PROBABILITY CHAIN"
    doPrelimStuff()
    if(algType == 'lshaped'):
        runLShaped()
    elif(algType == 'randlshaped'):
        print "L-Shaped with sampled scenarios"
        runLShaped(sampleScenarios = True, numSamples = numSAA_first_stage_samples)
    elif(algType == 'saa'):
        runSAA()
        
#createScenarios_withHazardsFromFile(instance.numCapLevels)
#generateRandomSetOfScenarios_withHazards_fromFile(10)
#instance.createScenarios_CapLevels_withHazards()
#instance.createScenarios_CapLevels_AlternateNumCapLevels_withHazards(5)