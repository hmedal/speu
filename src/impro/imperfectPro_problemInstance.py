'''
Created on Aug 23, 2013

@author: hmedal
'''

import lxml.etree as etree
import numpy as np
import itertools
import ast
import src.impro.myUtil as myutil

class Instance(object):
    '''
    classdocs
    '''
    
    utility_dissipation_constant = 3
    distance_rate = 100.0 # miles per hour perhaps (traveling by plane)

    def time_from_distance(self, distance):
        return distance/self.distance_rate
    
    def utility(self, distance, maxDistance):
        time = self.time_from_distance(distance)
        maxTime = self.time_from_distance(maxDistance)
        return self.utility_dissipation_constant * np.ma.core.exp(-self.utility_dissipation_constant * time/(maxTime + 0.0))
    
    def utilityOLD2(self, distance, maxDist):
        return maxDist - distance
    
    def utilityOLD(self,distance):
        return (1 + np.ma.core.exp(.0001*distance))**-1

    def readInExperimentData(self, path):
        global numCapLevels, numAllocLevels, budget, excess_capacity, penaltyMultiplier, numThreads, pool
        self.numAllocLevelsActual = []
        self.numCapLevelsActual = []
        d = etree.parse(open(path))
        self.excess_capacity = float(d.xpath('//instance/excess_capacity[1]/text()')[0])
        self.penaltyMultiplier = float(d.xpath('//instance/penaltyMultiplier[1]/text()')[0])
        self.numAllocLevels = int(d.xpath('//instance/numAllocLevels[1]/text()')[0])
        self.numCapLevels = int(d.xpath('//instance/numCapLevels[1]/text()')[0])
        print "numAllocLevelsActual: ", d.xpath('//instance/numAllocLevelsActual[1]/text()')[0]
        self.numAllocLevelsActual = ast.literal_eval(d.xpath('//instance/numAllocLevelsActual[1]/text()')[0])
        self.numCapLevelsActual= ast.literal_eval(d.xpath('//instance/numCapLevelsActual[1]/text()')[0])
        print "actual arrays: ", self.numAllocLevelsActual, self.numCapLevelsActual
        self.budgetMultiplier = float(d.xpath('//instance/budgetMultiplier[1]/text()')[0])
        
    def readInHazardsScenarioData(self, path):
        d = etree.parse(open(path))
        self.numHazardLevels = int(d.xpath('//hazardsData/numHazardLevels[1]/text()')[0])
        #print "numHazardLevels", self.numHazardLevels
        scenarios = d.xpath('//scenario')
        self.scenarioProbs = []
        self.scenarioHazardLevels = []
        for index in range(1, len(scenarios) + 1):
            self.scenarioProbs.append(float(d.xpath('//scenarios/scenario[' + str(index) + ']/prob[1]/text()')[0]))
            numLocs = len(d.xpath('//scenarios/scenario[' + str(index) + ']/hazardLevelAtFacility'))
            locHazardLevelsForScenario = []
            for loc in range(numLocs):
                locHazardLevelsForScenario.append(int(d.xpath('//scenarios/scenario[' + str(index) + ']/hazardLevelAtFacility[@facName=' + str(loc) + ']/text()')[0]))
            self.scenarioHazardLevels.append(locHazardLevelsForScenario)
        self.numHazardScenarios = len(scenarios)
        #print "scenario probs", self.scenarioProbs
        #print "scenarioHazardLevels", self.scenarioHazardLevels
    
    def createInstance(self, dataset):
        global numFacs
        self.facIDs = dataset.facIDs
        self.numFacs = dataset.numFacs
        self.demPtWts = dataset.demPtWts
        self.numDemPts = dataset.numDemPts
        self.sumDemand = sum(self.demPtWts)
        self.capacities = [(self.sumDemand/((1-self.excess_capacity)*self.numFacs)) for i in range(self.numFacs)]
        pairsDistMatrix = dataset.pairsDistMatrix
        maxDist = max([max(pairsDistMatrix[i]) for i in range(self.numFacs)])
        #print "maxDist", maxDist, self.utility(self.penaltyMultiplier * maxDist)
        
        self.pairsUtilityMatrix = [[self.utility(pairsDistMatrix[i][j], self.penaltyMultiplier * maxDist) for j in range(self.numFacs)] for i in range(self.numDemPts)]
        #for j in range(self.numFacs):
        #    for i in range(self.numDemPts):
        #        print pairsDistMatrix[i][j], self.pairsUtilityMatrix[i][j]
        #print "penaltyMultiplier", self.penaltyMultiplier
        for i in range(self.numDemPts):
            self.pairsUtilityMatrix[i].append(self.utility(self.penaltyMultiplier * maxDist, self.penaltyMultiplier * maxDist))
        #print "utilMatrix", self.pairsUtilityMatrix
        print "budget (without round): ", self.budgetMultiplier * self.numFacs * (self.numAllocLevels - 1)
        self.budget = round(self.budgetMultiplier * self.numFacs * (self.numAllocLevels - 1))
        print "self.budget: ", self.budget
            
    def createScenarios_CapLevels(self):
        cartProd = itertools.product(range(self.numCapLevels), repeat = self.numFacs)
        cartProdList = [list(i) for i in cartProd]
        return cartProdList
    
    def createScenarios_CapLevels_AlternateNumCapLevels(self, myNumCapLevels):
        cartProd = itertools.product(range(myNumCapLevels), repeat = self.numFacs)
        cartProdList = [list(i) for i in cartProd]
        return cartProdList
    
    def createScenarios_CapLevels_withHazards(self):
        allScenarios = []
        for hazardScenario in self.scenarioHazardLevels:
            cartProd = itertools.product(range(self.numCapLevels), repeat = self.numFacs)
            #allScenarios.append([list(i) for i in cartProd])
            for scenario in cartProd:
                allScenarios.append([[hazardScenario[facIndex], scenario[facIndex]] 
                                      for facIndex in range(self.numFacs)])
        print "allScenarios: ", allScenarios
        global numScenarios
        numScenarios = len(allScenarios)
        return allScenarios
    
    def createScenarios_CapLevels_AlternateNumCapLevels_withHazards(self, myNumCapLevels):
        allScenarios = []
        self.hazardScenarioIndicesAtCapLevels = []
        hazardScenarioIndex = 0
        for hazardScenario in self.scenarioHazardLevels:
            cartProd = itertools.product(range(myNumCapLevels), repeat = self.numFacs)
            #allScenarios.append([list(i) for i in cartProd])
            for scenario in cartProd:
                allScenarios.append([[hazardScenario[facIndex], scenario[facIndex]] 
                                  for facIndex in range(self.numFacs)])
                self.hazardScenarioIndicesAtCapLevels.append(hazardScenarioIndex)
            hazardScenarioIndex += 1
        #print "allScenarios: ", allScenarios
        global numScenarios
        numScenarios = len(allScenarios)
        return allScenarios
    
    def NOT_WORKING_createScenarios_CapLevels_AlternateNumCapLevels_withHazards(self, myNumCapLevels):
        allScenarios = []
        for scenario in self.scenarioHazardLevels:
            cartProd = itertools.product(range(myNumCapLevels), repeat = self.numFacs)
            cartProdList = [list(i) for i in cartProd]
            for myTuple in cartProdList:
                allScenarios.append([[scenario[facIndex], myutil.createSparseList(myNumCapLevels, myTuple[facIndex])] 
                                  for facIndex in range(self.numFacs)])
        #print "allScenarios: ", allScenarios
        #print "hazardScenarioIndices: ", hazardScenarioIndices
        global numScenarios
        numScenarios = len(allScenarios)
        print "numScenarios: createScenarios_CapLevels_AlternateNumCapLevels_withHazards", numScenarios
        return allScenarios
    
    def getRandomHazardLevelsScenarioIndex(self):
        cutPoints = [0]
        cumulativeProb = 0
        for scenarioIndex in range(self.numHazardScenarios):
            cumulativeProb += self.scenarioProbs[scenarioIndex]
            cutPoints.append(cumulativeProb)
        randNum = np.random.uniform()
        index = 1
        while(True):
            if(randNum > cutPoints[index - 1] and randNum <= cutPoints[index]):
                break
            index += 1
        #print "getRandomHazardLevelsScenario", index
        return index - 1
        
    def getRandomHazardLevelsVector(self):
        hazardScenarioIndex = self.getRandomHazardLevelsScenarioIndex()
        hazardScenario = self.scenarioHazardLevels[hazardScenarioIndex]
        return [hazardScenario[self.facIDs[fac]] for fac in range(self.numFacs)]
    
    def convertCapsAndHazardsScenarios_to_CapsScenarios(self, capsAndHazardsScenarios):
        capsScenarios = []
        numScenarios = len(capsAndHazardsScenarios)
        for scenarioIndex in range(numScenarios):
            capsScenarios.append([capsAndHazardsScenarios[scenarioIndex][facIndex][1] for facIndex in range(self.numFacs)])
        #print "capsScenarios", capsScenarios
        return capsScenarios
    
    def convertCapsAndHazardsScenarios_to_HazardScenarios(self, capsAndHazardsScenarios):
        hazardScenarios = []
        numScenarios = len(capsAndHazardsScenarios)
        for scenarioIndex in range(numScenarios):
            hazardScenarios.append([capsAndHazardsScenarios[scenarioIndex][facIndex][0] for facIndex in range(self.numFacs)])
        #print "capsScenarios", capsScenarios
        return hazardScenarios
    