import sqlite3 as lite
import sys
import datetime
import os

def getQueryForRunTimeTable():
    query = """SELECT DISTINCT ProbChainImproLShaped.date, MAX(GreedyImpro.date) AS greedyDate, ProbChainImproLShaped.datasetName, ProbChainImproLShaped.hazardType, ProbChainImproLShaped.numDemPts, ProbChainImproLShaped.numFacs,ProbChainImproLShaped.numAllocLevels AS k, ProbChainImproLShaped.numCapLevels AS l, ProbChainImproLShaped.budget AS b, ProbChainImproLShaped.penaltyMult, ProbChainImproLShaped.capMult, round(ProbChainImproLShaped.runTime,1) AS probChainRunTime, round(GreedyImpro.runTime,1) AS greedyRunTime, round(GreedyImpro.obValue,3) AS greedyObjValue, round(ProbChainImproLShaped.lb,3) AS lb, round(ProbChainImproLShaped.ub,3) AS ub, round((ProbChainImproLShaped.ub-ProbChainImproLShaped.lb)/ProbChainImproLShaped.ub,3) AS gap, round(GreedyImpro.obValue/ProbChainImproLShaped.lb,4) AS ratio
    FROM ProbChainImproLShaped, GreedyImpro
    WHERE ProbChainImproLShaped.date > '2014-03-21 00:00:37.508849'
    AND ProbChainImproLShaped.penaltyMult = 2.0
    AND ProbChainImproLShaped.capMult  = 0.1
    AND ProbChainImproLShaped.bendersType = 'classic'
    AND GreedyImpro.date > '2014-03-22 00:00:37.508849'
    AND ProbChainImproLShaped.datasetName = GreedyImpro.datasetName
    AND ProbChainImproLShaped.hazardType = GreedyImpro.hazardType
    AND ProbChainImproLShaped.numDemPts = GreedyImpro.numDemPts
    AND ProbChainImproLShaped.numFacs = GreedyImpro.numFacs
    AND ProbChainImproLShaped.numAllocLevels = GreedyImpro.numAllocLevels
    AND ProbChainImproLShaped.numCapLevels = GreedyImpro.numCapLevels
    AND ProbChainImproLShaped.budget = GreedyImpro.budget
    AND ProbChainImproLShaped.penaltyMult = GreedyImpro.penaltyMult
    AND ProbChainImproLShaped.capMult = GreedyImpro.capMult
    GROUP BY GreedyImpro.date
    ORDER BY ProbChainImproLShaped.datasetName, ProbChainImproLShaped.numDemPts, ProbChainImproLShaped.numFacs, 
    ProbChainImproLShaped.numAllocLevels, ProbChainImproLShaped.numCapLevels, ProbChainImproLShaped.budget, ProbChainImproLShaped.hazardType;"""
    print query
    return query

def getQueryForSensitivityTable():
    query = """SELECT DISTINCT t1.datasetName, t1.hazardType, t1.numDemPts, t1.numFacs AS p, t1.numAllocLevels AS k, t1.numCapLevels AS l, t1.numAllocLevelsActual AS kA, t1.numCapLevelsActual AS lA, t1.budget AS b, t1.budget/(t1.numFacs * (t1.numAllocLevels - 1)) AS bMult, t2.budget/(t2.numFacs * (t2.numAllocLevels - 1)) AS bMult2, round(t1.lb, 3) AS lb, objToOriginalProb, round(t2.lb, 3) AS lb2, objToOriginalProb/round(t2.lb, 3) AS ratio 
    FROM ProbChainImproLShaped AS t1, (
    SELECT DISTINCT datasetName, hazardType, numDemPts,numFacs, numAllocLevels, numCapLevels, budget, penaltyMult, capMult, round(lb, 3) AS lb, round(ub, 3) AS ub
    FROM ProbChainImproLShaped
        WHERE date > '2014-03-22 00:00:00.508849'
        AND budget = numFacs * (numAllocLevels - 1) * 0.25
        ORDER BY datasetName, numDemPts, numFacs, numCapLevels, numAllocLevels, budget
    ) AS t2
    WHERE date > '2014-03-22 00:00:00.508849'
    AND t1.numDemPts = 88
    AND t1.numFacs = 8
    AND t1.budget = t1.numFacs * (t1.numAllocLevels - 1) * 0.25
    AND t1.penaltyMult = 2.0
    AND t1.capMult = 0.1
    AND t1.datasetName = t2.datasetName
    AND t1.hazardType = t2.hazardType
    AND t1.numDemPts = t2.numDemPts
    AND t1.numFacs = t2.numFacs
    AND t1.numAllocLevelsActual = t2.numAllocLevels
    AND t1.numCapLevelsActual = t2.numCapLevels
    AND t1.penaltyMult = t2.penaltyMult
    AND t1.capMult = t2.capMult
    AND t1.budget/(t1.numFacs * (t1.numAllocLevels - 1)) = t2.budget/(t2.numFacs * (t2.numAllocLevels - 1))
    ORDER BY t1.datasetName, t1.numDemPts, t1.numFacs, t1.budget/(t1.numFacs * (t1.numAllocLevels - 1)), t1.numCapLevels, t1.numAllocLevels, t1.hazardType;"""
    #print query
    return query

def executeRunTimeQuery(databaseName, numDemPts, numFacs, bMult):
    ratioValues = []
    for k in range(3):
        ratioValues.append([])
        for l in range(3):
            ratioValues[k].append([])
    con = None
    try:
        con = lite.connect(databaseName)
        
        c = con.cursor()
        
        rows = c.execute(getQueryForRunTimeTable())
        for row in rows:
            print row
            k = row[3]
            l = row[4]
            kA = row[5]
            lA = row[6]
            print k, l, kA, lA, row[len(row)-1]
            ratioValues[k-2][l-2].append(row[len(row)-1])
            #print k, l, ratioValues[k-2][l-2]
    except lite.Error, e:
        
        if con:
            con.rollback()
            
        print "Error %s:" % e.args[0]
        sys.exit(1)
        
    finally:
        
        if con:
            con.close()
    for l in range(3):
        for k in range(3):
            myStr = ""
            #print k, l, ratioValues[k][l]
            for value in ratioValues[k][l]:
                myStr += str(round(value,3)) + "\t"
            print myStr
            
def executeSensitivityQuery(databaseName, numDemPts, numFacs, bMult):
    hazTypes = {'allExposed' : 0, 'halfExposed' : 1, 'conditional' : 2}
    ratioValues = []
    for k in range(3):
        ratioValues.append([])
        for l in range(3):
            ratioValues[k].append([])
            for hazType in range(3):
                ratioValues[k][l].append([])
                for kA in range(3):
                    ratioValues[k][l][hazType].append([])
                    for lA in range(3):
                        ratioValues[k][l][hazType][kA].append(-1)
    #print ratioValues[0][0][0]
    con = None
    try:
        con = lite.connect(databaseName)
        
        c = con.cursor()
        
        rows = c.execute(getQueryForSensitivityTable())
        for row in rows:
            #print row
            hazType = row[1]
            hazIndex = hazTypes[hazType]
            k = row[4]
            l = row[5]
            kA = row[6]
            lA = row[7]
            print hazType, k, l, kA, lA, row[len(row)-1]
            ratioValues[k-2][l-2][hazIndex][kA-2][lA-2] = row[len(row)-1]
            #print k-2, l-2, ratioValues[k-2][l-2]
    except lite.Error, e:
        
        if con:
            con.rollback()
            
        print "Error %s:" % e.args[0]
        sys.exit(1)
        
    finally:
        
        if con:
            con.close()
    #print "start printing", ratioValues
    for k in range(3):
        for l in range(3):
            for hazIndex in range(3):
                myStr = ""
                #print "ratioValues:", ratioValues[k][l][hazIndex]
                for kA in range(3):
                    #print ratioValues[k][l][hazIndex][kA]
                    for lA in range(3):
                        if(ratioValues[k][l][hazIndex][kA][lA] != -1):
                           myStr += str(round(ratioValues[k][l][hazIndex][kA][lA],3)) 
                        myStr += "\t"
                print hazIndex, (k+2), (l+2), myStr
            


if __name__ == '__main__':
    databaseName = "/home/hmedal/Documents/2_msu/1_MSU_Projects/facilityProtection_CSF/expr_output/impro-results_tgv.db"
    #databaseName = "/home/hmedal/Documents/2_msu/1_MSU_Projects/facilityProtection_CSF/expr_output/impro-results_local.db"
    #tableName = "MeanValueImpro"
    #tableNames = ["ProbChainImproLShaped", "MeanValueImpro", "GreedyImpro"]
    #testDataEntry_Greedy(databaseName, tableName)
    #createTable_MeanValue(databaseName, tableName)
    #createAllTables(databaseName)
    #createTable_LShaped(databaseName, tableName)
    #print getLShapedTableCreateString(tableName)
    #for tableName in tableNames:
    #    addColumn(databaseName, tableName, 'hazardType', 'TINYTEXT', 'datasetName')
    os.system("~/Documents/2_msu/1_MSU_Projects/facilityProtection_CSF/scripts/cp_db.sh")
    executeSensitivityQuery(databaseName, 49, 4, 0.25)
    print "finished"
