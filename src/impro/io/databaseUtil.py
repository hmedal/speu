'''
Created on Sep 30, 2013

@author: hmedal
'''

import sqlite3 as lite
import sys
import datetime
import os

def getGreedyTableCreateString(tableName):
    string =         'CREATE TABLE ' + tableName + '''(date timestamp, 
                     datasetName tinytext,
                     hazardType tinytext, 
                     numDemPts tinyint(100),
                     numFacs tinyint(100),
                     numAllocLevels tinyint(100),
                     numCapLevels tinyint(100),
                     budget real,
                     penaltyMult real,
                     capMult real,
                     algType tinytext,
                     numSamples tinyint(100),
                     runTime real,
                     obValue real,
                     hw real,
                     allocSoln text)'''
    return string

def getLShapedTableCreateString(tableName):
    string =         'CREATE TABLE ' + tableName + '''(date timestamp, 
                     datasetName tinytext,
                     hazardType tinytext,
                     numDemPts tinyint(100),
                     numFacs tinyint(100),
                     numAllocLevels tinyint(100),
                     numAllocLevelsActual tinyint(100),
                     numCapLevels tinyint(100),
                     numCapLevelsActual tinyint(100),
                     budget real,
                     penaltyMult real,
                     capMult real,
                     bendersType tinytext,
                     numBendersBunches tinyint(100),
                     runTime real,
                     lb real,
                     ub real,
                     allocSoln text,
                     allocSolnActual text,
                     objToOriginalProb real,
                     objWithoutPro real,
                     objValueWithFullProtection real)'''
    return string

def getMeanValueTableCreateString(tableName):
    string =         'CREATE TABLE ' + tableName + '''(date timestamp, 
                     datasetName tinytext,
                     hazardType tinytext, 
                     numDemPts tinyint(100),
                     numFacs tinyint(100),
                     numAllocLevels tinyint(100),
                     numCapLevels tinyint(100),
                     budget real,
                     penaltyMult real,
                     capMult real,
                     runTime real,
                     lb real,
                     ub real,
                     allocSoln text,
                     objToOriginalProb real)'''
    return string

def getSAATableCreateString(tableName):
    string =         'CREATE TABLE ' + tableName + '''(date timestamp, 
                     datasetName tinytext, 
                     numDemPts tinyint(100),
                     numFacs tinyint(100),
                     numAllocLevels tinyint(100),
                     numAllocLevelsActual tinyint(100),
                     numCapLevels tinyint(100),
                     numCapLevelsActual tinyint(100),
                     budget real,
                     penaltyMult real,
                     capMult real,
                     bendersType tinytext,
                     numBendersBunches tinyint(100),
                     numFirstStageProbs tinyint(100),
                     numFirstStageSamples tinyint(100),
                     numSecondStageSamples tinyint(100),
                     alpha real,
                     runTime real,
                     lbMean real,
                     lbHW real,
                     ubMean real,
                     ubHW real,
                     soln text(1000),
                     solnActual text(1000),
                     objToOriginalProb real)'''
    return string
    
def createTable_Greedy(databaseName, tableName):
    con = None
    try:
        con = lite.connect(databaseName)
        
        c = con.cursor()
        
        c.execute('drop table if exists ' + tableName)
        
        # Create table
        c.execute(getGreedyTableCreateString(tableName))

        con.commit()
    
    except lite.Error, e:
        
        if con:
            con.rollback()
            
        print "Error %s:" % e.args[0]
        sys.exit(1)
        
    finally:
        
        if con:
            con.close()

def createTable_LShaped(databaseName, tableName):
    
    con = None
    try:
        con = lite.connect(databaseName)
        
        c = con.cursor()
        
        c.execute('drop table if exists ' + tableName)
        
        # Create table
        c.execute(getLShapedTableCreateString(tableName))

        con.commit()
    
    except lite.Error, e:
        
        if con:
            con.rollback()
            
        print "Error %s:" % e.args[0]
        sys.exit(1)
        
    finally:
        
        if con:
            con.close()
            
def createTable_MeanValue(databaseName, tableName):
    
    con = None
    try:
        con = lite.connect(databaseName)
        
        c = con.cursor()
        
        c.execute('drop table if exists ' + tableName)
        
        # Create table
        c.execute(getMeanValueTableCreateString(tableName))

        con.commit()
    
    except lite.Error, e:
        
        if con:
            con.rollback()
            
        print "Error %s:" % e.args[0]
        sys.exit(1)
        
    finally:
        
        if con:
            con.close()
            
def createTable_SAA(databaseName, tableName):
    databaseName = "/home/hmedal/Documents/2_msu/1_MSU_Projects/facilityProtection_CSF/expr_output/impro-results.db"
    con = None
    try:
        con = lite.connect(databaseName)
        
        c = con.cursor()
        
        c.execute('drop table if exists ' + tableName)
        
        # Create table
        c.execute(getSAATableCreateString(tableName))

        con.commit()
    
    except lite.Error, e:
        
        if con:
            con.rollback()
            
        print "Error %s:" % e.args[0]
        sys.exit(1)
        
    finally:
        
        if con:
            con.close()
            
def printResultsToDB(databaseName, tableName, dataSetInfo, instanceInfo, algParams, algOutput, solnOutput):
    con = None
    date = datetime.datetime.now()
    combinedList = [date] + dataSetInfo + instanceInfo + algParams + algOutput + solnOutput
    print combinedList
    valuesStringQuestionMark = ""
    for index in range(len(combinedList) - 1):
        valuesStringQuestionMark += "?" + ", "
    valuesStringQuestionMark += "?"
    #print " VALUES(" + valuesStringQuestionMark + ")"
    print "databaseName", databaseName
    try:
        con = lite.connect(databaseName)
        con.execute("INSERT INTO " + tableName +" VALUES(" + valuesStringQuestionMark + ")", combinedList)
        con.commit()
    
    except lite.Error, e:
        
        if con:
            con.rollback()
            
        print "Error %s:" % e.args
        sys.exit(1)
        
    finally:
        
        if con:
            con.close()

def testDataEntry_SAA(databaseName, tableName):
    dataSetInfo = ['Daskin', 49, 5]
    instanceInfo = [3,3,3,2.0, 0.3]
    algParams = ['classic', 1, 10, 1000, 10000, 0.05]
    algOutput = [9.32, 1000.0, 10.0, 1100.0, 10.0]
    solnOutput = [str([0, 1, 2, 1])]
    printResultsToDB(databaseName, tableName, dataSetInfo, instanceInfo, algParams, algOutput, solnOutput)
    

def testDataEntry_Greedy(databaseName, tableName):
    dataSetInfo = ['Daskin', 49, 5]
    instanceInfo = [3,3,3,2.0, 0.3]
    algParams = ['enum', 4, 5, 4]
    algOutput = [9.32, 1142.72]
    solnOutput = [str([0, 1, 2, 1])]
    printResultsToDB(databaseName, tableName, dataSetInfo, instanceInfo, algParams, algOutput, solnOutput)
    
def addColumn(databaseName, tableName, columnNameToAdd, columnDescrToAdd, columnAfter):
    con = None
    try:
        con = lite.connect(databaseName)
        
        c = con.cursor()
        
        c.execute('ALTER TABLE ' + tableName + ' ADD ' + columnNameToAdd + ' ' + columnDescrToAdd + ' AFTER ' + columnAfter)

        con.commit()
    
    except lite.Error, e:
        
        if con:
            con.rollback()
            
        print "Error %s:" % e.args[0]
        sys.exit(1)
        
    finally:
        
        if con:
            con.close()
    
def createAllTables(databaseName):
    createTable_Greedy(databaseName, "GreedyImpro")
    createTable_LShaped(databaseName, "ProbChainImproLShaped")
    createTable_MeanValue(databaseName, "MeanValueImpro")
    
def getStringsToreateAllTables():
    print getGreedyTableCreateString("GreedyImpro")
    print getLShapedTableCreateString("ProbChainImproLShaped")
    print getMeanValueTableCreateString("MeanValueImpro")
    
if __name__ == '__main__':
    #databaseName = "/home/hmedal/Documents/2_msu/1_MSU_Projects/facilityProtection_CSF/expr_output/impro-results_tgv.db"
    databaseName = "/home/hmedal/Documents/2_msu/1_MSU_Projects/facilityProtection_CSF/expr_output/impro-results_local.db"
    #tableName = "MeanValueImpro"
    #tableNames = ["ProbChainImproLShaped", "MeanValueImpro", "GreedyImpro"]
    #testDataEntry_Greedy(databaseName, tableName)
    #createTable_MeanValue(databaseName, tableName)
    #createAllTables(databaseName)
    #createTable_LShaped(databaseName, tableName)
    #print getLShapedTableCreateString(tableName)
    #for tableName in tableNames:
    #    addColumn(databaseName, tableName, 'hazardType', 'TINYTEXT', 'datasetName')
    #os.system("~/Documents/2_msu/1_MSU_Projects/facilityProtection_CSF/scripts/cp_db.sh")
    #executeSensitivityQuery(databaseName, 49, 4, 0.25)
    getStringsToreateAllTables()
    print "finished"