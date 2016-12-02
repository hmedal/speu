"""
The Beer Distribution Problem for the PuLP Modeller

Authors: Antony Phillips, Dr Stuart Mitchell  2007
"""

# Import PuLP modeler functions
from pulp import *
from multiprocessing import Pool
import time
import numpy as np
import os
from gurobipy import *

numFacs = 2
numDemPts = 5

Warehouses = [str(i) for i in range(1, numFacs+1)]
Bars = [str(i) for i in range(1,numDemPts+1)]

# Creates a dictionary for the number of units of demand for each demand node
demandArray = [500,900,1800,200,700]
demand = {str(i+1): demandArray[i] for i in range(numDemPts)}
supplyArray = [3000, 4000]
# Creates a list of costs of each transportation path
costsArray = [   #Bars
         #1 2 3 4 5
         [200,400,500,200,100],#A   Warehouses
         [300,100,300,200,300], #B
         [3,1,3,2,3] #Dummy
         ]

# The cost data is made into a dictionary
costs = makeDict([Warehouses,Bars],costsArray,0)

print costs["2"]["1"]
# A dictionary called 'Vars' is created to contain the referenced variables(the routes)


# Creates a list of tuples containing all the possible routes for transport
Routes = [(w,b) for w in Warehouses for b in Bars]

def solveWarehouseProblemGurobi(supplyValues):
    model = Model("myLP")
    try:
        # Create variables)
        assignVars = [[model.addVar(0,10000,vtype=GRB.CONTINUOUS,name="x_"+str(i)+","+str(j)) for j in range(numFacs+1)] for i in range(numDemPts)]
        # Integrate new variables
        model.update()
        # Set objective
        #print sum([costsArray[j][i]*assignVars[i][j] for j in range(numFacs + 1) for i in range(numDemPts)])
        model.setObjective(sum([costsArray[j][i]*assignVars[i][j] for j in range(numFacs + 1) for i in range(numDemPts)]), GRB.MAXIMIZE)
        model.update()
        for j in range(numFacs):
            model.addConstr(sum([demandArray[i]*assignVars[i][j] for i in range(numDemPts)]) <= supplyValues[j], "capacity_"+str(j)) 
        for i in range(numDemPts):
            model.addConstr(sum([assignVars[i]  [j] for j in range(numFacs+1)]) >= demandArray[i], "demand_met"+str(i))
        model.update()
        model.setParam('OutputFlag', False )
    except GurobiError:
        print 'Error reported'
    model.write("/home/hmedal/Documents/Temp/gurobiLP.lp")
    model.optimize()
    return model.objVal

def solveWarehouseProblem(supplyValues):
    supply = {str(i+1): supplyValues[i] for i in range(2)}
    myVars = LpVariable.dicts("Route",(Warehouses,Bars),0,None,LpInteger)
    #print supply
    prob = LpProblem("Beer Distribution Problem", LpMinimize)
    prob += lpSum([myVars[w][b]*costs[w][b] for (w,b) in Routes]), "Sum_of_Transporting_Costs"
    for w in Warehouses:
        prob += lpSum([myVars[w][b] for b in Bars])<=supply[w], "Sum_of_Products_out_of_Warehouse_%s"%w
    for b in Bars:
        prob += lpSum([myVars[w][b] for w in Warehouses])>=demand[b], "Sum_of_Products_into_Bar%s"%b
    print "before solve", prob
    prob.writeLP("/home/hmedal/Documents/Temp/gurobiLP.lp")
    prob.solve(GLPK_CMD(msg=False))
    
    #print "Total Cost of Transportation = ", value(prob.objective)
    return value(prob.objective)
    
if __name__ == '__main__':
    pool = Pool(processes=4)              # start 4 worker processes
    #start = time.time()
    #result = pool.apply_async(f, [10])    # evaluate "f(10)" asynchronously
    #result.get(timeout=1)           # prints "100" unless your computer is *very* slow
    solveWarehouseProblemGurobi([500,1000])
    for p in np.arange(0.2,1,0.1):
        values = [[np.random.binomial(3000,p) for i in range(2)] for j in range(1000)]
        start = time.time()
        solveWarehouseProblemGurobi([4000,2000])
        objValues = pool.map(solveWarehouseProblemGurobi, values)
        #print time.time() - start
        print p, np.mean(objValues)     # prints "[0, 1, 4,..., 81]"