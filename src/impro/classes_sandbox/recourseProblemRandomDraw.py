'''
Created on Jun 17, 2013

@author: hmedal
'''
import recourseProblem
import pulp
#import problemInstanceDiscreteAlloc
import gurobipy

class RecourseProblemRandomDraw(recourseProblem.RecourseProblem):
    '''
    classdocs
    '''
    inst = None
    
    def __init__(self):
        global inst
        '''
        Constructor
        '''
        #inst = problemInstanceDiscreteAlloc.ProblemInstanceDiscreteAlloc()
        print("constructed2 ",inst)
        path ='/home/hmedal/Documents/2_msu/research_manager/data/facLoc/Daskin/Daskin88_FacPro_p4.xml'
        inst.readInDataset(path)
        print inst.numDemPts
        
    """def solveWarehouseProblem(self, capacityLevels):
        print "solveWarehouseProblem", "capLevels: ", capacityLevels
        supply = {str(i+1): capacityLevels[i] * self.inst.capacities[i] for i in range(self.inst.numFacs)}
        supply[str(self.inst.numFacs+1)] = 1000000
        print "right before vars"
        myVars = pulp.LpVariable.dicts("Route",(DemPts, Warehouses),0,1,LpInteger)
        prob = pulp.LpProblem("GAP", pulp.LpMaximize)
        print "right after prob"
        print myVars["1"]["2"]
        print sum([myVars[b][w] * costs[b][w] * demand[b] for (b,w) in Routes])
        prob += pulp.lpSum([myVars[b][w] * costs[b][w] * demand[b] for (b,w) in Routes]), "Sum_of_Transporting_Costs"
        print "right after add"
        for w in Warehouses:
            prob += pulp.lpSum([myVars[b][w]*demand[b] for b in DemPts]) <= supply[w], "Sum_of_Products_out_of_Warehouse_%s"%w
        for b in DemPts:
            prob += pulp.lpSum([myVars[b][w] for w in Warehouses]) == 1, "Sum_of_Products_into_Bar%s"%b
        #print prob
        print "right before solve"
        prob.solve(pulp.GLPK_CMD(msg=False))
        print "status: ", prob.status
        objVal = pulp.value(prob.objective)
        print "Total Cost of Transportation = ", objVal
        return objVal"""
    
    def createRandomDrawModel(self):
        print "creating model"
        model = gurobipy.Model("myLP")
        try:
            # Create variables)
            assignVars = [[model.addVar(0,1,vtype = gurobipy.GRB.CONTINUOUS,name="x_"+str(i)+","+str(j)) for j in range(self.inst.numFacs+1)] for i in range(self.inst.numDemPts)]
            activationVars = [[model.addVar(0,1,vtype=gurobipy.GRB.CONTINUOUS,name="activ_"+str(i)+","+str(j)) for j in range(self.inst.numCapLevels)] for i in range(self.inst.numFacs)]
            # Integrate new variables
            model.update()
            # Set objective
            model.setObjective(sum([self.inst.demPtWts[i]*self.inst.pairsUtilityMatrix[i][j]*assignVars[i][j] for j in range(self.inst.numFacs+1) for i in range(self.inst.numDemPts)]), gurobipy.GRB.MAXIMIZE)
            model.update()
            capacitySumsForEachFac = [sum([(l/float(self.inst.numCapLevels)) * float(self.inst.capacities[j]) * activationVars[j][l] for l in range(self.inst.numCapLevels)]) for j in range(self.inst.numFacs)]
            capacityConstraints = [model.addConstr(sum([self.inst.demPtWts[i]*assignVars[i][j] for i in range(self.inst.numDemPts)]) <= capacitySumsForEachFac[j], "capacity_"+str(j)) for j in range(self.inst.numFacs)]
            for i in range(self.inst.numDemPts):
                model.addConstr(sum([assignVars[i]  [j] for j in range(self.inst.numFacs+1)]) == 1, "demand_met"+str(i))
            for j in range(self.inst.numFacs):
                for l in range(self.inst.numCapLevels):
                    for m in range(self.inst.numAllocLevels):
                        model.addConstr()
            model.update()
            #model.setParam('OutputFlag', False )
        except gurobipy.GurobiError:
            print 'Error reported'
        return model, capacityConstraints
    

print "start"
prob = RecourseProblemRandomDraw()
prob.createRandomDrawModel()