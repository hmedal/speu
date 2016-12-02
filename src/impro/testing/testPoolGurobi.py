from multiprocessing import Pool
import gurobipy

m = None

def  f(x):
    m.getConstrs()[0].setAttr("rhs", 1.0 + x)
    m.optimize()
    return m.objVal
    
if __name__ == '__main__':
    global m
    # Create a new model
    m = gurobipy.Model("model")

    # Create variables
    x = m.addVar(vtype=gurobipy.GRB.BINARY, name="x")
    y = m.addVar(vtype=gurobipy.GRB.BINARY, name="y")
    z = m.addVar(vtype=gurobipy.GRB.BINARY, name="z")

    # Integrate new variables
    m.update()

    # Set objective
    m.setObjective(x + y + 2 * z, gurobipy.GRB.MAXIMIZE)

    # Add constraint: x + 2 y + 3 z <= 4
    m.addConstr(x + 2 * y + 3 * z <= 4, "c0")

    # Add constraint: x + y >= 1
    m.addConstr(x + y >= 1, "c1")
    m.update()
    m.setParam('OutputFlag', False )
    pool = Pool(processes=4)              # start 4 worker processes
    output = pool.map(f, range(1000))
    print output
    print sum(output)