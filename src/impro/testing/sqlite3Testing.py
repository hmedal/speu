import sqlite3 as lite
import sys

con = None
databaseName = "/home/hmedal/Documents/2_msu/1_MSU_Projects/facilityProtection_CSF/expr_output/impro-results.db"

try:
    con = lite.connect(databaseName)

    cur = con.cursor()  

    cur.executescript("""
        DROP TABLE IF EXISTS ProbChain;
        CREATE TABLE ProbChain(Id INT, Alg TEXT, RunTime DOUBLE, ObjValue DOUBLE);
        """)

    con.commit()
    
except lite.Error, e:
    
    if con:
        con.rollback()
        
    print "Error %s:" % e.args[0]
    sys.exit(1)
    
finally:
    
    if con:
        con.close() 