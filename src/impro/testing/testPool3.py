import multiprocessing
from time import time

K = 500
def CostlyFunction((z,)):
    r = 0
    for k in xrange(1, K+2):
        r += z ** (1 / k**1.5)
    return r

def CostlyFunction2(z):
    r = 0
    for k in xrange(1, K+2):
        r += z ** (1 / k**1.5)
    return r

if __name__ == "__main__":
    currtime = time()
    N = 10000
    print "num cpus: ", multiprocessing.cpu_count()
    po = multiprocessing.Pool()
    res = po.map_async(CostlyFunction2,[i for i in xrange(N)])
    w = sum(res.get())
    print w
    print '2: parallel: time elapsed:', time() - currtime