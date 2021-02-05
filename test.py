
import multiprocessing
from multiprocessing.pool import ThreadPool
pool = ThreadPool(10)

count = [0]

def f(i):
    res = 0
    for iteration in range(100):
        res += i*2
    return res

print(pool.map(f,range(1000)))



