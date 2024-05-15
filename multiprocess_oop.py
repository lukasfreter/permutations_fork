#!/usr/bin/env python

from multiprocessing import Pool
from time import sleep, time
import numpy as np

class A:
    def __init__(self):
        self.big_data = np.arange(1e7) # 76MB
        pass

    def one_task(self, x):
        sleep(x)
        return x

    def multiple_tasks(self):
        args = [0.1 for i in range(120)] # 120 sleeps of 0.1 sec (~1sec for 12 Cores)
        t0 = time()
        print('Start multiprocess in a class...')
        with Pool() as p:
            resuls = p.map(self.one_task, args)
        print('Finish {:.1f}s'.format(time()-t0))

testA = A()
testA.multiple_tasks()

class B:
    def __init__(self):
        self.big_data = np.arange(1e9) # 76MB
        pass

    @staticmethod
    def one_task(x):
        sleep(x)
        return x

    def multiple_tasks(self):
        args = [0.1 for i in range(120)] # 120 sleeps of 0.1 sec (~1sec for 12 Cores)
        t0 = time()
        print('Start multiprocess in a class (STATIC METHOD)...')
        with Pool() as p:
            resuls = p.map(self.one_task, args)
        print('Finish {:.1f}s'.format(time()-t0))

testB = B()
testB.multiple_tasks()

def one_task(x):
    sleep(x)
    return x

class C:
    def __init__(self):
        self.big_data = np.arange(1e9) # 76MB
        pass

    def multiple_tasks(self):
        args = [0.1 for i in range(120)] # 120 sleeps of 0.1 sec (~1sec for 12 Cores)
        t0 = time()
        print('Start multiprocess in a class (EXTERNAL FUNCTION)...')
        with Pool() as p:
            resuls = p.map(one_task, args)
        print('Finish {:.1f}s'.format(time()-t0))

testC = C()
testC.multiple_tasks()
