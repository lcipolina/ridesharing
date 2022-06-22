#!/usr/bin/env python
from itertools import combinations
import math
import bisect
import sys
import datetime
import numpy as np


def power_set(List):
    # It returns them as a simple list, like this: [[0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]]
    PS = [list(j) for i in range(len(List)) for j in combinations(List, i+1)]
    return PS
    

def getShapley(n,coalition_lst,characteristic_function):
    '''
    It returns how to split the value of the Grand Coalition
    '''

    if len(characteristic_function) < 2:
        #print ('input characteristic_function less than 2')
        return characteristic_function

    if n == 0:
        print ("No players, exiting")
        sys.exit(0)


    N = coalition_lst

    shapley_values = []
    for i in range(n):
        shapley = 0
        for j in N:
            if i not in j:
                cmod = len(j)
                Cui = j[:]
                bisect.insort_left(Cui,i)
                l = N.index(j)
                k = N.index(Cui)
                temp = float(float(characteristic_function[k]) - float(characteristic_function[l])) *\
                           float(math.factorial(cmod) * math.factorial(n - cmod - 1)) / float(math.factorial(n))
                shapley += temp
                # if i is 0:
                #     print j, Cui, cmod, n-cmod-1, characteristic_function[k], characteristic_function[l], math.factorial(cmod), math.factorial(n - cmod - 1), math.factorial(n)

        cmod = 0
        Cui = [i]
        k = N.index(Cui)
        temp = float(characteristic_function[k]) * float(math.factorial(cmod) * math.factorial(n - cmod - 1)) / float(math.factorial(n))
        shapley += temp

        shapley_values.append(shapley)

    return shapley_values


if __name__ == '__main__':

    '''
    Arguments:
    (Number of players in the first row and valuations for all the coalitions in the second row except the empty set. Valuations are ordered in this manner: 
    v(1),..,v(n),v(12),..,v(1n),v(23)..,v(2n),..,v(n-1n),v(123),v(124),.,v(12n),v(234)..,..,v(12..n) )  

    Hay que pasarle el 'n'
    # y los valores de las particiones/coaliciones en este orden:
    [[0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]]
    It prints out the payments for each. Ex, if N = 3, prints: {0,1,2}

    OBS: It assumes that every coalition I pass is the "grand coalition"
    '''


    #*******************************************************************
    # Example:
    '''
    How it needs the inputs:
    n = 3
    pwr_set =[ [0], [1],    [2],   [0, 1], [0, 2], [1, 2], [0, 1, 2]] --> it needs CONSECUTIVE numbers! there is no way aroudn (for now)
    pmt     = [100, 127.28, 56.57, 117.4,   90.39,  99.38,   132,19]]    
    '''
    lst = []
    for i in list(range(450)):
        n = 3
        tempList = list([i for i in range(n)]) #needs to start at zero and be consecutive (at least for now)

        begin_time = datetime.datetime.now()
        coalition_lst= power_set(tempList) #creates all the possible partitions

        characteristic_function = [100, 127.28, 56.57, 
                                    117.4, 90.39, 99.38, 
                                    132,19        
                                    ]


        print(getShapley(n, coalition_lst,characteristic_function))
        lst.append(datetime.datetime.now() - begin_time)

    print('mean time',np.mean(lst))         