import numpy as np
import ast

   

def set_partitions(set_):
    '''
    Input: coalition as a list. Ej: [1,2,3]
    Calculates sets partitions of up to 4. Ej: [1],[1,2],[2],[2,3],[3]
    https://stackoverflow.com/questions/29762628/partition-of-a-set-or-all-possible-subgroups-of-a-list
    https://stackoverflow.com/questions/19368375/set-partitions-in-python
    '''

    # I think it's easier to calculate everything and then discard what I don't need when I use it!
    if not set_:
        yield []
        return
    for i in range(int(2**len(set_)/2)):
        parts = [list(), list()]
        for item in set_:
            parts[i&1].append(item)
            i >>= 1
        for b in set_partitions(parts[1]):
            yield [parts[0]]+b  
    return        




#************************* WRONG **********************************************************
#******************************************************************************************
def calc_parts_save_txt(array,maxPoolers = 4):
    '''
    Taken from:
    https://stackoverflow.com/questions/25458879/algorithm-to-produce-all-partitions-of-a-list-in-order/25460561
    THIS IS ACTUALLY WRONG! 
    Test if for [1,2,3]-> we should get 5 partitions but we are getting 4, it is missing [[1,3][2]]
    FUNCTION TO BE USED WHEN we want to store and retrieve from TXT file (too big for memory)
    Input: coalition as a list. Ej: [1,2,3]
    Calculates sets partitions of up to 4. Ej: [1],[1,2],[2],[2,3],[3]
    Saves them into a txt file -> this is handy when number of partitions is larger than memory.
    '''

    print('Entering calc_parts_save ')    

    # Clear previous data on file
    fname = 'partition_lst.txt'
    open(fname, 'w').close() 

    # Calculate partitions
    n = len(array)
    for partition_index in range(2 ** (n-1)):
        # current partition, e.g., [['a', 'b'], ['c', 'd', 'e']]
        partition = []
        # used to accumulate the subsets, e.g., ['a', 'b']
        subset = []
        for position in range(n):
            subset.append(array[position])
            # check whether to "break off" a new subset
            if 1 << position & partition_index or position == n-1:
                partition.append(subset)
                subset = []

        # Prints into a separate file
        # For each partition, it writes sequentially into the file. It doesn't write all partitions in one go, but one by one.
        # This is a trick to avoid holding all the partitions in memory
        res = []
        for ele in partition: #if any of the coalitions in the Coal Struct has <=4, then print to file
            if len(ele) > maxPoolers:
               res.append(1)

        if (np.sum(res) < 1):
            with open(fname,'a') as f: 
                    print(partition, file=f) 
    return #res     


def set_partitions_txt(arr,maxPoolers = 4,fname = 'partition_lst.txt'):
    '''
    THIS IS ACTUALLY WRONG! 
    Test if for [1,2,3]-> we should get 5 partitions but we are getting 4, it is missing [[1,3][2]]
    storing things in files generally takes a longer time than putting things in memory. 
    but might sometimes be necessary with very large amounts (more than you got memory).
    Memory is pretty much always faster than writing and loading files.
    '''

    calc_parts_save_txt(arr, maxPoolers) #saves the partitions into a file

    print('Entering reading the file ')   
    with open(fname) as file:
         res = [ast.literal_eval(line) for line in file]
    return res 
 


if __name__ == '__main__':
    ar = [1,2,3]
    ar = list(range(10))
  
    #This one saves and reads from file (when array too big for memory)
    #calc_parts_save_txt(ar)
    #print(set_partitions_txt(ar) )
    lst = []   
    for p in set_partitions(ar):
        lst.append(p)
    print(lst)


 
#  USING ITERTOOLS 
# from more_itertools import set_partitions
# feas_partition_lst = [1,2,3]
# feas_partition_lst =list(range(30))  #this one won't work as it calculates everything inside and doesn't have enough RAM
# partition_lst = list(set_partitions(feas_partition_lst))
  
