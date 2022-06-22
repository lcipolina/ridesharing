# Returns partition lst of feasible coalitions for each agent
# following Vahid's methodology

import numpy as np
from scipy.spatial import distance
from more_itertools import set_partitions
import itertools

import Weiszfeld  #to calculate geometric median
import get_center
import walking_dist



'''
    * Calculate the Total Distance Matrix
    * For every point in the distance matrix
    * Discard negatives and sort from min dist to max
    * Start adding one by one: (calculate center, take the Eps' and see if it's feasible for everyone) 
     until stopping rule: 
       - either we have 4 already
       - either A cannot join
       - what if some other person now cannot join ?
         - drop this person and see if there is anotherone close to A and able to join the coalition
'''



def calc_partitions(feas_partition_lst):
    part_lst = []
    for iCol in feas_partition_lst:
        partition_lst = list(set_partitions(iCol))
        part_lst.append(partition_lst[0])    
    return part_lst


def calc_tot_dist(origin,destination):
    # Calculate a *total distance matrix* across all points    (a 4D distance matrix)
    origin_dist_matrix  = distance.cdist(origin, origin, 'euclidean')
    dest_dist_matrix    = distance.cdist(destination, destination, 'euclidean')
    return  origin_dist_matrix + dest_dist_matrix



def get_feasible_coalitions(origin, destination, epsilon_vct,maxpoolers=4):
    coalitions_lst  = get_coalitions(origin, destination, epsilon_vct,maxpoolers)
    return coalitions_lst 


def get_coalitions(origin, destination, epsilon_vct,maxpoolers=4):
    ''''
    Calculates neighborhood points within a radious in a 4D spere
    :origin: 2D matrix with the (x,y) at origin
    :destination: 2D with the (x,y) at destination

    :Returns: a Dictionary (because the length of each coalition can be different)

    KEY: don't discard negatives because the diff among points is too punitive
    compare one by one and see whit whom I can match
    calculate center, take the Eps' and see if it's feasible for everyone (if we have 4 stop)

    discard coalitions with too many maxpoolers
    '''

    # Punitive step *************
    # Calculate a *total distance matrix* across all points    (a 4D distance matrix)
    total_dist_matrix   = calc_tot_dist(origin,destination)
     # Get points within a radius for every point
    diff_m = np.vstack(np.array(epsilon_vct))-total_dist_matrix  #wants to walk vs needs to walk (subtract vct to every col in M )
    #Corrections for our usecase
    np.fill_diagonal(diff_m, np.zeros((diff_m.shape[0]))) #correct values on diagonal should be zero  
    diff_m = np.absolute(diff_m)  #convert to absolute distance    

    # Comparison against Epsilon  **************
    resLstfinal = []
    for iRow in range(len(epsilon_vct)): # iRow is the person in the matrix

        res_lst = []
        singleton_lst = []
        res_lst.append(iRow)
        # get neighbors (remember this is too punitive)
        row = diff_m[:,iRow]  
        sorted_dist_idx = np.argsort(row) # returns the original idx that makes a sorted array (so we dont destroy the original array)
                           
        for idx in sorted_dist_idx: #idx is the other person I am trying to match with

            if (iRow != idx):     #do not compare with itself               

                if len(res_lst) <maxpoolers:  
                    res_lst.append(idx) #add person and see if we can pool them with the ones already in pool

                    # join idx to the group and walk to the median center 
                    walk_required_lst = walking_dist.weighted_walking_dist(origin[res_lst], destination[res_lst],alpha = 10, base = 10,mode = 'weighted' )  #walking distance to the median
    
                    # need to see if every epsilon can walk 
                    diff_vct = epsilon_vct[res_lst]- walk_required_lst  

                    if len(diff_vct[diff_vct < 0]) !=0: #if the array of negative numbers is not empty
                        res_lst.remove(idx) #adding this person was a bad idea  
                        singleton_lst.append(idx) #add it as a singleton   
        
        intermediate = []                
        if singleton_lst:
           for el in singleton_lst:  #each singleton should be alone
               intermediate.append([el])                     
           intermediate.append(sorted(res_lst))
        else:
            intermediate.append(sorted(res_lst))  

        resLstfinal.append(intermediate) # gives back the coalition per person (i.e. per row)  

    # Remove duplicates before final return  
    resLstfinal.sort()
    result = list(resLstfinal for resLstfinal,_ in itertools.groupby(resLstfinal))
    return result


# EXAMPLE OF USAGE:
if __name__ == "__main__":  

    
    # Dummy example
  
    origin = np.array([
                        [0, 0],
                        [1, 1],
                        [5, 5] 
                        ])

    dest = np.array([
                    [0, 10],
                    [10, 10],
                    [9, 9] 
                    ])     

    epsilon_vct = np.array([100,100,100])
    print('epsilon_vct:',epsilon_vct)
    
    '''
    # Random coordinates
    origin =   np.array([[0.44402056, 1.14255782],
                        [0.48921279,7.6889942 ],
                       [5.38275467, 8.0628176 ]])

    dest = np.array( [[2.14675564, 1.92103719],
                      [9.53352377, 3.43183131],
                      [9.13178385, 8.03446556]])            
    
    mid_dist  = distance.cdist(origin, dest, 'euclidean') # people willing to make half of the trip by foot
    epsilon_vct   = np.array(mid_dist.diagonal())
    print('epsilon:', epsilon_vct)
   '''
    
    
    result = get_feasible_coalitions(origin, dest, epsilon_vct,maxpoolers=4)   
    print('coalitions formed: ',result)   
  