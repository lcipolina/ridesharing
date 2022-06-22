
'''
************ Returns partition lst of feasible coalitions for each agent  ****
Meaning, groups that are closer to each other

For each agent, calculates a radius of epsilon around them
The radius is calculated considering an epsilon around the TOTAL distance = Origin+Destination
Returns all other agents that are within that radius
  for all other agents, see if they fall within this total epsilon
Then for all agents within an epsilon distance of each other..
it calculates all the coalitions of *up tp* 4 members
As we need to see which of these are the optimals (in terms of SO)


Returns something like this: in this case, (1,2) are potential matches

 0       1     2  ->  who are we looking at
[empty   2     1]  -> who is a potential match
'''



import numpy as np
from scipy.spatial import distance
from more_itertools import set_partitions
import itertools



def calc_partitions(feas_partition_lst):
    part_lst = []
    for iCol in feas_partition_lst:
        partition_lst = list(set_partitions(iCol))
        part_lst.append(partition_lst[0])    
    return part_lst


def calc_tot_dist(origin,destination):
    # Calculate a *total distance matrix* across all points   = distance at origin + distance at dest
    origin_dist_matrix  = distance.cdist(origin, origin, 'euclidean')
    dest_dist_matrix    = distance.cdist(destination, destination, 'euclidean')
    return  origin_dist_matrix + dest_dist_matrix


def get_coalitions(origin, destination, epsilon_vct):
    ''''
    Returns a list of people closer to each other.
    Where 'closer' means people within an epsilon radius of (d_origin + d_dest)
    
    origin: 2D matrix with the (x,y) at origin
    destination: 2D with the (x,y) at destination    
    '''

    # Calculate a *total distance matrix* across all points (d_origin + d_dest)
    total_dist_matrix   = calc_tot_dist(origin,destination)

    print('total_dist_matrix',total_dist_matrix)

    # Get points within a radius for every point
    diff_m = np.vstack(np.array(epsilon_vct))-total_dist_matrix  #wants to walk vs needs to walk (subtract vct to every col in M )
    #Corrections for our usecase
    np.fill_diagonal(diff_m, np.zeros((diff_m.shape[0]))) #correct values on diagonal should be zero  
 
    print('diff_m', diff_m)

    # Returns the indices of the elements having positive values                 
    feas_partition_lst = []
    for iRow in range(len(epsilon_vct)):
            row = diff_m [:,iRow]
            index = [i for i, x in enumerate(row) if x > 0]
            if index: #ignore empty rows (i.e. no potential matches)
               index.append(iRow)  # returns pairs of agents colluding together
               feas_partition_lst.append(sorted(index)) 


    # Remove duplicates before final return  
    feas_partition_lst.sort()
    result = list(feas_partition_lst for feas_partition_lst,_ in itertools.groupby(feas_partition_lst))  
    return  result #return a list of list of people closer to each other


def get_feasible_coalitions(origin, destination, epsilon_vct,maxpoolers=4):
    '''
    Get a list and calculates all its partitions
    '''
    print('Getting feasible coalitions')

    coalition_lst  = get_coalitions(origin, destination, epsilon_vct) # brings who is closer to whom 
    #partitions_mtx = calc_partitions(coalition_lst) #get combinations - this is now calculated later on
    return coalition_lst


# EXAMPLE OF USAGE:
if __name__ == "__main__":  

    origin = np.array([
                        [0, 0],
                        [1, 1],
                        [5, 5], 
                        [6, 6],  
                        [4, 4] 
                        ])

    dest = np.array([
                    [0, 10],
                    [10, 10],
                    [9, 9], 
                    [11, 11], 
                    [9, 9]
                    ])     

    # epsilon is the same for everyone as we are just partitioning the space
    # this epsilon is compared against (d_origin + d_dest)
    epsilon_vct =  np.array([15]*len(origin))  # big epsilon to match everyone with everyone
    
    result = get_feasible_coalitions(origin, dest, epsilon_vct,maxpoolers=4)   

    print(result)  


    

        
