
from scipy.spatial import distance
import numpy as np

def weighted_walking_dist(origin, destination, mid_origin_coords, mid_dest_coords, alpha = 1.015,Pcar = 2, mode ='weighted' ):
    '''''
    CALCULATES THE VoT
    Total walking dist = walk from self's origin to car + from car to self's destination    
    Returns the weighted walking distance (in distance)
    The weight is an additive term that increases with the length waled
    '''   
      
    # Total walking cost per person   
    walk_dist_orig  = distance.cdist(origin, np.tile(mid_origin_coords, (3, 1)), 'euclidean') 
    walk_dist_dest  = distance.cdist(destination, np.tile(mid_dest_coords, (3, 1)), 'euclidean') 
    walk_dist_lst =  walk_dist_orig[:,1] + walk_dist_dest[:,1]
    # print('unweighted_walking:',np.around(walk_dist_lst,2) )

    if mode == 'unweighted':
       return walk_dist_lst

    else:
       # Weighting formula: walking distance + walking distance * log(base) / alpha
       #result  = walk_dist_lst +  walk_dist_lst * (np.log(walk_dist_lst+0.0001) / np.log(base))/alpha  
       #Simplified walking funct = (Pcar)*(dist(.)^alpha)
       result = ((np.array(walk_dist_lst))**alpha)*Pcar  #this is the VoT of the paper

       return  (np.array(result)) #VoT
       #Returns array with weighted distance walked by each (in distance units)  - this is the VoT of the paper - 
      
             
   
      
    