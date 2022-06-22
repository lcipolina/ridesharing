from more_itertools import set_partitions

'''
TODO: this doesn't run with N = 50
The problem is the set_partitions function, that needs to calculate all permutations
So it is practically not used
'''




def get_feasible_coalitions(origin, destination, epsilon_vct, pRide = 1, pWalk = 1, policy = 'even', metric = 'median', maxPoolers = 4):
      '''
      Creates a prunning of the coalitions based on
      1. number of members
      2. max acceptable distance
      '''

      print('calculating partitions')

      # Get all possible partitions of a set (i.e. coalitions)
      N = len(origin)
      N_lst = list(range(N))
      partition_lst = list(set_partitions(N_lst)) #returns list of partitions (inside a partition are coalitions)

      print('calculating feasibility of all')

      # Whether coalition formed under a partition is feasible
      feas_partition_lst   =[]
      for ele in partition_lst:    #for each sub group within a coalition       
          for iCol in ele: #iCol is the column of the partition_lst (iCol= the coalition)
              # Length restriction
              if len(iCol) <= maxPoolers:  
                  # Walk restriction: (w_origin(i) + w_dest(i)) < epsilon(i)
                  walk_dist_lst = self.walking_dist(self.origin[iCol], self.destination[iCol])
                  
                  if len(walk_dist_lst) >1:  #only compare with epsilon if carpooling
                     diff_vct = np.array(self.eps_vct)[iCol]-walk_dist_lst #wants to walk vs needs to walk
                     
                    #TODO: I don't get what I did here with the np.array, seems like: np.array(... icol)

                     if len(diff_vct[diff_vct < 0]) ==0: #if the array of negative numbers is empty
                        feas_partition_lst.append(ele)

                  #VECTORIZED #TODO
                  #mid_origin_coords,mid_dest_coords =  self.get_center(self.origin[iCol], self.destination[iCol])  #coordinates to the center
                  #mid_origin_coords_array = [np.array(mid_origin_coords)]*(self.origin[iCol].shape[0])
                  #walk_d_orig_matrix   = distance.cdist(self.origin[iCol], mid_origin_coords_array, 'euclidean') #walk per agent- everyone walks to midpoint 
  
      return {'feas_partition_lst':feas_partition_lst,'walk_dist_lst':walk_dist_lst} #save the distances, handy for later
