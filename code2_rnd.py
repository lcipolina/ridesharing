#!/usr/bin/python

#solve the group allocation problem

# GIVENS:
#  - Number of passengers: 10
#  - Number of cars available = nbr of passengers
#  - Origin location (in coordinates)
#  - Destination location  (in coordinates)
#  - Taxi fare per kilometer: 10 pounds per km


# TO DETERMINE:
# - Which group to join (solo, or any other formed group)
# - cost(i) = alpha.Walk + beta.(Travel) - > weighted waltk +car cost == value of coalition
# - We need to determine each component of the cost

# - Additionally:
#    - no more than 4 people in a car
#    - everyone should be allocated

# OBJECTIVE:
# - Determine stable matching

# DISTANCE CALCULATION
# For simplicity, we can assume that everyone travels on straight line
# otherwise, the distance is just the Euclidean distance with 2 coordinates.

# ************************************************
# Save output to a file (output getting too long)
import datetime
import gc
import sys

#stdoutOrigin=sys.stdout
#sys.stdout = open("/Users/lucia/Desktop/Southampton/00-Codes/log.txt", "w") # SAVE ALL PRINTOUTS TO FILE


# **************************************************************************************************
# IMPORTS
import numpy as np
import pandas as pd
import math
import ast
from scipy.spatial import distance
from sklearn.datasets import make_blobs
from random import randint
#import Weiszfeld  #to calculate geometric median
import print_to_file
import feasible_coal_dbscan4D as feasible_coal
import get_center
import walking_dist
import riding_cost as ride_cost
import partitions
import matplotlib.pyplot as plt



# **************************************************************************************************
# CLASSES ****************************************************************************************
# **************************************************************************************************

class RideShare:
  def __init__(self, origin, destination, epsilon_vct, pRide = 1, policy = 'even', metric = 'median', maxPoolers = 4, alpha = 1.015,flag_fall = 0.05):
    # some of these are overriden later for convenience
    self.origin      = origin
    self.destination = destination
    self.pRide       = pRide
    self.policy      = policy
    self.metric      = metric
    self.eps_vct     = epsilon_vct
    self.maxPoolers  = maxPoolers
    self.alpha       = alpha        # convexity of the walking VoT function
    self.flag_fall   = flag_fall    # pct of flag fall
     


  def calc_partitions(self,feas_partition_lst):
    '''
    Calculates all possible partitions of a list
    Returns only those lists with less than 4 members (maxPoolers)
    Input: list with integers  = [1,2,3,4,5,6...]
    '''
    print('Entering set_partitions')
    partition_lst = partitions.set_partitions(feas_partition_lst)
    return partition_lst


  def solo_travel_cost(self):
      '''
      Inputs the origin and destination matrices
      We assume that people who travel alone can either walk full distance or ride full distance
      Returns the min(solo ride, solo walking)
      '''
      # Calculates the cost of travelling alone for each passenger
      # For each member, calculate travelling time * price_taxi
      dist_matrix  = distance.cdist(self.origin, self.destination, 'euclidean') # distances travelled alone
      cost_matrix      = dist_matrix*self.pRide
      return cost_matrix.diagonal()  # HERE TODO: min(car, walking) -> hold on, because walking is more expensive than car (it takes more time)


  def pmt_policy(self,walk_lst):
      '''
      walk_lst: distance walked by every agent (in distance)
      '''
      # How to weight the portion of fare to pay
      # how to compensate for the walking time
      ones_vect = np.ones(len(walk_lst))
      if self.policy == 'even':
         res = (ones_vect*(1/len(walk_lst)))

      if self.policy == 'weighted':
         walk_weight = (walk_lst/ np.sum(walk_lst)) + 0.000001  #if walking weight = 0 added small term
        # print('walking_pct', list(np.around(np.array(walk_weight),2))) #rounding for display
         coeff =  np.sum([1/ele for ele in walk_weight]) # PMT weight: [1/Wi * coeff], coeff =  [Sum(1/Wi]^-1 * X
         res = [(1/ele)*(1/coeff) for ele in walk_weight]

      #print('payment_pct', list(np.around(np.array(res),2)))
      return np.array(res) #pct_of inverse walking


  def walking_riding_cost(self, origin, destination):
      '''
      Calculates Walking cost, Car Cost and central O-D points
      '''
      # Returns center of Origin and center of Dest based on chosen metric
      mid_origin_coords,mid_dest_coords = get_center.get_center(origin, destination, metric =self.metric)  
     
      # VoT = Total weighted walking distance per person (walking at origin + walk at destination) - in distance (not %)
      weighted_walk_lst  = walking_dist.weighted_walking_dist(origin, destination, 
                                                                mid_origin_coords, mid_dest_coords,
                                                                alpha = self.alpha, Pcar = self.pRide,
                                                                mode=self.metric)

      #Car cost  - single trip price that needs to be then split among carpoolers
      riding_cost = ride_cost.riding_cost(mid_origin_coords, mid_dest_coords, pRide = self.pRide)  #single number 

      return weighted_walk_lst, riding_cost, [mid_origin_coords,mid_dest_coords] #VoT of walking (not in pct), Pcar, central O-D pts


  def rideshare_cost(self, origin, destination):
      '''
      :input: simply the orgin and dest is enough to calc all costs
      Brings Walking cost, Car Cost and Allocates the PMT based on waling
      Splits the cost among members according to a pmt policy
      :returns: COST PER PERSON(i) = tot_walk_distance(i) + riding_dst / (pricing policy)
      '''

      #print('Entering rideshare_cost')

      # ************************************************************ 
      # Brings all costs
      # ************************************************************ 
      weighted_walk_lst, riding_cost, mid_coords = self.walking_riding_cost(origin, destination)  # VoT (Tot walk cost), Pcar, [meeting corrds]

      # ************************************************************ 
      # Splits the Car costs according to Pmt Policy
      # ************************************************************ 

      # Riding alone - Nothing to split  
      if (sum(weighted_walk_lst) ==0):  #no walking (i.e. riding alone)
          weighted_walk_lst = [0] #same dimensions to unpack as the other return
          return (np.array([riding_cost]) , mid_coords,weighted_walk_lst)  # ([Xorigin, Yorigin, Xdest, Ydest]) for the coalition (i.e. the coords of the meeting point) for Ian's trick

      # Coal with >1, split car according to Policy
      else: #coalition with more than one member
        policy_pct_lst  = self.pmt_policy(weighted_walk_lst) # pct of car pmt for each rider (inverse funct of walking)
        # Total cost of Coalition = (walking(o)+ walking(d) + riding) //riding: base price + distributed price
        flag_fall_pct = self.flag_fall # pct of base P that is distributed evenly
        ones = np.ones(len(policy_pct_lst))
        even_vct = ones*(1/len(policy_pct_lst))
        car_cost_per_member_lst   = (flag_fall_pct*riding_cost)*even_vct + ((1-flag_fall_pct)*riding_cost)* policy_pct_lst #base fare + distributed fare
        walking_plus_riding_cost  =  np.add(weighted_walk_lst, car_cost_per_member_lst)    #total cost per person
    
        return (walking_plus_riding_cost, mid_coords,weighted_walk_lst) #total cost per person + coalition meeting points ([Xorigin, Yorigin, Xdest, Ydest]) for Ian's trick, VoT


  def coalition_value(self, coal_lst):
      '''
      Calculates the value of all possible coalitions per cluster (the input is the elements on a cluster)
      We need to run this per-cluster
      input: list of lists of lists. Like this: [ [[1,2,3],[4]], [[1],[2,3,4] ] -->  list of coalitional structures (each is a Coal structure) on this cluster
      output: a DataFrame with all possible the Coal Struct, and the Tot value (walk +car)
      '''

      # Calculate the value of each coalition (ride+ walk)
      print('Entering coalition_value')

      coal_struct_cost = []             # coal_lst =different coal structs = [ele, ele, ele] // ele = coal structure
      for coal_struct in coal_lst:      # for each coal_struct = [coal,coal] = [[1],[2,3,4]]
          coal_cost  = 0                # gathers total value of a coal_struct (every coal in the structure)
          cost_per_member_lst, coal_struct_lst, coords_lst  = [], [],[]  # cost paid by each member (walk+car_share) per coal member
          skip = 0                      # -reset - if any subset (coalition) has >4, need to skip coal struct

          for coal in coal_struct: # coal is the column of the partition_lst (coal= the coalition = [2,3,4])
            if len(coal) > self.maxPoolers:
               skip = 1 #need to skip the entire coal struct

            # select only coalition w/ up to 4 members. OBS: coalition w/ 1 person = no walking
            if (skip == 0):                   #process each coal separately
                cost_per_member, coords, weight_walk_lst  = self.rideshare_cost(self.origin[coal], self.destination[coal]) #Returns: individual cost (walk+car), center of the coalition: np.arrray([(Xorig,Yorig),(Xdest,Ydest)])
                cost_per_member_lst.extend([cost_per_member])
                coal_struct_lst.extend([coal]) #save coal structures with less than 4 members
                coords_lst.extend([coords])
                coal_cost += sum(cost_per_member)

          # Add all coals in a Struct
          if (coal_struct_lst and skip == 0): #if coal lst has less than 4
             coal_struct_cost.append((coal_struct_lst,round(coal_cost,6), cost_per_member_lst,coords_lst,weight_walk_lst ))  #building the list of tuples for the DF
      #aggregate all clusters in a DF (one row per cluster) 
      df = pd.DataFrame(coal_struct_cost,  columns = ['coal_struct', 'cost','cost_per_member','coords', 'walking'])     # different coal structs and its costs //this is only for one cluster!
      return df
   

  def coalitions_social_welfare(self):
      '''
      It's like the main rider that calls other methods.
      1. Calculate the Feasible Coalitions - acording to an Epsilon = 'clusters'
      2. For each of the 'clusters', it calculates all the possible partitions (i.e. coalitional structure - CS)
      3. For all the possible partitions (coalitional structure), we calculate the value of each coalition inside the C.S
      :returns: A DFrame where the rows are the minimum-cost coalition structure (i.e. SO) per cluster
      '''

      #print('Entering coalitions_social_welfare')

      # Cluster of individuals. Returns Lit of lists which is a list of the clustered individuals (each cluster is a list)
      feas_partition_lst = feasible_coal.get_feasible_coalitions(self.origin, self.destination, self.eps_vct)

      #Calculate value of each coalition inside the cluster
      so_df = [] #list of dataframes
      for cluster in feas_partition_lst:                # for each cluster (=list of individuals)
          partition_lst = self.calc_partitions(cluster) # calculate all combinations. Returns List of lists.
          # All coal structs per cluster and its value
          df_cluster    = self.coalition_value(partition_lst) # Returns DFrame with [(coal structures, values, pmt_per_person, coords)] for this cluster
          # Get SO (min cost) per cluster, the structure and PMT per person         
          so_cluster_df =  df_cluster.loc[df_cluster['cost'] == df_cluster['cost'].min()] #minimum cost coal per cluster
          #store results per cluster into global list //df of social optimums
          if len(so_cluster_df)>1: # Case when >1 coalition have same (equal) min cost - take the one with more members per coalition (i.e. fewer cars)
              so_df.append(so_cluster_df.loc[so_cluster_df['coal_struct'].astype(str)==str(so_cluster_df['coal_struct'].min())])  #This one affects Shapley
          else:    
              so_df.append(so_cluster_df)   

      soDF =  pd.concat(so_df)  # concat lst of clusters into a single DF with each row is the SO of a cluster
      soDF.reset_index(drop=True, inplace=True)  # make sure indexes pair with the row number (too keep the cluster nbr)
      soDF.index.name = 'Idx'
      return soDF #returns a DataFrame where rows are the SO (structures, values) of per cluster


  # ***************************************** SHAPLEY VALUE *********************************************
  def shapley(self, soDF):
      '''
      Takes the SO coalition and distributes Pcar cost according to Shapley
      It  calculates 3 variations of Shapley
      It has a 'mapping trick' to make the code work. 
      '''

      from shapley_value import shapleyLck as shapley

      pmt_lst_w,pmt_lst_d,pmt_lst_t = np.zeros(len(self.origin)),np.zeros(len(self.origin)),np.zeros(len(self.origin))
      shap_comb_lst = []

      # Need to calculate the charact function for the coalitions in the right order
      for row in soDF.itertuples(index=True, name='Pandas'):  # For each cluster = row
          for coal in row.coal_struct:                        # For each coalition

              #Shapley needs power sets of the coal [1,2] (ej of pwr set: [1],[2],[1,2]) and its cost
              pwr_cost_lst,pwr_car_cost_lst, shap_t_cost_lst  = [],[],[] # cost of the pwr sets for Shapley
              coal_pwr_set = shapley.power_set(coal)          # all the possible power sets (different than partitions) / if n = 2 --> [0],[1],[1,2]
              for sub_coal in coal_pwr_set:                   # for each sub_coal of the pwr set / ej: [0]
                  # Build cost lst of the pwr set 
                  weighted_walk_lst, car_cost, coords  = self.walking_riding_cost(self.origin[sub_coal], self.destination[sub_coal]) # #Returns: VoT, PCar, center of the coalition: np.arrray([(Xorig,Yorig),(Xdest,Ydest)])
                  pwr_cost_lst.append((np.sum(weighted_walk_lst) + car_cost)) #collect cost vct of all pwr sets
                  pwr_car_cost_lst.append(car_cost) #collect cost vct of all pwr sets - for method3 (distribute only car cost)
              
              # Calculate coalitions (partitions) the way the code needs (this is bananas! - to amend)
              n = len(coal)
              tempList     = list([i for i in range(n)]) # needs to start at zero and be consecutive (at least for now)
              sub_coal_lst = shapley.power_set(tempList) # creates all the possible dummy partitions
             
              # Distribute the Tot Cost of coalition (car+ walk) according to a rule.
              # 1. Weighted: car_pmt(i) =  P_{car} .\frac{Shapley of i} {Tot cost of coal (walk+Car}
              shap_vals            = shapley.getShapley(n, sub_coal_lst,pwr_cost_lst)  # Calculate shapley  //tempList - note that needs list starting at zero
              shap_weight_car_lst  = car_cost * (shap_vals/(np.sum(shap_vals)))        # List with proportion of car_pmt
              shap_w_cost_arr      = (shap_weight_car_lst + weighted_walk_lst)          # Tot cost. Trick: the original coal is the last pwr set
              #Bananas: Add to Lst in the correct order  (ej, some coals are [3,9] pmt need to go on place 3 and 9 of lst)
              for values in list(zip(coal, shap_w_cost_arr)):
                  idx, value     = values
                  pmt_lst_w[idx] = value     # pmts of all 'i' under the same coal struct of a cluster            
              
              # 2. this gives pmt of car directly (then added the walking to get Tot Pmt)
              shap_car           = shapley.getShapley(n, sub_coal_lst,pwr_car_cost_lst)
              shap_d_cost_arr    = (shap_car + weighted_walk_lst)   # Add the walking cost of the original optimal coal - not the subcoals-  (which is always the last one of the pwr set! )
              #Bananas: Add to Lst in the correct order  (ej, some coals are [3,9] pmt need to go on place 3 and 9 of lst)
              for values in list(zip(coal, shap_d_cost_arr)):
                  idx, value     = values
                  pmt_lst_d[idx] = value     # pmts of all 'i' under the same coal struct of a cluster 

              # 3. default - uses Shap value as tot travel cost for a person (car+walking)
              shap_t_cost_lst    = shapley.getShapley(n, sub_coal_lst,pwr_cost_lst)             
              for values in list(zip(coal, shap_t_cost_lst)):
                  idx, value     = values
                  pmt_lst_t[idx] = value    # pmts of all 'i' under the same coal struct of a cluster
    
      # Aggregate all clusters (i.e. rows) together -one row for each individual (sorted by indiv number)
      data = np.array([pmt_lst_t,pmt_lst_d,pmt_lst_w]).T
      df = pd.DataFrame(data,  columns = ['shap_tot', 'shap_car','shap_weight'])     # different coal structs and its costs //this is only for one cluster!       
      return df  #returns Tot Cost for each individual (Walking VoT cost + share of Pcar)

  #*****************************************************************************************************
  # ***************************************** third step: AGGLOMERATION ******************************** 
  def agglomeration(self, soDF, e2 = 4 ):
    '''
    Agglomerates clusters and finds the new SO.
    input: a DFrame where the rows are the SO per cluster and the epsilon1, epsilon2
    This is a "approx" reclustering where "numbers" in the cluster are previously formed coalitions.
    '''

    # Unpack the mess of the DF
    lst, orig, dest   = [], [], [] # store coalition data to form a DFrame
    for row in soDF.itertuples(index=True, name='Pandas'): #For each cluster (i.e DF row)
        # Build rows of new DFrame
        for coalIdx, coalition in enumerate(row.coal_struct):  # for every coal in the struct
            cost_per_member_lst = row.cost_per_member[coalIdx] # need to keep it for the Indiv Rat
            walking_lst         = row.walking                  # VoT of walking per member   
            cost  = np.sum(cost_per_member_lst)                # total cost of the coal
            orig.append(row.coords[coalIdx][0])                # DBSCAN needs coords as a 4D vector: [x_o, y_o, x_d, y_d]
            dest.append(row.coords[coalIdx][1])
            lst.append((row.Index,coalition,cost,cost_per_member_lst,len(coalition),walking_lst )) #building tuple for the DFrame

    # New unpacked DFrame
    df_simple = pd.DataFrame(lst,  columns = ['clst_idx', 'coal','cost','cost_per_member','len','walking'])
    df_simple.index.name = 'Idx'
    # Reset variables (otherwise I get confused)
    coalition, cost, cost_per_member_lst,walking_lst = 0, 0, 0, 0

    ####### ReClustering - New DBSCAN  - needs coordinates from the formed clusters #############
    o, d = np.array(orig) , np.array(dest) #DBSCAN needs coords as a 4D vector: [x_o, y_o, x_d, y_d]
    feas_partition_lst = feasible_coal.get_feasible_coalitions(o, d, e2) #list of Idx over the new DFrame #rets: [[0,1],[2]]

    ###### COST OF JOINT TRIP - Uses original coordinates over  #################################
    so_lst = []
    for cluster in feas_partition_lst: # ex: cluster [0,1], where nbrs are the idx on the second DFrame
        all_coal_struct_lst = self.calc_partitions(cluster) # combination within elements of the new cluster
        # Value of the coal struct is the sum of value of each coalition - For final DF
        cluster_res_lst = []
        for coal_struct in all_coal_struct_lst: # Merging of coals on the dummy DFrame
            cost_per_member_lst, coal_struct_lst,walking_lst = [],[],[]
            coal_struct_cost,skip = 0,0
            for coal in coal_struct:  # 'coal' is the idx in the new (dummy) Df
                if (np.sum(df_simple.iloc[coal]['len']) > self.maxPoolers):  # if merging brings > 4 members, discard
                   skip = 1 #need to skip the entire coal struct
                if (skip == 0):                              # cost of coalition <4
                   if(len(coal) ==1):                        # Existing coalition -> BRING their value
                     coal_struct_lst.extend([df_simple.iloc[coal]['coal'].item()])
                     cost_per_member_lst.extend(list(df_simple.iloc[coal]['cost_per_member'].item()))
                     coal_struct_cost += np.around(np.sum(sum(df_simple.iloc[coal]['cost_per_member'])),3) #sum of vals from a pd Series
                     walking_lst.extend([df_simple.iloc[coal]['walking'].item()]) 
                   else:                                     # New merged coalition -> CALCULATE their value
                     coalition = np.concatenate([ele for ele in df_simple.iloc[coal]['coal']])  # Real coalition struct
                     coal_struct_lst.extend([list(coalition)])
                     walking_plus_riding_cost, mid_coords,weighted_walk_lst = self.rideshare_cost(self.origin[coalition], self.destination[coalition])
                     cost_per_member_lst.extend(walking_plus_riding_cost) # individual pmt
                     coal_struct_cost += np.round(sum(cost_per_member_lst),3)
                     walking_lst.extend(weighted_walk_lst)

            # Add all Coal Struct (<4) per new agglom cluster - building tuple for the DFrame
            if (coal_struct_lst and skip == 0):
               cluster_res_lst.append((coal_struct_lst,coal_struct_cost,cost_per_member_lst,walking_lst))

        ###### Keep the Struct with MIN COST PER CLUSTER - SO #################################
        if cluster_res_lst:
            df_pre = pd.DataFrame(cluster_res_lst,  columns = ['coal_struct', 'cost','cost_per_member','walking']) # All the possible Structures per cluster
            # Get SO (min cost) per cluster, the structure and PMT per person
            so_cluster_df = df_pre.loc[df_pre['cost'] == df_pre['cost'].min()] # minimum cost coal per cluster
            so_lst.append(so_cluster_df)   #store results per cluster into global list //df of social optimums

    #end For Cluster

    # DFrame with Min cost struct per cluster
    df =  pd.concat(so_lst)  # concat lst of clusters into a single DF with each row is the SO of a cluster
    df.reset_index(drop=True, inplace=True)  # make sure indexes pair with the row number (too keep the cluster nbr)
    df.index.name = 'Idx'
    return df #returns a DataFrame where rows are the SO (structures, values) of per cluster


  #*****************************************************************************************************
  # ************************ INDIV RATIONALITY *********************************************************
  def indiv_rationality(self, soDF,soloLst):
      '''
      Calculates whether a cost alloc method is individually rational     
      :inputs: Needs the List of cost of traveling alone and a DFrame of the SO coal structure
      :outputs: a DataFrame comparing solo, weighted ans Shapley cost per member
      :NOTE: only the PCar is distributed, but the walking is added on their resp. functions for comparison w.r.t. solo
      It uses the idx of both to do the comparison
      '''

      print('calculating Indiv Rationality')

      # Brings everyone's individual travel cost into a DF - For the Indiv Rationality results
      df = pd.DataFrame(soloLst, columns=['solo'])

      # Prepare Coalition data for comparison - loop over clusters (=rows)
      coal_counter = 0                                                     # stores the number of coalitions formed for report
      preDict = dict()
      df.drop(columns='Idx', errors='ignore')
      soDF    = soDF.reset_index()  #adds a column Idx                     # Make sure indexes pair with number of rows
      for row in soDF.itertuples(index=True, name='Pandas'):               # For each cluster
           # count number of coalitions (for report)
           for coal in row.coal_struct:
               if len(coal)>1:               
                   coal_counter +=1
           # flatten persons idx
           idx = [item for sublist in row.coal_struct for item in sublist] # idx = [item for sublist in b[0] for item in sublist]
           preDict.update(zip(idx,row.cost_per_member))                    # Add row result to dict (to then build a DF)

      # Build data for omparison
      df2 = pd.DataFrame.from_dict(preDict, columns = ['inv_pmt'], orient='index')
      # Add Shapley columns 
      shapleyDF = self.shapley(soDF)  #returns df with tot pmt for each coalition member (Walking VoT cost + share of Pcar)
      df3 = pd.concat([df,df2,shapleyDF], axis =1)
      # Add comparison column - OBS: all the Shaps have the Walking added to the car - they are the Tot Cost for 'i'
      df3['inv_vs_solo']        = ((np.round(df3.inv_pmt,6) - np.round( df3.solo,6)) / np.round(df3.solo,6)) * 100     
      df3['shap_tot_vs_solo']   = ((np.round(df3.shap_tot,6) - np.round( df3.solo,6)) / np.round(df3.solo,6)) * 100
      df3['shap_car_vs_solo']   = ((np.round(df3.shap_car,6) - np.round( df3.solo,6)) / np.round(df3.solo,6)) * 100
      df3['shap_weight_vs_solo'] = ((np.round(df3.shap_weight,6) - np.round( df3.solo,6)) / np.round(df3.solo,6)) * 100
      
      # Calc 'Indiv Rationality' for all cases 
      ir_lst = [] 
      for s in [df3['inv_vs_solo'],df3['shap_tot_vs_solo'],df3['shap_car_vs_solo'],df3['shap_weight_vs_solo'] ]:   
          ir_lst.append(1-(len(s[s>0])/len(s)))    
      df4 = pd.DataFrame([ir_lst], columns = ['ir_inv_vs_solo','ir_shap_tot_vs_solo','ir_shap_car_vs_solo','ir_shap_weight_vs_solo'])  

      #Calc 'pct_saving' for all cases
      mean_savings_lst = [df3['inv_vs_solo'].mean(), df3['shap_tot_vs_solo'].mean(), df3['shap_car_vs_solo'].mean(), df3['shap_weight_vs_solo'].mean()]    
      df5 = pd.DataFrame([mean_savings_lst],columns = ['sav_inv_vs_solo','sav_shap_tot_vs_solo','sav_shap_car_vs_solo','sav_shap_weight_vs_solo'] )  

      # Output files for report
      print_to_file.print_to_file(ir_lst,'pct_rationality.csv')
      print_to_file.print_to_file(mean_savings_lst,'pct_saving.csv')   
      #print_to_file.print_to_file(coal_counter,'coal_counter.txt')  
      

      #print('social optimum:',soDF)
      return df4,df5,soDF, coal_counter  # (df with IR comparison, df with mean pct savings, df with SO)


#**************************************************************************************
#***************************   MAIN  **************************************************
#**************************************************************************************

def main():
    # Params
    p_ride        = 1         # taxi fare per km
    alpha         = 1.0085     # normal: 1.015 # manhattan = 1.0085  # controls cost of walking in VoT
    policy_str    = 'weighted' #'even'     # weighted', 'even' -  how we distribute the pmt based on walking dist
    distMetric    = 'median'   #'mean'     # 'median' ,'mean'  - how we calculate the centroids
    maxpoolers    = 4                      # how many people I want to share the car with
    flag_fall_pct = 0.05



 #********************** COORDINATES **************************************************
    
    # Read coordinates from file
    '''
    import read_coords as generate_coords
    coords_file = 'txt/coords_3basic_pts.txt'
    origin, dest = generate_coords.main(coords_file)
    epsilon = 500
    e2 = epsilon
    #OBS: with alpha >1 the optimal coalition becomes non-indiv rational (but Shapley is!)  
    '''
    
   #************************************************************************************  
   #********************** RND PTS *****************************************************     
    # Generate rnd clusters for Origin and Destination
    # OBS: This doesn't take the centers in order! it randomizes them! (e.g. the last center can appear in the middle)
    # STDev/Eps: as we increase the std dev (i.e decrease density), we can increase the epsilon

    '''
    #small dispersion
    n = 20
    sted_dev = 5
    epsilon = 25
    e2      = 35

    centers = [
        [60,200], [11,20],  [12,30],  [70, 70] #[70,80],[70, 90],[70,190]
        ] #centroids of clusters
    origin, y = make_blobs(n_samples=n, centers=centers, cluster_std=sted_dev, random_state=None)  #Create dataset as np array

    # Generate rnd clusters for dest
    centers = [
        [10,100], [0,200], [50,100],  [0, 140]#, # [100,150],[100, 155],[100,260]
        ] #centroids of clusters
    dest, y = make_blobs(n_samples=n, centers=centers, cluster_std=sted_dev, random_state=None)
    '''

    ######################### SHAPLEY 3 pts in paper ####################################################
    '''
    epsilon = 0.05 
    e2 = epsilon
    origin = np.array([
			     [0.109814, 0.109250], 
			     [0.108488, 0.110473],                                                
                  [ 0.109033, 0.113327]
				])                     

    dest = np.array([
                     [0.111928, 0.103092], 
                     [ 0.111773,0.105158],
                     [ 0.110595,0.107919] 
                  ])      
    '''                         
   
    
    ######################### MANHATTAN ####################################################

   
    # Long Trips: LGA Trips
    n = 28
    sted_dev = 0.005
    centers = [ [40.767937,-73.982155 ]
	         ] #centroids of clusters
    origin, y = make_blobs(n_samples=n, centers=centers, cluster_std=sted_dev, random_state=None)  #Create dataset as np array

	 # Generate rnd clusters for dest
    centers = [[40.782516469430846, -73.90468240035995]
	        ] #centroids of clusters
    dest, y = make_blobs(n_samples=n, centers=centers, cluster_std=sted_dev, random_state=None)

    epsilon = 0.0049
    e2 = 0.0073
  
    
    '''
    # Medium Trips: Manhattan-bound Trips (from chinatown to upper Manhattan)
    n = 15
    sted_dev = 0.0060 #0.0099
    centers = [ [40.722173344099865, -73.99752853816993]
	         ] #centroids of clusters
    origin, y = make_blobs(n_samples=n, centers=centers, cluster_std=sted_dev, random_state=None)  #Create dataset as np array

	 # Generate rnd clusters for dest
    centers = [[40.793704136973105, -73.95936978084814]
	        ] #centroids of clusters
    dest, y = make_blobs(n_samples=n, centers=centers, cluster_std=sted_dev, random_state=None)

    epsilon = 0.004
    e2 = 0.008
    '''
    
    #********************** Simple pts riding together for code tests *****************************
    '''
    n = 3
    sted_dev = 1
    epsilon = 25
    e2 = 500 
    #Origin
    centers = [
        [60,200] 
        ] #centroids of clusters
    origin, y = make_blobs(n_samples=n, centers=centers, cluster_std=sted_dev, random_state=0)  #Create dataset as np array
    #Dest 
    centers = [
        [10,100]
        ] #centroids of clusters
    dest, y = make_blobs(n_samples=n, centers=centers, cluster_std=sted_dev, random_state=1)  #Cambiar este rnd state sino
    '''
 



    #*****************************************************************************
    '''
    # Plot the points
    plt.scatter(origin[:,0],origin[:,1],color='k')
    plt.scatter(dest[:,0],dest[:,1],color='g')
    plt.show()
    '''
    
    #*****************************************************************************


    ###############################################################################
    ######################## Create Class Object and Run ##########################


    rideshare = RideShare(origin,
                          dest,
                          epsilon,
                          alpha  = alpha,
                          pRide  = p_ride,
                          policy = policy_str,
                          metric = distMetric,
                          maxPoolers = maxpoolers,
                          flag_fall  = flag_fall_pct
                          )



    # Individual rationality + Shapley
    soloLst       = rideshare.solo_travel_cost()
    soDF          = rideshare.coalitions_social_welfare()  #returns DF with SO[structure, value, cost_per_member] for all clusters
    #soDFrame.to_excel("soDataFrame.xlsx")
    df_agglom     = rideshare.agglomeration(soDF, e2)
      # df_agglom .to_excel("so_agglom.xlsx")
    #to compare with no agglom -> epsilon2  = 0
    df_ir,df_savings,soDF, coal_counter  = rideshare.indiv_rationality(df_agglom,soloLst) #(df with IR comparison, df with mean pct savings, df with SO)
 
    return df_ir, df_savings,soDF, coal_counter

    '''
    COLUMNS on the reports:
     # Indiv Rationality
       columns = ['ir_inv_vs_solo','ir_shap_tot_vs_solo','ir_shap_car_vs_solo','ir_shap_weight_vs_solo'] 
     
     # mean pct Savings
       columns = ['sav_inv_vs_solo','sav_shap_tot_vs_solo','sav_shap_car_vs_solo','sav_shap_weight_vs_solo']    
    
    '''

#*****************************************************************************
#**********************   RIDER **********************************************
#*****************************************************************************
#  Use this code with NYC's real dataset
#*****************************************************************************
if __name__ == "__main__":

    begin_time = datetime.datetime.now()

    ######## Iterate many times to obtain random samples ###############
    # Results saved in 'pct_rationality.csv' and 'pct_saving.csv'

    for i in list(range(1)):
        df_ir, df_savings,soDF, coal_counter = main()

    #Return output to terminal
    #sys.stdout.close()
    #sys.stdout=stdoutOrigin

    print('SO coalitions:',soDF['coal_struct']) #Returns a DF with SO coals per cluster (rows)
    print('time_elapsed:',datetime.datetime.now() - begin_time)

    #garbage management
    del soDF
    gc.collect()

    #**********************************************************************************
    #***********************************************************************

    #scratch code
    # Flatten coalitions structures
    # a = row.coal_struct       # comes as string of list of lists
    # b = [ast.literal_eval(a)] # from string to number


    #Previous Indiv Rat Code
    '''
    coals_lsts          = [ast.literal_eval(soDF['coal_struct'])] #[ast.literal_eval(keyLst)] #convert str to lst of lst
    cost_per_member_lst = soDF['cost_per_member']#self.coalition_value(coals_lsts)[1]  #take only the cost per member
    members             = self.flatten(ast.literal_eval(soDF['coal_struct']))#self.flatten(ast.literal_eval(keyLst))
    pmt                 = self.flatten(cost_per_member_lst)
    '''


