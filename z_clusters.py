'''
Code to generate the feasible coalitions
Works similar to DBSCAN but it is not agglomerative in the sense that 
points are only clustered with their own radius, 
and not clustered beyond their own epsilon

Based on this code:
https://medium.com/analytics-vidhya/dbscan-from-scratch-almost-b02096600c14
'''

# Imports
from scipy.spatial import distance
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Helpers ****************************************************************************************
def neighborsGen(origins, point, eps_vct, metric):
    """
    Generates neighborhood graph for a given point
    """
    o_matrix   = distance.cdist(origins, origins, 'euclidean') # distances among member's origins
    clusters = []
    
    for i in range(origins.shape[0]):
        if metric(origins[point], origins[i]) < eps_vct:
            clusters.append(i)
    
    return clusters


def expand(data, coalition_vct, point, neighbors, currentPoint, eps_vct, minPts, metric):
    """
    Expands cluster from a given point until neighborhood boundaries are reached
    """
    coalition_vct[point] = currentPoint
    
    i = 0
    while i < len(neighbors):
        
        nextPoint = neighbors[i]
        
        if coalition_vct[nextPoint] == -1:
            coalition_vct[nextPoint] = currentPoint
        
        elif coalition_vct[nextPoint] == 0:
            coalition_vct[nextPoint] = currentPoint
            
            nextNeighbors = neighborsGen(data, nextPoint, eps_vct, metric)
            
            if len(nextNeighbors) >= minPts:
                neighbors = neighbors + nextNeighbors
        
        i += 1


# Driver *****************************************************************************************
def get_coalitions(data, coalition_vct, eps_vct, minPts=1, metric=distance.euclidean):
    """
    Driver; 
    iterates through neighborsGen for every point in data
    expands cluster for every point not determined to be noise
    """
    currentPoint = 0
    
    for i in range(0, data.shape[0]):  # Take one elem from the row, check if the point has already been clustered
        if coalition_vct[i] is not 0:
            continue
    
        neighbors = neighborsGen(data, i, eps_vct, metric) # Build circle

        if len(neighbors) < minPts:
            coalition_vct[i] = -1

        else:
            currentPoint += 1
            expand(data, coalition_vct, i, neighbors, currentPoint, eps_vct, minPts, metric)
    
    return coalition_vct

# Class ******************************************************************
class Basic_DBSCAN:
    """
    Parameters:
    
    eps_vct: Vector of radii of neighborhood graph. Each agent has an acceptable epsilon
    
    minPts: Number of neighbors required to label a given point as a core point.
    
    metric: Distance metric used to determine distance between points; 
            currently accepts scipy.spatial.distance metrics for two numeric vectors
    
    """
    
    def __init__(self, eps_vct, minPts, metric=distance.euclidean):
        self.eps_vct = eps_vct
        self.minPts = minPts
        self.metric = metric
    
    def fit_predict(self, data):
        """
        Parameters:
        
        data: An n-dimensional array of numeric vectors to be analyzed
        
        Returns:
        
        [n] cluster labels
        """
    
        coalition_vct = [0] * data.shape[0]
        
        get_coalitions(data, coalition_vct, self.eps_vct, self.minPts, self.metric)
        
        return coalition_vct

# EXAMPLE OF USAGE:
#if __name__ == "__main__":  
# scanner = Basic_DBSCAN(eps_vct=0.3, minPts=1)
# data= df[[cols[0], cols[1]]]
# data = StandardScaler().fit_transform(data)

#clusters = scanner.fit_predict(data)        