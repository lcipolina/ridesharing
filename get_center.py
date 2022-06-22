import Weiszfeld  #to calculate geometric median
import numpy as np

def get_center(origin, destination,metric = 'median'):
    if metric == 'median':
        mid_origin_coords = Weiszfeld.getWeiszfeld(origin)       #coordinates of origin's geom median
        mid_dest_coords   = Weiszfeld.getWeiszfeld(destination) #coordinates of destiny's geom median
    else:   # use mean  
         mid_origin_coords = np.mean(origin, axis=0)        #coordinates of origin's baricenter/centroic
         mid_dest_coords   = np.mean(destination, axis=0)  #coordinates of destiny's baricenter/centroid

    return [mid_origin_coords,mid_dest_coords]