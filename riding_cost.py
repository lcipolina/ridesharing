  
from scipy.spatial import distance

def riding_cost(mid_origin_coords, mid_dest_coords, pRide = 2):
      '''
      Calculation of: distance travelled * Price per distance
      It is the single ride fare that then needs to be split among members
      '''

      #print('Entering riding_cost')
      riding_dist   = distance.euclidean(mid_origin_coords, mid_dest_coords)
      riding_cost   = riding_dist*pRide
      #print('trip_cost (riding only):',round(riding_cost,2))
      return riding_cost