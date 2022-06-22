#https://leetcode.com/problems/best-position-for-a-service-centre/discuss/733153/Detailed-explanation-with-Weiszfeld%27s-algorithm-beats-100

# Time:  O(n * iter), iter is the number of iterations
# Space: O(1)

# see reference:
# - https://en.wikipedia.org/wiki/Geometric_median
# - https://wikimedia.org/api/rest_v1/media/math/render/svg/b3fb215363358f12687100710caff0e86cd9d26b
# https://stackoverflow.com/questions/57277247/the-point-that-minimizes-the-sum-of-euclidean-distances-to-a-set-of-n-points
# Weiszfeld's algorithm

# EXPLAIN:
# [1] If there is only 1 point, then return 0 obviously;
# [2] Assume the initial position of the service center to be the arithmetic mean of all points (to speed up the process);
# [3] Utilize Weiszfeld's algorithm to iteratively update the position
# [4] until the distance between two iterations is within tolerated error (converged);
# [5] Add an infinitesimal number to the denominator to avoid divide-by-zero error during iterations;

# Weiszfeld formula:
# https://handwiki.org/wiki/Geometric_median
# Weiszfeld's algorithm after the work of Endre Weiszfeld,[15] 
# is a form of iteratively re-weighted least squares. 
# This algorithm defines a set of weights that are inversely proportional to the distances 
# from the current estimate to the sample points, 
# and creates a new estimate that is the weighted average of the sample according to these weights.

import numpy as np
from scipy.spatial import distance

def getWeiszfeld(positions) :
        '''
        positions: List[List[int]]
        '''
        #if len(np.array([positions])) == 1: return [0,0]  # [1]  #if there is only one element, median = mean
        
        # Get arithmetic mean - initial guess-
        curr = list(map(lambda a: sum(a)/len(a), zip(*positions)))  # [2]
        prev = [float('inf')] * 2
        
        # Calculate Euclidean distance
        def get_norm(p1, p2):
            return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** 0.5
        
        # Move the mean iteratively until error is min
        err = 1e-7  # [4]
        epsilon = 1e-20  # [5]
        while get_norm(curr, prev) > err: # [3]
            numerator, denominator = [0, 0], 0
            for p in positions:
                #calculate Weiszfeld formula (gradient of the objective function)
                l2 = get_norm(curr, p) + epsilon
                numerator[0] += p[0] / l2
                numerator[1] += p[1] / l2
                denominator += 1 / l2
            next_p = [numerator[0]/denominator, numerator[1]/denominator]
            curr, prev = next_p, curr
            
        return curr #sum([get_norm(p, curr) for p in positions]) # Lucia I am returning the coordinates (not the distance)

# EXAMPLE OF USAGE:
#if __name__ == "__main__":  
    # https://leetcode.com/problems/best-position-for-a-service-centre/
    # positions = [[0,1],[1,0],[1,2],[2,1]]

    # import numpy as np
    #positions = np.array([
    #                  [0, 0],
    #                  [1, 1],
    #                  [5, 5] 
    #                   ])
    #print(getWeiszfeld(positions) ) 
     # result = 4 OK    - distance for original return 
     # result = [1,1] OK - coordinates of midpoint 