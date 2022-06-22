'''
Script that generates coordinates to use by the main script
'''

import numpy as np
import ast


def main(fname = '/Users/lucia/Desktop/Southampton/00-Codes/txt/coords_3basic_pts.txt'): 
    
    with open(fname) as file:
         res = [ast.literal_eval(line) for line in file]
         origin = res[0]
         dest   = res[1]
         
    #print(origin,dest, sep="\n")     
    return origin, dest#np.array(origin), np.array(dest) #regular script requires arrays for slicing  // not the NY tho     


    


if __name__ == "__main__":
    main()


