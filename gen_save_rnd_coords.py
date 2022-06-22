
import numpy as np
import csv
import json


# Random coordinates - this generates a list of lists 

#if rndSeed = True:
np.random.seed(0) #make them always the same rnd
nbr_pts = 2
max_coordinate = 10 # Dimensions of the plane I want coordinates within a certain square    
o = np.random.rand(nbr_pts, 2) * max_coordinate    
d  =  np.random.rand(nbr_pts, 2) * max_coordinate 

origin = o.tolist()
dest = d.tolist()

print(origin)

jsonOrigin = json.dumps(origin)
jsonDest = json.dumps(dest)

print(jsonOrigin)

with open('data.json', 'w') as f:
    json.dump(jsonOrigin, f)
    f.write("\n")
    json.dump(jsonDest, f)

'''
# save to csv
with open("out.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(origin)
    writer.writerows(dest)
  
  # csv_out = csv.writer(f)
  # csv_out.writerows([origin[index]] for index in range(0, len(origin)))
''' 


#with open('file4.csv','w') as f:
#    for row in salary:
#        for x in row:
#            f.write(str(x) + ',')
#        f.write('\n')

