import numpy as np
import csv
def print_to_file(arr,fname = 'new_file.txt'):
    # writes a list sequentially into file
    with open(fname, "a") as f:
        writer = csv.writer(f)
        writer.writerow(np.round(arr,5))
    
   #writes an array 
   # with open(fname, 'a') as f: 
        # print(str(np.round(arr,5)), file=f)      


if __name__ == "__main__":

    data = [1,2,3]
    with open('test.csv', "w") as f:
        writer = csv.writer(f)
        writer.writerow(data)

        '''
        equivalent to:
        a = [1, 2, 3, 4]
        print(a)
        data = ', '.join(map(str, a))
        print(data)
        '''
