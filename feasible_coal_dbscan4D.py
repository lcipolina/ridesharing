from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd

'''
Performs DB scan in 4 dimensions
Each point will belong to only 1 cluster
Depending on the epsilon, the epsilon is a measure of 'distance' in a 4D space....

returns a Dataframe from where we can filter out the data corres

'''

def DBscan(data, epsilon,minPts = 1):

    # Fit cluster
    db = DBSCAN(eps=epsilon, min_samples=minPts)
    db.fit(data)

    # Process results
    clusters = db.labels_  #list with nbr of cluster corresp to the data pont (see below)

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(clusters)) - (1 if -1 in clusters else 0)
    #n_noise_ = list(clusters).count(-1)

    print("Number of clusters: %d" % n_clusters_)

    return clusters #return list with cluster number for each datapoint

    


def get_feasible_coalitions(origin, dest, epsilon = 10):
    '''
    Converts the results of the cluster into something manageable
    '''

    # Convert points to a 4d space
    concatenated_data = np.concatenate([origin,dest], axis =1)  #create 1 vector: [x_o, y_o, x_d, y_d]

    # Fit DBScan
    cluster_lst = DBscan(concatenated_data, epsilon = epsilon) #it's like every point (i.e. every row) has 4 features and we are clustering across all these 4 features

    # Convert to DF to filter out easily
    df = pd.DataFrame({'cluster':cluster_lst})

    # Get elements on the same cluster
    uniques = df.cluster.unique()

    res = []
    for element in uniques:
        df2 = df.loc[df['cluster'] == uniques[element]]
        res.append(df2.index.tolist())

    return res
    


# EXAMPLE OF USAGE:
if __name__ == "__main__":  

    origin = np.array([
                        [0, 0],
                        [1, 1],
                        [5, 5] 
                        ])

    dest = np.array([
                    [0, 10],
                    [10, 10],
                    [9, 9] 
                    ]) 

    '''
    origin = np.array([
                        [0, 0],
                        [1, 1],
                        [5, 5], 
                        [6, 6],  
                        [4, 4] 
                        ])

    dest = np.array([
                    [0, 10],
                    [10, 10],
                    [9, 9], 
                    [11, 11], 
                    [9, 9]
                    ])                   
    '''

    eps = 25
    result = get_feasible_coalitions(origin, dest, epsilon = eps)                
    print('clusters:', result)


    '''
    NOTA sobre el return del DB scan:
    
    >> clusters = db.labels_
    #Este te da el numero de cluster con el que se junta el corresp input

    # Ejemplo:
    # data     = [A B C]
    # clusters = [0, 1, 1]
    # significa que A se va al cluster 0 (y es el unico elemento)
    # B y C van ambos al cluster 1

    # si tengo muchos puntos, me va a dar algo tipo:
    # [0,0,1,1,1,2,2,2,2,0,0,0,1,1...]

    # para sacar los que estan clustered together, combiene ponerlos en un PD dataframe y filtrar/agrupar por la columna "cluster" y listo

    
    
    
    '''