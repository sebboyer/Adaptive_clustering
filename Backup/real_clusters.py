import numpy as np
from sklearn import cluster


def kmeans_clusters(set_of_objects,k):
    N=len(set_of_objects)
    f=set_of_objects[0].get_n_feat()
    X=np.zeros((N,f))
    for i in range(len(set_of_objects)):
        X[i]=set_of_objects[i].features
        
    km=cluster.KMeans(n_clusters=k)
    km.fit(X)
    return km.predict(X)