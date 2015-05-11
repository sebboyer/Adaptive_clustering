# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
#%matplotlib inline
from objects_cluster_generation import *
from copy import deepcopy

# <codecell>

def clusters_from_adjacency(A):
    n=np.shape(A)[0]
    B=deepcopy(A)
    clusters=np.zeros((1,n))
    for i in range(n):
        if B[0,i]!=-1:
            for j in range(n):
                if B[i,j]==1:
                    B[0,j]=-1
                    clusters[0,j]=i
    a=list(set(clusters[0]))
    clusters=[a.index(clusters[0,i]) for i in range(n)]
    return clusters

# <codecell>

def visualization_2D(set_of_objects,clusters_indexes,dim1,dim2):
    n=len(set_of_objects)
    x=[o.features[dim1] for o in set_of_objects]
    y=[o.features[dim2] for o in set_of_objects]
    colours_available=['blue','red','green','c','m']
    colours=[colours_available[clusters_indexes[i]] for i in range(n)]
    fig, ax = plt.subplots(figsize=(8,4))
    l = ax.scatter(x, y, c=colours)
    return fig

# <codecell>

def real_adjacency(n_clusters,n_objects):
    A=np.zeros((n_objects,n_objects))
    n_objects_per_cluster=n_objects/n_clusters
    for i in range(n_clusters):
        A[i*n_objects_per_cluster:(i+1)*n_objects_per_cluster,i*n_objects_per_cluster:(i+1)*n_objects_per_cluster]=np.ones((n_objects_per_cluster,n_objects_per_cluster))
    return A

# <codecell>

# ############### Creating the objects within different clusters

# # Setting parameters of the objects
# n_objects=12
# n_clusters=2
# dim=2

# # Creating n_object objects in n_clusters clusters with dim features
# set_of_objects=list()
# for i in range(n_clusters):
#     C=Cluster(n_objects/n_clusters,3*i,dim)
#     C.populate_random_multivariate()
#     set_of_objects+=[C.objects[i] for i in range(n_objects/n_clusters)]

# A=real_adjacency(n_clusters,n_objects)
# clust=clusters_from_adjacency(A,n_clusters)
# visualization_2D(set_of_objects,clust,0,1)

