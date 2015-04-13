# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
from sympy import Matrix
from scipy.sparse.linalg import *
from sklearn import *

# <codecell>

# Compute a similarity matrix from a squared distance matrix

def similarity(Dist):
    n=np.shape(Dist)[0]
    M=np.max(Dist)
    A=Dist/M
    A=np.ones((n,n))-A
    for i in range(n):
        A[i,i]=0
    return A

# <codecell>

# Compute the Laplacian of a Distance matrix

def laplacian(A):
    n=np.shape(A)[0]
    D=np.zeros((n,n))
    A=np.around(A,decimals=3)
    S=np.sum(A,axis=1)
    for i in range(n):
        D[i,i]=S[i]
    L=D-A
    return L 

# <codecell>

# Group lines of v into k clusters according to the (k-1)th first columns values

def Kmeans_cluster(k,v):
    km=cluster.KMeans(n_clusters=k)
    km.fit(v[:,:k-1])
    return km.labels_



# Group in 2 clusters of defined size

def Cluster_equally(size0,size1,v):
    n=np.shape(v)[0]
    km=cluster.KMeans(n_clusters=2)
    v_col=np.array([v[:,1]]).T
    km.fit(v_col)
    centers=km.cluster_centers_ 
    dist_0=abs(v_col-centers[0])
    dist_1=abs(v_col-centers[1])
    best_1=dist_1-dist_0
    best_1_ind=np.argsort(best_1,axis=0)
    
    clusters=np.zeros((1,n))
    count_1=0
    while count_1<size1:
        clusters[0,best_1_ind[count_1]]=1
        count_1+=1
    return clusters[0]

# <codecell>

# Main function of the script
# Takes
#- Distance matrix : square with 0 in the diagonal
#- k the number of clusters to estimate
# Returns
#- the Adjacency matrix corresponding to the clusterization estimated
def adjacency_KClusters(Dist,k):
    S=similarity(Dist)
    n=np.shape(S)[0]
    L=laplacian(S)
    w,v=np.linalg.eig(L)
    ind_sorted_eigvals=np.argsort(w)
    v=v[:,ind_sorted_eigvals]
    w=w[ind_sorted_eigvals]
    v=v[:,1:k]
    print v
    clusters=Kmeans_cluster(k,v)
    A=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if clusters[i]==clusters[j]:
                A[i,j]=1
    return A

def adjacency_2Clusters_eq(Dist,size0,size1,k):
    S=similarity(Dist)
    n=np.shape(S)[0]
    L=laplacian(S)
    w,v=np.linalg.eig(L)
    ind_sorted_eigvals=np.argsort(w)
    v=v[:,ind_sorted_eigvals]
    w=w[ind_sorted_eigvals]
    clusters=Cluster_equally(size0,size1,v)
    A=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if clusters[i]==clusters[j]:
                A[i,j]=1
    return A
    

# <codecell>

#  Delta=[[ 0. ,0.00170612,  0.00169131,  0.02469532,  0.0017214,   0.01065122],
#  [ 0.00170612 , 0.        ,  0.00156305 , 0.02263081 , 0.00165207 , 0.02359098],
#  [ 0.00169131 , 0.00156305  ,0.         , 0.02215095  ,0.00170867 , 0.02426879],
#  [ 0.02469532 , 0.02263081 , 0.02215095 , 0.          ,0.00105833 , 0.00064471],
#  [ 0.0017214  , 0.00165207 , 0.00170867 , 0.00105833 , 0.          ,0.00068122],
#  [ 0.01065122 , 0.02359098 , 0.02426879 , 0.00064471  ,0.00068122 , 0.        ]]

#A=adjacency_KClusters(Delta,2)
# print A

