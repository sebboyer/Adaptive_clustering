# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import sys
sys.path.insert(0, '/home/sebastien/Documents/Adaptive clustering/Simulation python/Consensus')
sys.path.insert(0, '/home/sebastien/Documents/Adaptive clustering/Simulation python/Pb_sim')


import Assessor_model as ass
import real_clusters as rc
import votes_EM_model as vem
import visualizations as vis
import matplotlib.pyplot as plt
import spectral_graph_clustering as spg
import EM_algorithm as em
import performance_testing as pt


# <codecell>

############### Creating the objects within different clusters

# # Setting parameters of the objects
# n_objects=28
# n_clusters=2
# dim=2

# # Creating n_object objects in n_clusters clusters with dim features
# set_of_objects=list()
# for i in range(n_clusters):
#     C=ass.Cluster(n_objects/n_clusters,3*i,dim)
#     C.populate_random_multivariate()

    
#     set_of_objects+=[C.objects[i] for i in range(n_objects/n_clusters)]
    
# real_clusters=rc.kmeans_clusters(set_of_objects,n_clusters)
# vis.visualization_2D(set_of_objects,real_clusters,0,1)

# # <codecell>

# # Setting parameters of the Assessments
# n_assessors=6
# n_assessments_per_assessor=3
# n_object_per_assessment=6

# # Creating Assessors
# set_of_assessors=[ass.Assessor((i%10)/float(2),5) for i in range(n_assessors)]

# Creating adaptive method
def ass_gene(results,n_ass,n_objects):
    return np.random.randint(0,n_ass)
def set_of_object_gene(results,n_ass,n_objects,n_object_per_assessment):
    res=list()
    for i in range(n_object_per_assessment):
        res.append(np.random.randint(0,n_objects))
    return res
adaptive_method=ass.Adaptive_method(ass_gene,set_of_object_gene)

# # Creating experiment
# exp=ass.Experiment(set_of_assessors,set_of_objects,adaptive_method)

# # Procede to first adaptive assessment
# for i in range(50):
#     exp.procede_adaptive_assessment(n_clusters,n_object_per_assessment)
# Votes=exp.get_results()
# ass.symmetrized_Votes(Votes)

# <codecell>

def naive_distance(Votes,N):
    occ=np.zeros((N,N))
    D=np.zeros((N,N))
    for assessor in Votes:
        for pair in Votes[assessor]:
            occ[pair[0],pair[1]]+=1
            if Votes[assessor][pair]==True:
                D[pair[0],pair[1]]+=1
    for i in range(n_objects):
        for j in range(n_objects):
            if occ[i,j]!=0:
                D[i,j]=D[i,j]/float(occ[i,j])
    return D

# <codecell>

# Delta_naive=naive_distance(Votes,n_objects)

# # <codecell>

# plt.pcolor(Delta_naive)
# plt.show()

# # <codecell>

# # Estimating Adjacency from distance using equal size kmeans
# est_adj=spg.adjacency_2Clusters_eq(Delta_naive,n_objects/2,n_objects/2,2)
# # Computing clusters indexes from adjacency
# est_clusters_naive=vis.clusters_from_adjacency(est_adj,n_clusters)
# # Visualizing the cluster separation
# vis.visualization_2D(set_of_objects,est_clusters,0,1)

# # <codecell>

# pt.n_mutual_info(est_clusters_naive,real_clusters,2)

# # <codecell>

# # Estimating the distance matrix from a Maximum likelihood EM algorithm

# Delta_est,Assessors_est,S_est=em.EM_est(Votes,n_objects,0.0001)
# plt.pcolor(np.ones((n_objects,n_objects))-Delta_est)
# plt.show()

# # <codecell>

# # Estimating Adjacency from distance using equal size kmeans
# est_adj=spg.adjacency_2Clusters_eq(Delta_est,n_objects/2,n_objects/2,2)
# # Computing clusters indexes from adjacency
# est_clusters=vis.clusters_from_adjacency(est_adj,n_clusters)
# # Visualizing the cluster separation
# vis.visualization_2D(set_of_objects,est_clusters,0,1)

# # <codecell>

# pt.n_mutual_info(est_clusters,real_clusters,2)

# <codecell>


