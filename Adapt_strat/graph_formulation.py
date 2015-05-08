# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as col
import sys
sys.path.insert(0, '/home/sebastien/Documents/Adaptive clustering/Simulation python/Consensus')
sys.path.insert(0, '/home/sebastien/Documents/Adaptive clustering/Simulation python/Pb_sim')
#%matplotlib inline

# <codecell>
import Assessor_model as ass
import visualizations as vis
import EM_algorithm as em

# <codecell>

#  Delta=[[ 0. ,0.00170612,  0.00169131,  0.02469532,  0.0017214,   0.01065122],
#  [ 0.00170612 , 0.        ,  0.00156305 , 0.02263081 , 0.00165207 , 0.02359098],
#  [ 0.00169131 , 0.00156305  ,0.         , 0.02215095  ,0.00170867 , 0.02426879],
#  [ 0.02469532 , 0.02263081 , 0.02215095 , 0.          ,0.00105833 , 0.00064471],
#  [ 0.0017214  , 0.00165207 , 0.00170867 , 0.00105833 , 0.          ,0.00068122],
#  [ 0.01065122 , 0.02359098 , 0.02426879 , 0.00064471  ,0.00068122 , 0.        ]]

# # <codecell>

# Delta=np.array(Delta)
# n_objects=6
# n_clusters=2

# # <codecell>

# Delta_norm=Delta/np.max(Delta,axis=None)

# <codecell>

def init_graph(Dist,threshold):
    
    ## Creating graph
    G=nx.Graph()    
    # Creating nodes
    n=np.shape(Dist)[0]
    for i in range(n):
        G.add_node(i)
    # Creating edges
    for i in range(n):
        for j in range(n):
            if Dist[i,j]>threshold and i!=j:
                G.add_edge(i, j,weight=Dist[i,j])
                #G.add_edge(j, i,weight=Dist[i,j])
    return G
    

# <codecell>

def visua_graph(G,n_objects,n_clusters):
    n_object_per_cluster=n_objects/n_clusters
    
    # Listing good and bad edges
    list_good_edges=list()
    list_bad_edges=list()
    for i in range(n_clusters):
        for j in range(n_object_per_cluster):
            node=i*n_object_per_cluster+j
            for key in G[node]:
                if key!=node and (node,key) not in list_good_edges and (node,key) not in list_bad_edges:
                    if key in range(i*n_object_per_cluster,(i+1)*n_object_per_cluster):
                        list_good_edges.append((key,node))
                    else:
                        list_bad_edges.append((key,node))
    list_good_edges=list(set(list_good_edges))
    list_bad_edges=list(set(list_bad_edges))
    
    # Display nodes, good edges, bad edges
    pos=nx.circular_layout(G)
    colors='rbg'
    
    # Nodes
    for i in range(n_clusters): 
        first=i*n_object_per_cluster
        last=(i+1)*n_object_per_cluster
        nx.draw_networkx_nodes(G,pos,nodelist=range(first,last), node_color=colors[i%3],node_size=500,alpha=0.8)
    # Edges
    #nx.draw_networkx_edges(G,pos,width=1.0,alpha=0.5)
    nx.draw_networkx_edges(G,pos,edgelist=list_bad_edges,width=8,alpha=0.5,edge_color='r')
    nx.draw_networkx_edges(G,pos, edgelist=list_good_edges,width=8,alpha=0.5,edge_color='g')

# <codecell>

def betweeness(G):
    sp=nx.shortest_path(G)
    n_nodes=len(G.nodes())
    d_edges={}
    for i in range(len(G.edges())):
        d_edges[G.edges()[i]]=0
    sp=nx.shortest_path(G)
    for i in range(n_nodes):
        for j in range(i):
            if j in nx.node_connected_component(G, i):
                l=sp[i][j]
                for k in range(len(l)-1):
                    if (l[k],l[k+1]) in d_edges:
                        d_edges[(l[k],l[k+1])]+=1
                    else:
                        d_edges[(l[k+1],l[k])]+=1
    return d_edges

# <codecell>

def draw_best_betweeness_blue(G,d_edges,n_objects,n_clusters):
    pos=nx.circular_layout(G)
    best_ind=list(np.argsort(d_edges.values()))[-1]
    best_edge=d_edges.keys()[best_ind]
    visua_graph(G,n_objects,n_clusters)
    nx.draw_networkx_edges(G,pos,edgelist=[best_edge],width=8,alpha=0.5,edge_color='b')

# <codecell>

# threshold=np.median(Delta_norm)
# G=init_graph(Delta_norm,threshold)
# visua_graph(G,n_objects,n_clusters)

# # <codecell>

# d_betweeness=betweeness(G)
# draw_best_betweeness_blue(G,d_betweeness,n_objects,n_clusters)

# # <codecell>

# print d_betweeness

# <codecell>

############### Creating the objects within different clusters

# Setting parameters of the objects
# n_objects=12
# n_clusters=2
# dim=2

# # Creating n_object objects in n_clusters clusters with dim features
# set_of_objects=list()
# for i in range(n_clusters):
#     C=ass.Cluster(n_objects/n_clusters,3*i,dim)
#     C.populate_random_multivariate()
#     set_of_objects+=[C.objects[i] for i in range(n_objects/n_clusters)]
    
# real_adj=vis.real_adjacency(n_clusters,n_objects)
# real_clusters=vis.clusters_from_adjacency(real_adj,n_clusters)
# vis.visualization_2D(set_of_objects,real_clusters,0,1)

# # <codecell>

# ############### Creating assessors and generating votes


# # Setting parameters of the Assessments
# n_assessors=5
# n_assessments_per_assessor=5
# n_object_per_assessment=7

# # Creating Assessors
# set_of_assessors=[ass.Assessor((i%10)/float(0.4),5) for i in range(n_assessors)]

# # <codecell>

# # Creating adaptive method
# def ass_gene(results,n_ass,n_objects):
#     return np.random.randint(0,n_ass)
# def set_of_object_gene(results,n_ass,n_objects,n_object_per_assessment):
#     res=list()
#     for i in range(n_object_per_assessment):
#         res.append(np.random.randint(0,n_objects))
#     return res
# adaptive_method=ass.Adaptive_method(ass_gene,set_of_object_gene)

# # <codecell>

# ############### Creating assessors and generating votes


# # Setting parameters of the Assessments
# n_assessors=5
# n_assessments_per_assessor=3
# n_object_per_assessment=5

# # Creating Assessors
# set_of_assessors=[ass.Assessor((i%10)/float(2),5) for i in range(n_assessors)]

# # Creating Assessments
# assessment_list=ass.generate_random_assessment(n_object_per_assessment,n_objects,n_assessors,n_assessments_per_assessor)   

# # Creating experiment
# exp=ass.Experiment(set_of_assessors,set_of_objects,adaptive_method)

# # Procede to the assessments
# for i in range(200):
#     exp.procede_adaptive_assessment(n_clusters,n_object_per_assessment)
# Votes=exp.get_results()
# ass.symmetrized_Votes(Votes)

# # <codecell>

# Delta_est,Assessors_est,S_est=em.EM_est(Votes,n_objects,0.001)
# plt.pcolor(Delta_est)
# plt.show()
# Delta_norm=Delta_est/np.max(Delta_est,axis=None)

# # <codecell>

# threshold=0.02
# G=init_graph(Delta_norm,threshold)
# visua_graph(G,n_objects,n_clusters)

# # <codecell>

# d_betweeness=betweeness(G)
# draw_best_betweeness_blue(G,d_betweeness,n_objects,n_clusters)

# <codecell>


