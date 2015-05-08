import numpy as np
import networkx as nx

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
            if Dist[i,j]<threshold and i!=j:
                G.add_edge(i, j,weight=Dist[i,j])
                #G.add_edge(j, i,weight=Dist[i,j])
    return G


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

def best_betweeness(G,d_edges,n_objects,n_clusters):
    best_ind=list(np.argsort(d_edges.values()))[-1]
    best_edge=d_edges.keys()[best_ind]
    return best_edge