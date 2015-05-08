import numpy as np
import Assessor_model as ass
import EM_algorithm as em
import matplotlib.pyplot as plt
import graph_formulation as gra
import networkx as nx


#############################################  Random method   ####################################################

def rand_ass_gene(results,n_ass,n_objects):
    return np.random.randint(0,n_ass)

def rand_ob_gene(results,n_ass,n_objects,n_object_per_assessment):
    res=list()
    for i in range(n_object_per_assessment):
        res.append(np.random.randint(0,n_objects))
    return res

#############################################  Explore method   #################################################
 # Returns the least asked assessor
def equi_ass_gene(results,n_ass,n_objects): # Return least asked assessor
    return np.argsort([ len(results.values()[i]) for i in results.keys()])[0]

# Returns a group of objects formed by randomly chosen elements from the least viewed ones
def equi_ob_gene(results,n_ass,n_objects,n_object_per_assessment): # Return the list of pairs in increasing orders of views
    occurrences={}
    for i in results:
        for pair in results[i]:
            if pair[0] not in occurrences:
                occurrences[pair[0]]=1
            else:
                occurrences[pair[0]]+=1
            if pair[1] not in occurrences:
                occurrences[pair[1]]=1
            else:
                occurrences[pair[1]]+=1
    item_list=list()
    occurences_list=list()
    for item in occurrences:
        item_list.append(item)
        occurences_list.append(occurrences[item])
    sorted_ind=np.argsort(occurences_list)
    sorted_pairs=[item_list[i] for i in sorted_ind]
    
    if len(sorted_pairs)<n_object_per_assessment:
        res=list()
        for i in range(n_object_per_assessment):
            res.append(np.random.randint(0,n_objects))
    else:
        res=sorted_pairs
    
    
    return res[:n_object_per_assessment]

#############################################  Confidence method   ####################################################

# Choose assessor with the highest possible accuracy
def discriminatory_ass_gene(results,n_ass,n_objects):
    ass.symmetrized_Votes(results)
    Delta_est,Assessors_est,S_est=em.EM_est(results,n_objects,0.001)
    return np.argsort(Assessors_est)[0]

# Choose set of objects by pair among the least confident pairs
def discriminatory_ob_gene(results,n_ass,n_objects,n_object_per_assessment):
    ass.symmetrized_Votes(results)
    Delta_est,Assessors_est,S_est=em.EM_est(results,n_objects,0.001)
    s=np.argsort(S_est,axis=None)
    n=np.shape(S_est)[0]
    sorted_ind=[(s[i]/n,s[i]-(s[i]/n)*n) for i in range(len(s))]
    res=objects_from_edges(sorted_ind,n_object_per_assessment)
    return res

def objects_from_edges(l_edges,nob):
    res=list()
    i=0
    while len(res)<nob:
        ob1=l_edges[i][0]
        ob2=l_edges[i][1]
        if ob1 not in res:
            res.append(ob1)
        if ob2 not in res:
            res.append(ob2)
        i+=1
    return res


#############################################  Random method   ####################################################

def graph_ob_gene(results,n_ass,n_objects,n_object_per_assessment):
    # Compute Distance
    ass.symmetrized_Votes(results)
    Delta_est,Assessors_est,S_est=em.EM_est(results,n_objects,0.00001)
    Delta=np.ones(np.shape(Delta_est))-Delta_est
    Delta=Delta/np.max(Delta,axis=None)
    sorted_Delta=np.sort(Delta,axis=None)
    # Set threshold as median of non zeros values
    threshold=np.median([sorted_Delta[i] for i in np.nonzero(sorted_Delta)])
    # Create graph and compute betweeness of edges
    G=gra.init_graph(Delta,threshold)
    edges_betweeness=gra.betweeness(G)
    # Return the appropriate number of objects choosing the high betweeness first
    best_ind=list(np.argsort(edges_betweeness.values()))[-n_object_per_assessment:]
    best_edges=[edges_betweeness.keys()[i] for i in best_ind]
    res=objects_from_edges(best_edges,n_object_per_assessment)
    return res