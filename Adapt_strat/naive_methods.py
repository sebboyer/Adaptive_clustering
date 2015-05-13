import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/sebastien/Documents/Adaptive clustering/Simulation python/Consensus')
sys.path.insert(0, '/home/sebastien/Documents/Adaptive clustering/Simulation python/Pb_sim')


import Assessor_model as ass
import EM_algorithm as em
import graph_formulation as gra
import networkx as nx
from scipy import stats
import spectral_graph_clustering as spc

'''
The framework for adaptive methods is as follows:
- one function should generate the assessor index : from 0 to $n_{assessor}$
- another function should generate a set of $n_{object Per Assessment}$ objects index from 0 to $n_{objects}$ 
- both function should use only the existing results contained in $results$ (a dictionnary : $results[ass_{ind}][ob_{ind1},ob_{ind2}]=True$ or $False$
'''


###################################################################################################
##############################      Random method
###################################################################################################

def rand_ass_gene(results,n_ass,n_objects):
    return np.random.randint(0,n_ass)

def rand_ob_gene(results,n_ass,n_objects,n_object_per_assessment,Delta_est,S_est,n_cluster,Known_K):
    res=list()
    for i in range(n_object_per_assessment):
        res.append(np.random.randint(0,n_objects))
    return res


###################################################################################################
###############################       Method based on the number of occurences (meta data of the votes : assessor + set of objects)
###################################################################################################


# Returns the least asked assessor
def equi_ass_gene(results,n_ass,n_objects): # Return least asked assessor
    return np.argsort([ len(results.values()[i]) for i in results.keys()])[0]

# Returns a group of objects formed by randomly chosen elements from the least viewed ones
def equi_ob_gene(results,n_ass,n_objects,n_object_per_assessment,Delta_est,S_est,n_cluster,Known_K): # Return the list of pairs in increasing orders of views
    occurences=np.zeros((1,n_objects))
    for i in range(n_objects):
        occurences[0,i]=0
    for ass in results:
        for pair in results[ass]:
            occurences[0,pair[0]]+=1
    occurences=occurences[0]
    ind=list(np.argsort(occurences))
    return ind[:n_object_per_assessment]

###################################################################################################
###############################       Method based on presenting "easy" sets
###################################################################################################


def diversity_ob_gene(results,n_ass,n_objects,n_object_per_assessment,Delta_est,S_est,n_cluster,Known_K):


    pairs=cooccurences_pairs(results)
    adja=adja_list(pairs,n_objects)
            
    choice=list()
    for i in range(n_object_per_assessment/4):
        remaining=range(n_objects)
        o=choose_rand_in(remaining,n_objects)
        p=choose_rand_in(adja[o],n_objects)
        choice.append(o)
        choice.append(p)
        remaining=update(remaining,adja,[o,p])
        q=choose_rand_in(remaining,n_objects)
        r=choose_rand_in(list(set(remaining)&set(adja[q])),n_objects)
        choice.append(q)
        choice.append(r)
        
    while len(choice)<n_object_per_assessment:
        r=np.random.randint(n_objects-1)
        if r not in choice:
            choice.append(r)
            
    return choice

def choose_rand_in(remaining,n_objects):
    if len(remaining)==0:
        r=np.random.randint(n_objects-1)
    else:
        r=np.random.choice(remaining)
    return r

def update(remaining,adja,seen):
    for ob in seen:
        if len(adja[ob])!=0:
            remaining=list(set(remaining)-set(adja[ob]))
        remaining=list(set(remaining)-set([ob]))
    return remaining

def cooccurences_pairs(results):
    pairs={}
    for assess in results:
        for pair in results[assess]:
            if pair not in pairs:
                pairs[pair]=[0,0]
            pairs[pair][0]+=1
            if results[assess][pair]:
               pairs[pair][1]+=1
    return pairs

def adja_list(pairs,n_objects):
    adja={}
    t=0.5
    for i in range(n_objects):
        adja[i]=list()
    for pair in pairs:
        if pairs[pair][1]/float(pairs[pair][0])>t and pair[0]!=pair[1]:
            adja[pair[0]].append(pair[1])
    return adja


###################################################################################################
###############################     Method based on the centroids of the current clusters
###################################################################################################

def centroid_ob_gene(results,n_ass,n_objects,n_object_per_assessment,Delta_est,S_est,n_cluster,Known_K):
    
    # If Delta_est=0 (ie first querys) choose at random
    if np.shape(Delta_est)==():
        #print 'Pick at random'
        return rand_ob_gene(results,n_ass,n_objects,n_object_per_assessment,Delta_est,S_est,n_cluster,Known_K)

    S=spc.similarity(Delta_est)
    n=np.shape(S)[0]
    L=spc.laplacian(S)
    w,v=np.linalg.eig(L)
    ind_sorted_eigvals=np.argsort(w)
    v=v[:,ind_sorted_eigvals]
    w=w[ind_sorted_eigvals]
    v=v[:,1:n_cluster]
    
    if Known_K:
        km=spc.cluster.KMeans(n_clusters=n_cluster)
        km.fit(v[:,:n_cluster-1])
        centers=km.cluster_centers_
    else:
        [clusters,centers]=spc.GMM_cluster(v,n_cluster)
    
    # Enter the closest points to the centroids
    choice=[]
    for center in centers:
        dmin=99999
        bestind=-1
        for i in range(np.shape(v)[0]):
            d=np.sum(np.square(center-v[i,:]))
            if d<dmin:
                bestind=i
                dmin=d
        choice.append(bestind)

    #print 'Choosed first : ',choice
        
    # Complete choice until full
    while len(choice)<n_object_per_assessment:
        r=np.random.randint(n_objects)
        if r not in choice:
            choice.append(r)
    
    return choice  
    
###################################################################################################
###############################     Method based on the distance and the confidence values of each pair
###################################################################################################


# Choose assessor with the highest possible accuracy
def discriminatory_ass_gene(results,n_ass,n_objects):
    ass.symmetrized_Votes(results)
    Delta_est,Assessors_est,S_est=em.EM_est(results,n_objects,0.001)
    return np.argsort(Assessors_est)[0]

# Choose set of objects by pair among the least confident pairs
def discriminatory_ob_gene(results,n_ass,n_objects,n_object_per_assessment,Delta_est,S_est,n_cluster,Known_K):

    if np.shape(S_est)==():
        return rand_ob_gene(results,n_ass,n_objects,n_object_per_assessment,Delta_est,S_est,n_cluster,Known_K)
    
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
        if ob2 not in res and len(res)<nob:
            res.append(ob2)
        i+=1
    return res

###################################################################################################
###############################      Graph 
###################################################################################################


def graph_ob_gene(results,n_ass,n_objects,n_object_per_assessment,Delta_est,S_est,n_cluster,Known_K):

    # If Delta_est=0 (ie first querys) choose at random
    if np.shape(Delta_est)==():
        return rand_ob_gene(results,n_ass,n_objects,n_object_per_assessment,Delta_est,S_est,n_cluster,Known_K)

    G=graph_from_dist(Delta_est)
    edges_betweeness=gra.betweeness(G)
    
    # Return the appropriate number of objects choosing in decreasing order of betweeness
    res=dec_bet_objects(edges_betweeness,n_object_per_assessment,n_objects)
    
    # Visua graph
    #gra.draw_best_betweeness_blue(G,edges_betweeness,n_objects,2)
    
    return res

def graph2_ob_gene(results,n_ass,n_objects,n_object_per_assessment,Delta_est,S_est,n_cluster,Known_K):

    # If Delta_est=0 (ie first querys) choose at random
    if np.shape(Delta_est)==():
        return rand_ob_gene(results,n_ass,n_objects,n_object_per_assessment,Delta_est,S_est)

    G=graph_from_dist(Delta_est)
    edges_betweeness=gra.betweeness(G)
    
    # Return the appropriate number of objects choosing in decreasing order of betweeness
    res=best_bet_objects(results,edges_betweeness,n_object_per_assessment,n_objects)
    
    # Visua graph
    #gra.draw_best_betweeness_blue(G,edges_betweeness,n_objects,2)
    
    return res

def dec_bet_objects(edges_betweeness,nob,n_objects):
    best_ind=list(np.argsort(edges_betweeness.values()))[-nob:]
    best_edges=[edges_betweeness.keys()[i] for i in best_ind]
    res=list()
    i=0
    while len(res)<nob:  
        if len(best_edges)>i:      
            ob1=best_edges[i][0]
            ob2=best_edges[i][1]
        else:
            ob1=np.random.randint(0,n_objects)
            ob2=np.random.randint(0,n_objects)
        if ob1 not in res:
            res.append(ob1)
        if ob2 not in res and len(res)<nob:
            res.append(ob2)
        i+=1
    return res

def graph_from_dist(Delta_est):
    Delta=np.ones(np.shape(Delta_est))-Delta_est
    Delta=Delta/np.max(Delta,axis=None)
    sorted_Delta=np.sort(Delta,axis=None)
    n=np.shape(Delta_est)[0]
    threshold=[sorted_Delta[i] for i in np.nonzero(sorted_Delta)]
    n_edges=min(n**2/4,len(threshold[0] ))
    threshold=threshold[0][-n_edges]
    
    # Create graph and compute betweeness of edges
    G=gra.init_graph(Delta,threshold)

    return G



def best_bet_objects(results,edges_betweeness,nob,n_objects):
    best_ind=list(np.argsort(edges_betweeness.values()))[-nob:]
    best_edges=[edges_betweeness.keys()[i] for i in best_ind]

    pairs=cooccurences_pairs(results)
    adja=adja_list(pairs,n_objects)

    if len(best_edges)==0:
        res=list()
        for i in range(nob):
            res.append(np.random.randint(0,n_objects))
        return res
        
    a=best_edges[-1][0]
    b=best_edges[-1][1]
    choice=[a,b]
    remaining=range(n_objects)
    remaining=list(set(remaining)-set([a,b]))
    while len(choice)<nob:
        c=choose_rand_in(list(set(remaining)&set(adja[a])),n_objects)
        remaining=list(set(remaining)-set([c]))
        choice.append(c)
        d=choose_rand_in(list(set(remaining)&set(adja[b])),n_objects)
        remaining=list(set(remaining)-set([d]))
        choice.append(d)
    return choice


###################################################################################################
###############################      Explore Strategy (cf Sorting paper Kalyan Una May)
###################################################################################################


def explore_ob_gene(results,n_ass,n_objects,n_object_per_assessment,Delta_est,S_est,n_cluster,Known_K):

    # If Delta_est=0 (ie first querys) choose at random
    if np.shape(Delta_est)==():
        return rand_ob_gene(results,n_ass,n_objects,n_object_per_assessment,Delta_est,S_est,n_cluster,Known_K)

    pairs=cooccurences_pairs(results)
    w=np.max([x[0] for x in pairs.values()])
    
    xk=[] ## Ordering pairs
    for i in range(n_objects):
        for j in range(i):
            xk.append(i*n_objects+j)
                 
    pk=[] ## Computing pairs probabilities p(G)
    for i in range(n_objects):
        for j in range(i):
            if (i,j) in pairs:
                pk.append(np.exp(-pairs[(i,j)][0]/float(w)))
            else:
                pk.append(1)
    s=np.sum(pk)
    for k in range(len(pk)):
        pk[k]=pk[k]/s
    
    rv=stats.rv_discrete(values=(xk,pk)) ## Creating discrete random variable
    pair_numbers=rv.rvs(size=3*n_object_per_assessment)
    
    choice=[] ## Choicing randomly out of p(G) distribution
    k=0
    while len(choice)<n_object_per_assessment:
        p=pair_numbers[k]
        i=p/n_objects
        j=p % n_objects
        if i not in choice:
            choice.append(i)
        if j not in choice and len(choice)<n_object_per_assessment:
            choice.append(j)
        k+=1
    
    return choice

###################################################################################################
###############################      Exploit-pairwise Strategy (cf Sorting paper Kalyan Una May)
###################################################################################################


def exploitpw_ob_gene(results,n_ass,n_objects,n_object_per_assessment,Delta_est,S_est,n_cluster,Known_K):
    # If Delta_est=0 (ie first querys) choose at random
    if np.shape(Delta_est)==():
        return rand_ob_gene(results,n_ass,n_objects,n_object_per_assessment,Delta_est,S_est,n_cluster,Known_K)

    pairs=cooccurences_pairs(results)
    w=np.max([x[0] for x in pairs.values()])
    
    
    xk=[]
    for i in range(n_objects):
        for j in range(i):
            xk.append(i*n_objects+j)
            
    mu=0.5
    sigma=0.01

    pk=[]
    for i in range(n_objects):
        for j in range(i):
            if (i,j) in pairs:
                pk.append(np.exp(-((pairs[(i,j)][1]/float(pairs[(i,j)][0]))-mu)**2/(2*sigma)))
            else:
                pk.append(1)
    s=np.sum(pk)
    for k in range(len(pk)):
        pk[k]=pk[k]/float(s)    
    
    rv=stats.rv_discrete(values=(xk,pk))
    pair_numbers=rv.rvs(size=3*n_object_per_assessment)
    
    choice=[]
    k=0
    while len(choice)<n_object_per_assessment:
        p=pair_numbers[k]
        i=p/n_objects
        j=p % n_objects
        if i not in choice:
            choice.append(i)
        if j not in choice and len(choice)<n_object_per_assessment:
            choice.append(j)
        k+=1
    
    return choice



