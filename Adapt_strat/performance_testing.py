# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import sys
sys.path.insert(0, '/home/sebastien/Documents/Adaptive clustering/Simulation python/Pb_sim')
import visualizations as vis

# <codecell>

def score(real_adj,est_adj):
    n=np.shape(real_adj)[0]
    d=real_adj-est_adj
    d=np.abs(d)
    res=np.sum(d)
    res=res/(n*(n-1))
    return res

def mutual_info(C1,C2,n_clusters):
	n=np.shape(C1)[0]
	I=0
	for i in range(n_clusters):
		ind_i1=[C1[k]==i for k in range(len(C1))]
		ci1=list(np.argwhere(ind_i1).T[0])
		for j in range(n_clusters):
			ind_j2=[C2[k]==i for k in range(len(C2))]
			cj2=list(np.argwhere(ind_j2).T[0])

			inter=len(set(ci1)&set(cj2))
			if inter!=0 and len(set(ci1))*len(set(cj2))!=0:
				I+=(inter/float(n))*np.log((n*inter)/float(len(set(ci1))*len(set(cj2))))
	return I
	
def n_mutual_info(C1,C2,n_clusters):
	I=mutual_info(C1,C2,n_clusters)
	C3=np.ones((1,len(C1)))-C2
	C3=list(C3[0])
	J=mutual_info(C1,C3,n_clusters)
	I=np.max([I,J])
	return I/float(entropy(C1,n_clusters)+entropy(C2,n_clusters))		

def entropy(C,n_clusters):
	n=np.shape(C)[0]
	H=0
	for i in range(n_clusters):
		ind_i=[C[k]==i for k in range(len(C))]
		ci=list(np.argwhere(ind_i).T[0])
		H+=-(len(ci)/float(n))*np.log(len(ci)/float(n))
	return H