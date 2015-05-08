# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np

# <codecell>

def initialize_U(K,N):
    U=np.random.rand(K,N)
    for k in range(N):
        U[:,k]=U[:,k]/float(sum(U[:,k]))
    return U

# <codecell>

def updated_U(Delta,U,beta,m):
    
    # Calculate the K-means vectors
    K,n=np.shape(U)
    V=list()
    for i in range(K):
        v=U[i]/float(sum(U[i,:]))
        V.append(v)
    V=np.array(V)
    
    # Calculate the distances
    D=np.zeros((K,n))
    for i in range(K):
        D_beta=Delta+beta*(np.ones((n,n))-np.eye(n))
        DV=np.dot(D_beta,V[i])
        VDV=np.dot(V[i].T,DV)
        for k in range(n):
            D[i,k]=DV[k]-VDV/2.0
    
    # Update beta if necessary
    delta_beta=0
    E=np.eye(n)
    test=len(D[D<-10**(-10)])>0
    new_beta=beta
    if test:
    
        # Update delta_beta
        m=0
        for i in range(K):
            for k in range(n):
                v=-2*D[i,k]/float(np.dot(V[i]-E[k],(V[i]-E[k]).T))
                m=max(m,v)
        delta_beta=m+0.001
        
        # Update D_ik
        for i in range(K):
            for k in range(n):
                D[i,k]=D[i,k]+(delta_beta/2.0)*np.dot(V[i]-E[k],(V[i]-E[k]).T)
        
        #Update beta
        new_beta=beta+delta_beta
    
    # Update U
    for k in range(n):
        for i in range(K):
            seq=[(D[i,k]/D[j,k])**(1/float(m-1)) for j in range(K)]
            s=sum(seq)
            U[i,k]=1/s        

    return U,new_beta

# <codecell>

def NERFCM(D,K,m,epsilon):
    n=np.shape(D)[0]
    U_minus=initialize_U(K,n)
    beta=0
    diff=2*epsilon
    while diff>epsilon:
        U_plus,beta=updated_U(D,U_minus,beta,m)
        diff=np.max(np.abs(U_minus-U_plus))
        U_minus=U_plus
    return U_minus 

def NERFCM_average_Adj(D,K,m,epsilon,it):
    Res_Adj=0
    Res_U=0
    Res_U_proba=0
    for i in range(it):
        U=NERFCM(D,K,m,epsilon)
        Res_U+=U
        Adj=max_likelihood_adj(U)
        Res_Adj+=Adj
        max_U=max_likelihood_clusters(U)
        Res_U_proba+=max_U
    Res_Adj=Res_Adj/it
    Res_U=Res_U/it
    Res_U_proba=Res_U_proba/it
    print Res_Adj
    print Res_U
    print Res_U_proba
    n=np.shape(Res_Adj)[0]
    for i in range(n):
        for j in range(n):
            if Res_Adj[i,j]>0.5:
                Res_Adj[i,j]=1
            else:
                Res_Adj[i,j]=0
    return Res_Adj

# <codecell>

def max_likelihood_adj(U):
    K,n=np.shape(U)
    Adj=np.zeros((n,n))
    d={}
    for i in range(K):
        d[i]=list()
    for k in range(n):
        d[np.argmax(U[:,k])].append(k)
    for i in range(K):
        for j in d[i]:
            for l in d[i]:
                Adj[j,l]=1
                Adj[l,j]=1
    return Adj

def max_likelihood_clusters(U):
    K,n=np.shape(U)
    clusters=np.zeros((K,n))
    d={}

    for j in range(n):
        clusters[np.argmax(U[:,j]),j]=1               
    
    return clusters

# <codecell>

# N=5
# D=np.zeros((N,N))
# D[0,3]=1.9
# D[0,4]=1
# D[1,3]=1
# D[1,4]=1.9
# D[2,3]=1.8
# D[2,4]=1.9
# D[0,1]=0.2
# D[0,2]=0.2
# D[2,1]=0.2
# D[3,4]=0.2
# D=(0.5)*(D+D.T)
# Delta=D
# print Delta

# # <codecell>

# Delta=[[ 0. ,0.00170612,  0.00169131,  0.02469532,  0.0017214,   0.01065122],
#  [ 0.00170612 , 0.        ,  0.00156305 , 0.02263081 , 0.00165207 , 0.02359098],
#  [ 0.00169131 , 0.00156305  ,0.         , 0.02215095  ,0.00170867 , 0.02426879],
#  [ 0.02469532 , 0.02263081 , 0.02215095 , 0.          ,0.00105833 , 0.00064471],
#  [ 0.0017214  , 0.00165207 , 0.00170867 , 0.00105833 , 0.          ,0.00068122],
#  [ 0.01065122 , 0.02359098 , 0.02426879 , 0.00064471  ,0.00068122 , 0.        ]]

# # <codecell>

# Delta

# # <codecell>

# m=2
# K=2
# epsilon=0.0000001
# U=NERFCM(Delta,K,m,epsilon)

# # <codecell>

# U

# # <codecell>

# it=1000
# Adj=NERFCM_average_Adj(Delta,K,m,epsilon,it)

# # <codecell>

# print Adj

# <codecell>


