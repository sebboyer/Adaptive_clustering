# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
#%matplotlib inline

# <codecell>

def generate_mean(dim,mag):
    return mag*np.random.randn(dim)+mag


def generate_covariance(dim,var):
    Cov=(np.random.rand(dim,dim)-0.5)*0.2
    for i in range(dim):
        Cov[i,i]=var[i]
    return Cov



def generate_samples(n,mean,cov):
    return np.random.multivariate_normal(mean,cov,n)

# <codecell>

def generate_sample_list(mag,dim,n,K):
    L=list()
    for i in range(K):
        mean=generate_mean(dim,mag)
        cov=generate_covariance(dim)
        samp=generate_samples(n,mean,cov)
        L.append(samp)
    return L

# <codecell>

def display_2D(samp_list,dim1,dim2):
    colors="rgbcmykw"
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for i in range(len(samp_list)):
        #c=to_rgb(colors[i])
        samp=samp_list[i]
        x=samp[:,dim1]
        y=samp[:,dim2]
        ax.scatter(x, y, alpha=0.5,color=(i/float(len(samp_list)),0,1))
    fig.show()

# <codecell>

# mag=10
# dim=5
# n=30
# K=2
# L=generate_sample_list(mag,dim,n,K)
# proj1=0
# proj2=1
# display_2D(L,proj1,proj2)

# <codecell>

class Object:
    def __init__(self,n_feat,index):
        self.ind=index
        self.features=np.zeros((1,n_feat))
    def get_n_feat(self):
        return np.shape(self.features)[0]
    def set_features(self,samp):
        self.features=samp
    def print_features(self):
        fig, ax = plt.subplots(figsize=(8, 3.5))
        width=0.2
        ax.bar(np.array(range((len(self.features))))-width/2.0,self.features,width)
        ax.set_xticks(ticks= range((len(self.features))))
        ax.set_xlim(-0.6,len(self.features)-0.4)
        ax.set_ylim(0,10)
        fig.show()
    def select_features(list_indices):
        return self.features[0,indices]

# <codecell>

class Cluster:
    def __init__(self,n_objects,mag,dim):
        self.N=n_objects
        self.objects=list()
        self.dim=dim
        self.mean=generate_mean(dim,mag)
        self.cov=generate_covariance(dim)


    def add_object(self,ob):
        self.objects.append(ob)

    def add_objects(self,list_objects):
        self.objects+=list_objects

    def select_objects(self,indices):
        return self.objects[indices]

    def populate_random_multivariate(self):
        samp=generate_samples(self.N,self.mean,self.cov)
        for i in range(self.N):
            o=Object(self.dim,i)
            o.set_features(samp[i,:])
            self.add_object(o)




class SetOfObjects:
    def __init__(self,n_clusters,n_objects,dim):
        self.dim=dim
        self.nob=n_objects
        self.nclust=n_clusters
        self.set_of_objects=list()
        self.set_of_clusters=list()
        self.means=np.zeros((n_clusters,dim))
        self.variances=np.zeros((n_clusters,dim))

    def generate_clusters(self,tau):

        mag=100
        for d in range(self.dim):

            # generate means along the 1-dimensional axis
            M=mag*np.random.randn(self.nclust)+mag
            index_array=np.argsort(np.argsort(M))
            m=np.sort(M)
            self.means[:,d]=M

            # Assign sigma
            s1=999999
            s2=(m[1]-m[0])/float(tau)
            for c in range(self.nclust-2):
                sigma=np.minimum(s1,s2)
                self.variances[c,d]=sigma
                s1=(m[c+1]-m[c])/float(tau)
                s2=(m[c+2]-m[c+1])/float(tau)

            sigma=np.minimum(s1,s2)
            self.variances[self.nclust-2,d]=sigma

            s1=(m[self.nclust-1]-m[self.nclust-2])/float(tau)
            s2=999999
            sigma=np.minimum(s1,s2)
            self.variances[self.nclust-1,d]=sigma

        self.variances=self.variances[index_array,:]

        for c in range(self.nclust):
            ob_per_clust=self.nob/self.nclust
            samp=generate_samples(ob_per_clust,self.means[c,:],generate_covariance(self.dim,self.variances[c,:]))
            for i in range(ob_per_clust):
                o=Object(self.dim,i)
                o.set_features(samp[i,:])
                self.set_of_objects.append(o)



# <codecell>

def compute_KL(cluster1,cluster2):
    sigma2_inv=np.linalg.inv(cluster2.cov)
    det1=np.linalg.det(cluster1.cov)
    det2=np.linalg.det(cluster2.cov)
    tr_s2inv_s1=np.trace(np.dot(sigma2_inv,cluster1.cov))
    a=np.dot(np.dot(cluster2.mean-cluster1.mean,sigma2_inv),(cluster2.mean-cluster1.mean).T)
    res=0.5*(np.log(det2/float(det1))-cluster1.dim+tr_s2inv_s1+a)
    if abs(res)<10**(-15):
        res=0
    return res

# <codecell>

# C=Cluster(30,1,3)
# C.populate_random_multivariate()
# C.objects[1].print_features()

# # D=Cluster(30,10,3)
# # D.populate_random_multivariate()
# # D.objects[1].print_features()

# # kl=compute_KL(C,D)
# # print kl

# # <codecell>

# C.objects[1].get_n_feat()

# <codecell>


