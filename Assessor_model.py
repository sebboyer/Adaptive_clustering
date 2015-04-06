
# coding: utf-8

# In[97]:

import numpy as np
from objects_cluster_generation import *
from sklearn.cluster import KMeans


# In[3]:

class Assessor:
    def __init__(self,volatility,n_feat_param):
        self.volatility=volatility
        self.n_feat_param=n_feat_param
    
    # Number of feature the assessor is aware of
    def number_feature_awareness(self,n_feat_total):
        return int(np.random.beta(2,self.n_feat_param)*n_feat_total)
    
    # The indices of the feature the assessor is aware of
    def indices_feature_awareness(self,n_feat_total):
        n_feat_awareness=self.number_feature_awareness(n_feat_total)
        a=np.random.dirichlet([1]*n_feat_total)
        return a.argsort()[-n_feat_awareness:]


# In[309]:

class Assessment:
    def __init__(self,assessor,set_of_objects):
        self.assessor=assessor
        self.set_of_objects=set_of_objects
        self.N=len(set_of_objects)
        self.n_feat=set_of_objects[0].get_n_feat()
        self.features=0
        
    # Extract only the feature the assessor is aware of from the set of objects
    def set_features(self):
        features=list()
        features_index=self.assessor.indices_feature_awareness(self.n_feat)
        for o in self.set_of_objects:
            noise=np.random.randn(len(features_index))*self.assessor.volatility
            features.append(o.features[features_index]+noise)
        features=np.array(features)
        self.features=features
        
    # Vote of the assessor based on the features he/she perceives
    def do_clusterization(self,n_cluster):
        km=KMeans(n_cluster)
        km.fit(self.features)
        return km.labels_


# In[1]:

class Experiment:
    
    # set_of_assessors = list of Assessor 
    # set_of_objects= list of objects 
    # assessment_list= list of list of the form [ind_assessor,[ind1,ind2,...,indk]]
    def __init__(self,set_of_assessors,set_of_objects,assessment_list):
        self.set_of_assessors=set_of_assessors
        self.set_of_objects=set_of_objects
        self.assessment_list=assessment_list
        self.results={}
        for a in range(len(set_of_assessors)):
            self.results[a]={}
    
    # For each line in assessment_list procede to the clustering and print result
    def procede_assessments(self):
        
        # For each line in assessment_list
        for i in range(len(self.assessment_list)):
            # Setting of the assessment
            assessor_ind=self.assessment_list[i][0]
            assessor=self.set_of_assessors[assessor_ind]
            set_of_objects=[self.set_of_objects[j] for j in self.assessment_list[i][1]]
            assessment=Assessment(assessor,set_of_objects)
            assessment.set_features()
            
            # Assessment
            result=assessment.do_clusterization(2)
            
            # Record result of the assessment in dictionnary
            for j in range(len(set_of_objects)):
                for k in range(j):
                    ind1=self.assessment_list[i][1][k]
                    ind2=self.assessment_list[i][1][j]
                    self.results[assessor_ind][ind1,ind2]=(result[k]==result[j])
            print result
            
    def get_results(self):
        return self.results


# In[342]:

# C=Cluster(30,1,3)
# C.populate_random_multivariate()
# D=Cluster(30,10,3)
# D.populate_random_multivariate()
# a=Assessor(5,5)
# b=Assessor(10,5)

# set_of_assessors=[a,b]
# set_of_objects=[C.objects[0],C.objects[1],D.objects[0],D.objects[1]]
# assessment_list=[[0,[0,1,2]],[1,[3,1,0]],[0,[3,2,1]]]
# exp=Experiment(set_of_assessors,set_of_objects,assessment_list)
# exp.procede_assessments()
# exp.get_results()

