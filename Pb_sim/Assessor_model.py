
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

    # Vote of the assessor based on the features he/she perceives
    def do_randomK_clusterization(self):
        m=len(self.set_of_objects)
        # Choose K at random between 2 and m/2
        k=int(2+(m/2)*np.random.rand(1))
        km=KMeans(k)
        km.fit(self.features)
        return km.labels_


# In[1]:

class Experiment:
    
    # set_of_assessors = list of Assessor 
    # set_of_objects= list of objects 
    # assessment_list= list of list of the form [ind_assessor,[ind1,ind2,...,indk]]
    def __init__(self,set_of_assessors,set_of_objects,Adaptive_method):
        self.set_of_assessors=set_of_assessors
        self.set_of_objects=set_of_objects
        self.adaptive_method=Adaptive_method
        self.results={}
        for a in range(len(set_of_assessors)):
            self.results[a]={}
    
    # For each line in assessment_list procede to the clustering and print result
    def procede_assessments(self,n_cluster,assessment_list):
        
        # For each line in assessment_list
        for i in range(len(assessment_list)):
            # Setting of the assessment
            assessor_ind=assessment_list[i][0]
            assessor=self.set_of_assessors[assessor_ind]
            set_of_objects=[self.set_of_objects[j] for j in assessment_list[i][1]]
            assessment=Assessment(assessor,set_of_objects)
            assessment.set_features()
            
            # Assessment
            result=assessment.do_clusterization(n_cluster)
            
            # Record result of the assessment in dictionnary
            for j in range(len(set_of_objects)):
                for k in range(j):
                    ind1=assessment_list[i][1][k]
                    ind2=assessment_list[i][1][j]
                    self.results[assessor_ind][ind1,ind2]=(result[k]==result[j])
            #print result

    def procede_adaptive_assessment(self,n_cluster,true_K,Known_K,n_object_per_assessment,Delta_est,S_est,it):

        # Generate the next assessor from results
        assessor_ind=self.adaptive_method.assessor_generator(self.results,len(self.set_of_assessors),len(self.set_of_objects))
        assessor=self.set_of_assessors[assessor_ind]

        # Generate the next set of objects indexes from results
        set_of_objects_ind=self.adaptive_method.objects_generator(self.results,len(self.set_of_assessors),len(self.set_of_objects),n_object_per_assessment,Delta_est,S_est,n_cluster,Known_K,it)
        set_of_objects=[self.set_of_objects[j] for j in set_of_objects_ind]

        assessment=Assessment(assessor,set_of_objects)
        assessment.set_features()
            
        # Assessment
        if true_K:
            result=assessment.do_clusterization(n_cluster)
        else:
            result=assessment.do_randomK_clusterization()
            
        # Update results
        for j in range(len(set_of_objects)):
                for k in range(j):
                    ind1=set_of_objects_ind[k]
                    ind2=set_of_objects_ind[j]
                    self.results[assessor_ind][ind1,ind2]=(result[k]==result[j])
                
    def get_results(self):
        symmetrized_Votes(self.results)
        return self.results



class Adaptive_method:
    def __init__(self,assessor_generator_function,objects_generator_function):
        self.name='default'
        self.assessor_generator_function=assessor_generator_function
        self.objects_generator_function=objects_generator_function

    def assessor_generator(self,results,n_assessors,n_objects):
        return self.assessor_generator_function(results,n_assessors,n_objects)

    def objects_generator(self,results,n_assessors,n_objects,n_object_per_assessment,Delta_est,S_est,n_cluster,Known_K,it):
        return self.objects_generator_function(results,n_assessors,n_objects,n_object_per_assessment,Delta_est,S_est,n_cluster,Known_K,it)



## Functions


def symmetrized_Votes(votes):
    for key in votes:
        key_pairs=votes[key].keys()
        for key_pair in key_pairs:
            votes[key][key_pair[1],key_pair[0]]=votes[key][key_pair]


def generate_random_assessment(n_object_per_assessment,n_objects,n_assessors,n_assessments_per_assessor):
    res=list()
    for i in range(n_assessors):
        for k in range(n_assessments_per_assessor):
            ass_i=[i,list()]
            for j in range(n_object_per_assessment):
                ass_i[1].append(np.random.randint(0,n_objects))
            res.append(ass_i)
    return res


