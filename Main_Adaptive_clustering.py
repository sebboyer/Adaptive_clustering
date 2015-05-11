from definitions import *
import montecarlo_experiments as mce
import read_experiment_table as ret



'''
############### Load objects
# One of the following :
# soo_100_3
# soo_500_easy
# soo_500_hard
# soo_50_easy
# soo_50_hard

[set_of_objects,real_clusters]=pck.load( open( "SetObjects/soo_100_3.p", "rb" ) )	

# ############# Load Assessors
# One of the following :
# soa_noisy
# soa_accurate

set_of_assessors=pck.load( open( "SetAssessors/soa_noisy.p", "rb" ) )

# ################  Creating adaptive method
# # The second function should be one of the following 
# # - rand_ob_gene      	random querys
# # - equi_ob_gene			less seen objects first
# # - diversity_ob_gene 	couples from different apriori clusters
# # - graph_ob_gene 		objects in decreasing order of betweeness (update at each evaluation of Delta only)
# # - graph2_ob_gene		best betweeness couple and then exclusive friends
# # - explore_ob_gene		cf paper
# # - exploitpw_ob_gene		cf paper

assessors_choice=nm.rand_ass_gene

##################  Conducting experiments

###########################################################################
###  Setting parameters of the experiment
###########################################################################
n_clusters=3
n_assessments=300
n_object_per_assessment=7
K=10
eval_num=pointsOfUpdate(n_assessments,K)
n_exps=50
em_prec=0.00001
EM=False
true_K=False  # Do the assessors know the number of clusters ? (Yes= Kmeans, No=GMM)
Known_K=False  # Do the experiment maker know the number of clusters ? (Yes= Kmeans, No=GMM)


###########################################################################
###  Number of the experiment
###########################################################################
nexp="17"


###########################################################################
###  Run experiment for different adaptive methods
###########################################################################

csvname="rand"+nexp
setOfObjects_choice=nm.rand_ob_gene
adaptive_method=ass.Adaptive_method(assessors_choice,setOfObjects_choice)
res=mce.repeat_exp(set_of_assessors,set_of_objects,n_clusters,adaptive_method,n_assessments,n_object_per_assessment,real_clusters,eval_num,n_exps,em_prec,EM,true_K,Known_K,csvname)

# csvname="diversity"+nexp
# setOfObjects_choice=nm.diversity_ob_gene
# adaptive_method=ass.Adaptive_method(assessors_choice,setOfObjects_choice)
# res=mce.repeat_exp(set_of_assessors,set_of_objects,n_clusters,adaptive_method,n_assessments,n_object_per_assessment,real_clusters,eval_num,n_exps,em_prec,EM,true_K,Known_K,csvname)

csvname="exploit"+nexp
setOfObjects_choice=nm.exploitpw_ob_gene	
adaptive_method=ass.Adaptive_method(assessors_choice,setOfObjects_choice)
res=mce.repeat_exp(set_of_assessors,set_of_objects,n_clusters,adaptive_method,n_assessments,n_object_per_assessment,real_clusters,eval_num,n_exps,em_prec,EM,true_K,Known_K,csvname)

# csvname="explore"+nexp
# setOfObjects_choice=nm.explore_ob_gene
# adaptive_method=ass.Adaptive_method(assessors_choice,setOfObjects_choice)
# res=mce.repeat_exp(set_of_assessors,set_of_objects,n_clusters,adaptive_method,n_assessments,n_object_per_assessment,real_clusters,eval_num,n_exps,em_prec,EM,true_K,Known_K,csvname)

csvname="centroid"+nexp
setOfObjects_choice=nm.centroid_ob_gene
adaptive_method=ass.Adaptive_method(assessors_choice,setOfObjects_choice)
res=mce.repeat_exp(set_of_assessors,set_of_objects,n_clusters,adaptive_method,n_assessments,n_object_per_assessment,real_clusters,eval_num,n_exps,em_prec,EM,true_K,Known_K,csvname)


#emails.sendMe_email("Congrats, the simulation has terminated successfully ! ")
'''


exp_file="experiment/exp_parameters.csv"
exp_list=['a']
ret.conduct_experiment(exp_file,exp_list)







