import csv
from definitions import *
import montecarlo_experiments as mce



def read_experiment(exp_file,exp_list):
	content=list()
	with open(exp_file, 'r') as csv_file:
	    reader= csv.reader(csv_file, delimiter = ',')
	    next(reader,None)
	    for row in reader:
	    	if row[0] in exp_list:
	        	content.append(row)
	return content


def conduct_experiment(exp_file,exp_list):
	content=read_experiment(exp_file,exp_list)
	for params in content:

		try:

			[set_of_objects,real_clusters]=pck.load( open( params[1], "rb" ) )
			set_of_assessors=pck.load( open( params[2], "rb" ) )
			assessors_choice=nm.rand_ass_gene

			n_clusters=int(params[3])
			n_assessments=int(params[4])
			n_object_per_assessment=int(params[5])
			K=int(params[6])
			eval_num=pointsOfUpdate(n_assessments,K)
			n_exps=int(params[7])
			em_prec=0.00001
			EM=int(params[8])
			true_K=int(params[9])  # Do the assessors know the number of clusters ? (Yes= Kmeans, No=GMM)
			Known_K=int(params[10])

			nexp=params[0]

			if int(params[11]):
				csvname="rand"+nexp
				setOfObjects_choice=nm.rand_ob_gene
				adaptive_method=ass.Adaptive_method(assessors_choice,setOfObjects_choice)
				res=mce.repeat_exp(set_of_assessors,set_of_objects,n_clusters,adaptive_method,n_assessments,n_object_per_assessment,real_clusters,eval_num,n_exps,em_prec,EM,true_K,Known_K,csvname)

			if int(params[12]):
				csvname="diversity"+nexp
				setOfObjects_choice=nm.diversity_ob_gene
				adaptive_method=ass.Adaptive_method(assessors_choice,setOfObjects_choice)
				res=mce.repeat_exp(set_of_assessors,set_of_objects,n_clusters,adaptive_method,n_assessments,n_object_per_assessment,real_clusters,eval_num,n_exps,em_prec,EM,true_K,Known_K,csvname)
			
			if int(params[13]):
				csvname="exploit"+nexp
				setOfObjects_choice=nm.exploitpw_ob_gene
				adaptive_method=ass.Adaptive_method(assessors_choice,setOfObjects_choice)
				res=mce.repeat_exp(set_of_assessors,set_of_objects,n_clusters,adaptive_method,n_assessments,n_object_per_assessment,real_clusters,eval_num,n_exps,em_prec,EM,true_K,Known_K,csvname)
			
			if int(params[14]):
				csvname="explore"+nexp
				setOfObjects_choice=nm.explore_ob_gene
				adaptive_method=ass.Adaptive_method(assessors_choice,setOfObjects_choice)
				res=mce.repeat_exp(set_of_assessors,set_of_objects,n_clusters,adaptive_method,n_assessments,n_object_per_assessment,real_clusters,eval_num,n_exps,em_prec,EM,true_K,Known_K,csvname)
			
			if int(params[15]):
				csvname="centroid"+nexp
				setOfObjects_choice=nm.centroid_ob_gene
				adaptive_method=ass.Adaptive_method(assessors_choice,setOfObjects_choice)
				res=mce.repeat_exp(set_of_assessors,set_of_objects,n_clusters,adaptive_method,n_assessments,n_object_per_assessment,real_clusters,eval_num,n_exps,em_prec,EM,true_K,Known_K,csvname)
			
			if int(params[16]):
				csvname="cheating"+nexp
				setOfObjects_choice=nm.cheating_ob_gene
				adaptive_method=ass.Adaptive_method(assessors_choice,setOfObjects_choice)
				res=mce.repeat_exp(set_of_assessors,set_of_objects,n_clusters,adaptive_method,n_assessments,n_object_per_assessment,real_clusters,eval_num,n_exps,em_prec,EM,true_K,Known_K,csvname)
			
			if int(params[17]):
				csvname="centroid_exploit"+nexp
				setOfObjects_choice=nm.centroid_exploit_ob_gene
				adaptive_method=ass.Adaptive_method(assessors_choice,setOfObjects_choice)
				res=mce.repeat_exp(set_of_assessors,set_of_objects,n_clusters,adaptive_method,n_assessments,n_object_per_assessment,real_clusters,eval_num,n_exps,em_prec,EM,true_K,Known_K,csvname)
			
		except:
			print "Error with experiment "+str(params[0])
		
