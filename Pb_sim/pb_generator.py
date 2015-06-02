import Assessor_model as ass
import visualizations as vis
import real_clusters as rc
import os,sys


sys.path.append("../Tools")
sys.path.append("../Pb_sim")
import utils as utils
import Assessor_model as ass


def generate_objects(n_objects,n_clusters,dimension,tau):
    set_of_objects=list()
    # Generate clusters
    S=ass.SetOfObjects(n_clusters,n_objects,dimension)
    S.generate_clusters(tau)
    set_of_objects=S.set_of_objects

	# Compute the real Kmeans clusterization
    real_clusters=rc.kmeans_clusters(set_of_objects,n_clusters)
    fig=vis.visualization_2D(set_of_objects,real_clusters,0,1)
    return [set_of_objects,real_clusters,fig]


def generate_assessors(n_assessors,var_min,var_max,beta_param):
	step_var=(var_max-var_min)/float(n_assessors)
	set_of_assessors=[ass.Assessor(var_min+i*step_var,beta_param) for i in range(n_assessors)]

	return set_of_assessors

def generate_objects_from_csv(filename,n_objects,n_clusters,dimension):
	data=utils.extractArray_fromCSV(filename,False)
	opc=n_objects/n_clusters
	set_of_objects=list()

	for c in range(n_clusters):
		for i in range(opc):
			ob_ind=c*opc+i
			obj=data[ob_ind]
			o=ass.Object(dimension,ob_ind)
			o.set_features(obj[:dimension])
			set_of_objects.append(o)

	real_clusters=rc.kmeans_clusters(set_of_objects,n_clusters)
	fig=vis.visualization_2D(set_of_objects,real_clusters,0,1)
	return [set_of_objects,real_clusters,fig]


def generate_objects_from_csv2(filename,n_objects,n_clusters,dimension,class_col):
	data=utils.extractArray_fromCSV(filename,False)
	opc=n_objects/n_clusters
	set_of_objects=list()

	for c in range(n_clusters):
		for i in range(opc):
			ob_ind=c*opc+i
			obj=data[ob_ind]
			o=ass.Object(dimension,ob_ind)
			o.set_features(obj[:dimension])
			set_of_objects.append(o)

	real_clusters=rc.kmeans_clusters(set_of_objects,n_clusters)
	fig=vis.visualization_2D(set_of_objects,real_clusters,0,1)
	return [set_of_objects,real_clusters,fig]
	

