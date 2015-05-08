import Assessor_model as ass
import visualizations as vis
import real_clusters as rc




def generate_objects(n_objects,n_clusters,dimension,scale):
	set_of_objects=list()
	for i in range(n_clusters):
	    C=ass.Cluster(n_objects/n_clusters,scale*i,dimension)
	    C.populate_random_multivariate()
	    set_of_objects+=[C.objects[i] for i in range(n_objects/n_clusters)]
	real_clusters=rc.kmeans_clusters(set_of_objects,n_clusters)
	fig=vis.visualization_2D(set_of_objects,real_clusters,0,1)
	return [set_of_objects,real_clusters,fig]


def generate_assessors(n_assessors,var_min,var_max,beta_param):
	step_var=(var_max-var_min)/float(n_assessors)
	set_of_assessors=[ass.Assessor(var_min+i*step_var,beta_param) for i in range(n_assessors)]

	return set_of_assessors


	
