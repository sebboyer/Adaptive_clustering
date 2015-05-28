
from definitions import *



############### GENERATE SET OF OBJECTS



# n_objects=500
# n_clusters=10
# dimension=2
# tau=0.1

# [set_of_objects,real_clusters,fig_ob]=pbg.generate_objects(n_objects,n_clusters,dimension,tau)
# pck.dump( [set_of_objects,real_clusters],open( "SetObjects/soo_500_tau01.p", "wb" ) )
# fig_ob.savefig("SetObjects/soo_500_tau01.png")

# n_objects=[100,500]
# n_clusters=[5,10]
# dimension=8
# tau=[0.1,0.2,0.5,1,2]
# tau_string=["01","02","05","1","2"]

# for i in range(2):
# 	nob=n_objects[i]
# 	nc=n_clusters[i]
# 	for j in range(len(tau)):
# 		t=tau[j]
# 		name="SetObjects/soo_"+str(nob)+"_tau"+tau_string[j]
# 		[set_of_objects,real_clusters,fig_ob]=pbg.generate_objects(nob,nc,dimension,t)
# 		pck.dump( [set_of_objects,real_clusters],open( name+".p", "wb" ) )
# 		fig_ob.savefig(name+".png")


############### GENERATE SET OF ASSESSORS



# n_assessors=10
# var_min=0.1
# var_max=0.1
# beta_param=4
# set_of_assessors=pbg.generate_assessors(n_assessors,var_min,var_max,beta_param)
# pck.dump( set_of_assessors, open( "SetAssessors/soa_accurate.p", "wb" ) )