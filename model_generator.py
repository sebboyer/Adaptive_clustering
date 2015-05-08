
from definitions import *



############### GENERATE SET OF OBJECTS



n_objects=100
n_clusters=3
dimension=2
scale=3

[set_of_objects,real_clusters,fig_ob]=pbg.generate_objects(n_objects,n_clusters,dimension,scale)
pck.dump( [set_of_objects,real_clusters],open( "SetObjects/soo_100_hard.p", "wb" ) )
fig_ob.savefig("SetObjects/soo_100_hard.png")




############### GENERATE SET OF ASSESSORS



# n_assessors=10
# var_min=0.1
# var_max=1
# beta_param=5
# set_of_assessors=pbg.generate_assessors(n_assessors,var_min,var_max,beta_param)
# pck.dump( set_of_assessors, open( "SetAssessors/soa_noisy.p", "wb" ) )