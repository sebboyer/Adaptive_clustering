# General importations

import numpy as np
import matplotlib.pyplot as plt
from copy import copy, deepcopy
import os,sys


sys.path.append("./Consensus")
sys.path.append("./Pb_sim")
sys.path.append("./Adapt_strat")
sys.path.append("./Tools")



import EM_algorithm as em
import votes_EM_model as vem
import Assessor_model as ass
import spectral_graph_clustering as spc
import visualizations as vis
import performance_testing as per
import real_clusters as rc
import pb_generator as pbg
import naive_methods as nm
import pickle as pck
import emails as emails 
import utils as utils
import graph_formulation as gra


def pointsOfUpdate(n_ass,K):
	res=list()
	for i in range(n_ass):
		if i%K==0 and i>0:
			res.append(i)
	return res


