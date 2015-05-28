from definitions import *
import montecarlo_experiments as mce
import read_experiment_table as ret



exp_file="experiment/exp_parameters.csv"
#exp_list=['a','b','c','d','e','f','g']+['a1','b1','c1','d1','e1','f1','g1']+['a0','b0','c0','d0','e0','f0','g0']
exp_list=['a6','b6','c6','d6','e6','f6','g6']
ret.conduct_experiment(exp_file,exp_list)





