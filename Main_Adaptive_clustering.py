from definitions import *
import montecarlo_experiments as mce
import read_experiment_table as ret



exp_n=raw_input('Experiment number : ')

exp_file="experiment/exp_parameters.csv"
#exp_list=['a','b','c','d','e','f','g']+['a1','b1','c1','d1','e1','f1','g1']+['a0','b0','c0','d0','e0','f0','g0']

# exp_list=['a7','b7','c7','d7','e7','f7']
exp_list= ['a'+str(exp_n)]
ret.conduct_experiment(exp_file,exp_list)





