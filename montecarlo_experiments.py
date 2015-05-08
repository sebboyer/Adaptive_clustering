
from definitions import *
import csv


def procede_exp(set_of_assessors,set_of_objects,n_clusters,adaptive_method,n_assessments,n_object_per_assessment,real_clusters,eval_num,em_prec,EM,true_K,Known_K):
    result=list()
    n_objects=len(set_of_objects)
    exp=ass.Experiment(set_of_assessors,set_of_objects,adaptive_method)
    Delta_est=0
    S_est=0

    for j in range(n_assessments):

        # Choose and submit new query 
        exp.procede_adaptive_assessment(n_clusters,true_K,Known_K,n_object_per_assessment,Delta_est,S_est)

        # Evaluate the quality of the new clusters
        if j in eval_num:

            # Compute estimated distance matrix
            Votes=exp.get_results()
            if EM:
                Delta_est,Assessors_est,S_est=em.EM_est(Votes,n_objects,em_prec)
            else:
                Delta_est=em.naive_distance(Votes,n_objects)

            # Estimating Adjacency from distance using equal size kmeans
            if Known_K:
                #est_adj=spc.adjacency_2Clusters_eq(Delta_est,n_objects/2,n_objects/2,2)
                est_adj=spc.adjacency_KClusters(Delta_est,n_clusters)
            else:
                est_adj=spc.adjacency_Clusters(Delta_est,n_clusters)

            est_clusters=vis.clusters_from_adjacency(est_adj,n_clusters)
            s=per.n_mutual_info(real_clusters,est_clusters,2)
            result.append(s)
    return result

        
def repeat_exp(set_of_assessors,set_of_objects,n_clusters,adaptive_method,n_assessments,n_object_per_assessment,real_clusters,eval_num,n_exps,em_prec,EM,true_K,Known_K,csvname):
    results=list()
    count_badexp=0

    writefile="Results/"+csvname+".csv"
    fd = open(writefile,'wb')

    for i in range(n_exps):
        print "Start experiment number "+str(i)
        #try:
        res=procede_exp(set_of_assessors,set_of_objects,n_clusters,adaptive_method,n_assessments,n_object_per_assessment,real_clusters,eval_num,em_prec,EM,true_K,Known_K)
        results.append(res)
        writer=csv.writer(fd)
        writer.writerow(res)
        # except :
        #     print "The experiment number "+str(i)+" failed !"
        #     count_badexp+=1
    #print str(100*count_badexp/float(n_exps))+"% of the experiments didn't finished"
    fd.close()

    return np.array(results)


def fig_exp(L_res,n_ass,eval_num,display_err):
    fig, ax = plt.subplots(figsize=(16, 6))
    x=eval_num

    L_res=list(L_res)
    for i in range(len(L_res)):
        res=L_res[i]
        m=np.mean(res,axis=0)
        v=np.std(res,axis=0)
        y=m
        err=v   
        if display_err:
            ax.errorbar(x, y,yerr=v)
        else:
            ax.plot(x,y)
    
    ax.set_ylim((0, 1))
    ax.set_xlim((0, max(x)))
    ax.set_ylabel("Normalized Mutual information", fontsize=16)
    ax.set_xlabel("Number of assessment", fontsize=16)

    return fig

