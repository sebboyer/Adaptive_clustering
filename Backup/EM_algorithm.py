

import numpy as np



# Initialize Delta_est and Assessors_est
def initialize_est(N,K):
    Delta_est=np.random.rand(N,N)
    Delta_est=0.5*(Delta_est+Delta_est.T)
    Assessors_est=np.random.rand(K)
    return Delta_est,Assessors_est




# Compute the matrix q_k for all k, where q^k_ij=E(d_ij/y,delta,e)
def compute_q(Delta_est,Assessors_est,Votes):
    N=np.shape(Delta_est)[0]
    q={}
    for k in range(len(Assessors_est)):
        q[k]=np.zeros((N,N))
        for pair in Votes[k]:
            i,j=pair
            d=Delta_est[i,j]
            e=Assessors_est[k]
            if Votes[k][i,j]==1:
                num=d*(1-e)
                den=d*(1-e)+(1-d)*e
            elif Votes[k][i,j]==0:
                num=d*e
                den=d*e+(1-d)*(1-e)
            else:
                print "Error"
            q[k][i,j]=num/float(den)
            q[k][j,i]=q[k][i,j]
    return q




def compute_Delta_est(q):
    N=np.shape(q[q.keys()[0]])[0]
    Delta_est=np.zeros((N,N))
    N_exp=np.zeros((N,N))
    
    # Look every pair of object i,j
    for i in range(N):
        for j in range(i):
            
            # For each pair sum the q_ij of all assessors having voted for the pair
            card_Nij=0
            for k in q.keys():
                # If Assessor k has already voted for pair i,j, add his vote to the count
                if q[k][i,j]!=0:
                    card_Nij+=1
                    Delta_est[i,j]+=q[k][i,j]
                else:
                    continue
                    
            # If pair i,j has at least one vote, compute the estimate for delta_ij otherwise set default value to -1
            if card_Nij!=0:
                Delta_est[i,j]=Delta_est[i,j]/float(card_Nij)
                N_exp[i,j]+=1
            else:
                continue

    N_exp[N_exp==0]=1
    S_est=np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            S_est[i,j]=np.sqrt(Delta_est[i,j]*(1-Delta_est[i,j])/float(N_exp[i,j]))
                
    Delta_est =0.5*(Delta_est +Delta_est.T)
    N_exp =N_exp +N_exp.T
    
    return Delta_est,S_est      




def compute_Assessors_est(q,Votes):
    K=len(q.keys())
    N=np.shape(q[q.keys()[0]])[0]
    Assessors_est=np.zeros((1,K))
    
    # Look at all assessors
    for k in q.keys():
        
        # For each assessor sum the terms in the expression of e_k
        card_Ik=0
        for i in range(N):
            for j in range(i):
                
                # If assessor k has already voted for pair i,j add his vote to the count
                if q[k][i,j]!=0:
                    Assessors_est[0,k]+=Votes[k][i,j]*q[k][i,j]+(1-Votes[k][i,j])*(1-q[k][i,j])
                    card_Ik+=1
                else:
                    continue
                # If assessor k has voted at least once, compute Assessors_est[k], otherwise default=-1
        if card_Ik!=0:
            Assessors_est[0,k]=1-Assessors_est[0,k]/float(card_Ik)
        else:
            Assessors_est[0,k]=-1
                    
    return Assessors_est[0]



# Main function
# Return the estimated distances between objects as well as the estimated Assessors' error
# Input :
#- Votes : A dictionnary whose keys are each one of the assessor and values are dictionnary whose keys are [i,j] and values are y_ij^k = 0 or 1
#- N : The number of objects
#- epsilon : the convergence threshold

def EM_est(Votes,N,epsilon):
    count=0
    K=len(Votes.keys())
    Delta_down,Assessors_down=initialize_est(N,K)
    S_est=np.zeros((N,N))
    q=compute_q(Delta_down,Assessors_down,Votes)
    Delta_up,S_est=compute_Delta_est(q)
    Assessors_up=compute_Assessors_est(q,Votes)
    while np.amax(abs(Delta_down-Delta_up))>epsilon:
        count+=1
        Delta_down,Assessors_down=Delta_up,Assessors_up
        q=compute_q(Delta_down,Assessors_down,Votes)
        Delta_up,S_est=compute_Delta_est(q)
        Assessors_up=compute_Assessors_est(q,Votes)
    print "Convergence reached in "+str(count)+" iterations"
    Delta=np.ones((N,N))-Delta_up
    return Delta,Assessors_up,S_est




