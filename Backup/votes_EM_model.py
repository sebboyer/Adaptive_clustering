import numpy as np



# Generate a single vote
def Vote(k,i,j,Assessors,Delta):
    e=Assessors[k]
    delta=Delta[i,j]
    p=delta*(1-e)+e*(1-delta)
    r=np.random.rand(1)
    if r<p:
        y=1
    else:
        y=0
    return y


# Update Votes with the vote of k on ij if k has not already voted for pair i,j (according to the Assessor matrix and the Distance matrix)
def update_Votes(k,i,j,Votes,Assessors,Delta):
    if (i,j) not in Votes[k]:
        v=Vote(k,i,j,Assessors,Delta)
        Votes[k][i,j]=v
        Votes[k][j,i]=v
        print "Vote of Assessor "+str(k)+" on pair "+str(i)+","+str(j)+" is "+str(v)
    else:
        print "Assessor "+str(k)+" has already vote for pair "+str(i)+","+str(j)    

# Update votes for the list of assessors k_list and all the pair in pair_matrix
def Update_Votes(k_list,pair_matrix,Votes,Assessors,Delta):
    for x in range(0,len(k_list)):
        k=k_list[x]
        for pair in pair_matrix[k]:
            update_Votes(k,pair[0],pair[1],Votes,Assessors,Delta)

