# simulate Percolations on different graphs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from math import sqrt, pi

# Fixing random state for reproducibility
# np.random.seed(19680801)

# Standard coupling of edge percolation in a w by h rectangle of square lattice
# width w
# height h
# total number of edges is h*(w+1) vertical ones and (h+1)*w horizontal ones, edges are labeled with vertex, the vertex (i,j) is associate to two edges, the right one (i,j)--(i+1,j) is recorded in the element x[i,j,0],  the up one (i,j)--(i,j+1) is recorded in x[i,j,1].

def sample_unif_x(w,h):
    return np.random.random((w+1,h+1,2))

# the function compute_spin take an array x and a value 0<p<1, compute another binary array spin, spin[i]=0 means that edge i is empty
def lessthan(x,p):
    if x<p: return 0
    else: return 1
# make the function able to do vectorized operation
vlessthan = np.vectorize(lessthan)
# this function take an array x and a 0<p<1, return the same dimensional array value 0 if coordinate is less than p and value 1 otherwise.
def compute_spin(x,p:float):
    return vlessthan(x,p)

# This function will give a percolation sample in a rectangle of height h, width w, parameter p, the output is two array X[(h+1,w,0)], the horizontal edges and X[(h,w+1,1)] the vertical edges. order the horizontal edges by (i,j,0); i=1..w; j= 1..h+1; from the bottom left corner, i coordinate go to the right and j coordinate go to the top. order the vertical edges by (i,j,1); i=1..w+1; j=1..h, same coordinate as the horizontals.
def simu_perco_square(w,h,p):
    return compute_spin(sample_unif_x(w,h),p)

# this function compute the clusters of a percolation sample x of width w and height h, on the square lattice, return a list cluster[i,j]=k where k indicate to which cluster the site (i,j) belongs to.
def find_all_cluster(x,w,h):
    #order the vertices by order(i,j)=i+1+(w+1)j, that is left to right, bottom to top. the variable order record the next unvisited vertex

    cluster=np.full((w+1,h+1),0)
    visited=np.full((w+1,h+1),0)
    k = 0
    myvertex = 1
    # as long as we havent treated the last myvertex, continue
    while myvertex<(w+1)(h+1):
        # set the cluster to ++1
        k = k+1
        # put the next site in myvertex in to the stack if the site is unvisited, otherwise myvertex ++
        if visited[[(myvertex-1) % (w+1), (myvertex-1) // (w+1)]]==0:
            stack.append([(myvertex-1) % (w+1), (myvertex-1) // (w+1)])
        else:
            myvertex = myvertex +1
        while stack != []:
            # pop the current myvertex from the stack and set its cluster label to k and mark as visited
            current=stack.pop(0)
            cluster[current[0],current[1]]=k
            visited[current[0],current[1]]=1
            #check all of its 4 neighbors, if neighbor is unvisited and connected to current site, then set its cluster label to k and marked visited and push this site into stack, otherwise do nothing
            #check the left neighbor, first coordinate must >0 to have a left neighbor
            if current[0]>0:
                if visited[current[0]-1,current[1]]==0 and x[current[0]-1,current[1],0]==1:
                    cluster[current[0]-1,current[1]]=k
                    visited[current[0]-1,current[1]]=1
                    stack.push[[current[0]-1,current[1]]]
            # check the right neighbor, first coordinate must <w to have a right neighbor
            if current[0]<w:
                if visited[current[0]+1,current[1]]==0 and x[current[0]+1,current[1],0]==1:
                    cluster[current[0]+1,current[1]]=k
                    visited[current[0]+1,current[1]]=1
                    stack.push[[current[0]+1,current[1]]]
            # check the up neighbor, second coordinate must <h to have such a neighbor
                if visited[current[0],current[1]+1]==0 and x[current[0],current[1]+1,0]==1:
                    cluster[current[0],current[1]+1]=k
                    visited[current[0],current[1]+1]=1
                    stack.push[[current[0],current[1]+1]]
            # check the down neighbor, second coordinate must >0
                if visited[current[0],current[1]-1]==0 and x[current[0],current[1]-1,0]==1:
                    cluster[current[0],current[1]-1]=k
                    visited[current[0],current[1]-1]=1
                    stack.push[[current[0],current[1]-1]]
    return cluster

sample=simu_perco_square(3,2,0.5)
print(sample)
print(find_all_cluster(sample,3,2))
#print(sq_perco_find_largest_cluster(sample,3,2))



