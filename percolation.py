"""simulate Percolations on different graphs"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from math import sqrt, pi

# Fixing random state for reproducibility
np.random.seed(19680801)

# Standard coupling of edge percolation in a w by h rectangle of square lattice
# width w
w = 6
# height h
h = 3
# total number of edges is h*(w+1) vertical ones and (h+1)*w horizontal ones, the element x[i,j,k] denote the random unif[0,1] numbers at the edge i,j, k=0 horizontal and k=1 is vertical.
x = np.random.random((2*h+1,w+1,2))

print(x[6,6,1])

# the function compute_spin take an array x and a value 0<p<1, compute another binary array spin, spin[i]=0 means that edge i is empty
def lessthan(x,p):
    if x<p: return 0
    else: return 1
# make the function able to do vectorized operation
vlessthan = np.vectorize(lessthan)
# this function take an array x and a 0<p<1, return the same dimensional array value 0 if coordinate is less than p and value 1 otherwise.
def compute_spin(x,p:float,spin):
    return vlessthan(x,0.2)





