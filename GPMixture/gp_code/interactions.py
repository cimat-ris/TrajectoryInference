"""
Functions to handle interaction of multiple agents
@author: karenlc
"""
from utils.manip_trajectories import euclidean_distance
import numpy as np
import math

#Dado un valor en [0,1], regresa un punto (x,y) con x en [trajx_i-1, trajx_i]... usando interpolacion lineal
def get_approximation(val, traj, i):
    x = val*traj[0][i-1] + (1-val)*traj[0][i]
    y = val*traj[1][i-1] + (1-val)*traj[1][i]
    return x,y

def potential_value(p,q):
    alpha, h = 0.9, 10.
    val = 1. - alpha*math.exp( (-1.)*(1./(2*h**2))*euclidean_distance(p,q) )
    return val

#Para un par de agentes regresa el valor del potencial
#En este caso si fi(t) existe pero fj(t) no, hacemos una interpolacion lineal para aproximar fj(t)
def interaction_potential_using_approximation(fi, fj):
    i,j = 0,0
    n, m = len(fi[2]), len(fj[2])
    potentialProduct = 1.
    while(i < n and j < m):
        potentialVal = 1.
        ti, tj = fi[2][i], fj[2][j]
        if ti < tj:
            if ti > fj[2][j-1]:
                val = (ti - tj)/(fj[2][j-1] - tj)
                x,y = get_approximation(val,fj,j)
                potentialVal = potential_value([fi[0][i],fi[1][i]], [x,y])
            if i < n:
                i += 1
        elif tj < ti:
            if tj > fi[2][i-1]:
                val = (tj - ti)/(fi[2][i-1] - ti)
                x,y = get_approximation(val,fi,i)
                potentialVal = potential_value([fj[0][j],fj[1][j]], [x,y])
            if j < m:
                j += 1
        else:
            potentialVal = potential_value([fj[0][j],fj[1][j]], [fi[0][i],fi[1][i]])
            if i < n:
                i += 1
            if j < m:
                j += 1
        potentialProduct *= potentialVal
    return potentialProduct

def interaction_potential_DCA(fi, fj):
    i,j = 0,0
    n, m = len(fi[2]), len(fj[2])
    potentialVal = 0.
    dca = 1100

    while(i < n and j < m):
        ti, tj = fi[2][i], fj[2][j]
        if ti < tj:
            if ti > fj[2][j-1]:
                val = (ti - tj)/(fj[2][j-1] - tj)
                x,y = get_approximation(val,fj,j)
                p = [fi[0][i],fi[1][i]]
                q = [x,y]
                dist = euclidean_distance(p,q)
                if dist < dca:
                    potentialVal = potential_value([fi[0][i],fi[1][i]], [x,y])
                    dca = dist
            if i < n:
                i += 1
        elif tj < ti:
            if tj > fi[2][i-1]:
                val = (tj - ti)/(fi[2][i-1] - ti)
                x,y = get_approximation(val,fi,i)
                p = [fj[0][j],fj[1][j]]
                q = [x,y]
                dist = euclidean_distance(p,q)
                if dist < dca:
                    potentialVal = potential_value([fj[0][j],fj[1][j]], [x,y])
                    dca = dist
            if j < m:
                j += 1
        else:
            p = [fj[0][j],fj[1][j]]
            q = [fi[0][i],fi[1][i]]
            dist = euclidean_distance(p,q)
            if dist < dca:
                dca = dist
                potentialVal = potential_value([fj[0][j],fj[1][j]], [fi[0][i],fi[1][i]])
            if i < n:
                i += 1
            if j < m:
                j += 1
    return potentialVal

#Distance of Closest Approach
def interaction_potential_DCA_(fi, fj):
    minDist = -1.
    iMin, jMin = 0, 0
    for i in range(len(fi[0])):
        for j in range(len(fj[0])):
            p = [fi[0][i], fi[1][i]]
            q = [fj[0][j], fj[1][j]]
            dist = euclidean_distance(p, q)
            if minDist < 0:
                minDist = dist
            elif dist < minDist:
                minDist = dist
                iMin = i
                jMin = j
    p = [fi[0][iMin], fi[1][iMin]]
    q = [fj[0][jMin], fj[1][jMin]]
    val = potential_value(p,q)

    return val

#Para un par de agentes regresa el valor del potencial
#En este caso si fi(t) existe pero fj(t) no, el valor del potencial en este t es 1
def interaction_potential(fi, fj):
    i,j = 0,0
    n, m = len(fi[2]), len(fj[2])
    potentialProduct = 1.
    while(i < n and j < m):
        potentialVal = 1.
        ti, tj = fi[2][i], fj[2][j]
        if abs(ti - tj) < 3:
            potentialVal = potential_value([fj[0][j],fj[1][j]], [fi[0][i],fi[1][i]])
            if i < n:
                i += 1
            if j < m:
                j += 1
        elif ti < tj:
            if i < n:
                i += 1
        elif tj < ti:
            if j < m:
                j += 1

        potentialProduct *= potentialVal
    return potentialProduct

def interaction_potential_for_a_set_of_trajectories(trajSet):
    # Number of pedestrians in the set
    n = len(trajSet)
    potentialProduct = 1.
    for i in range(n):
        for j in range(i+1,n):
            #Simple Interaction Potential
            #val = interaction_potential(trajSet[i],trajSet[j])
            #Interaction potential using approximation
            val = interaction_potential_using_approximation(trajSet[i],trajSet[j])
            #interaction potental using Distance of Closest Approach
            #val = interaction_potential_DCA(trajSet[i],trajSet[j])
            potentialProduct *= val
    return potentialProduct
