# -*- coding: utf-8 -*-
"""
Functions to handle interaction of multiple agents
@author: karenlc
"""

from path import *
from utils.plotting import *
from kernels import *
from statistics import*
from sampling import*
import matplotlib.pyplot as plt
import numpy as np
import math
from copy import copy

timeRange = 5

#Busco un alpha en [0,1] tal que t = alpha*T1 + (1-alpha)*T2
def search_value(a, b, t, T1, T2):
    alpha = (a+b)/2
    val = (1-alpha)*T1 + alpha*T2
    if(abs(t-val) < timeRange):
        return alpha
    elif val > t:
        return search_value(a,alpha,t,T1,T2)
    elif val < t:
        return search_value(alpha,b,t,T1,T2)

#Dado un valor en [0,1], regresa un punto (x,y) con x en [path.x_i-1, pathx_i]... usando interpolacion lineal
def get_approximation(val,path,index):
    _x = (1-val)*path.x[index-1] + val*path.x[index]
    _y = (1-val)*path.y[index-1] + val*path.y[index]
    _t = (1-val)*path.t[index-1] + val*path.t[index]
    return _x,_y

def potential_value(p,q):
     alpha, h = 0.9, 10.
     val = 1. - alpha*math.exp( (-1.)*(1./(2*h**2))*euclidean_distance(p,q) )
     return val

#Para un par de agentes regresa el valor del potencial
#En este caso si fi(t) existe pero fj(t) no, hacemos una interpolacion lineal para aproximar fj(t)
def interaction_potential_using_approximation(fi, fj):
    i,j = 0,0
    n, m = len(fi.t), len(fj.t)
    potentialProduct = 1.
    while(i < n and j < m):
        potentialVal = 1.
        ti, tj = fi.t[i], fj.t[j]
        #checar si el tiempo coincide
        #print("[Current time]: i=",fi.t[i],", j=",fj.t[j])
        if ti < tj:
            if ti > fj.t[j-1]:
                val = search_value(0,1,ti,fj.t[j-1],tj)
                x,y = get_approximation(val,fj,j)
                potentialVal = potential_value([fi.x[i],fi.y[i]], [x,y])
            if i < n:
                i += 1
        elif tj < ti:
            if tj > fi.t[i-1]:
                val = search_value(0,1,tj,fi.t[i-1],ti)
                x,y = get_approximation(val,fi,i)
                potentialVal = potential_value([fj.x[j],fj.y[j]], [x,y])
            if j < m:
                j += 1
        else:
            potentialVal = potential_value([fj.x[j],fj.y[j]], [fi.x[i],fi.y[i]])
            if i < n:
                i += 1
            if j < m:
                j += 1
        potentialProduct *= potentialVal
        print("Potential value:", potentialVal)
        print("Potential product:",potentialProduct)
    #print("Potential product:",potentialProduct)

#Para un par de agentes regresa el valor del potencial
#En este caso si fi(t) existe pero fj(t) no, el valor del potencial en este t es 1
def interaction_potential(fi, fj):
    i,j = 0,0
    n, m = len(fi.t), len(fj.t)
    potentialProduct = 1.
    while(i < n and j < m):
        potentialVal = 1.
        ti, tj = fi.t[i], fj.t[j]
        #print("[Current time]: i=",ti,", j=",tj)
        if abs(ti - tj) < 3:
            potentialVal = potential_value([fj.x[j],fj.y[j]], [fi.x[i],fi.y[i]])
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
        #print("Potential value:", potentialVal)
    #print("Potential product:",potentialProduct)
    return potentialProduct

def interaction_potential_for_a_set_of_pedestrians(pathSet):
    n = len(pathSet)
    potentialProduct = 1.
    for i in range(n):
        for j in range(i+1,n):
            val = interaction_potential(pathSet[i],pathSet[j])
            potentialProduct *= val

    print("Potential product of set:",potentialProduct)
