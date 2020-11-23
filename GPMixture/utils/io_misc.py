import numpy as np
import math

def write_data(data, fileName):
    n = len(data)
    f = open(fileName,"w+")
    for i in range(n):
        s = "%d\n"%(data[i])
        f.write(s)
    f.close()

def read_data(fileName):
    data = []
    f = open(fileName,'r')
    lines = f.readlines()
    for i in range(len(lines)):
        lines[i] = lines[i].strip("\n")
        data.append( float(lines[i]) )
    f.close()
    return data

# Euclidean distance
def euclidean_distance(p, q): #p = (x,y)
    dist = math.sqrt( (p[0]-q[0])**2 + (p[1]-q[1])**2 )
    return dist
