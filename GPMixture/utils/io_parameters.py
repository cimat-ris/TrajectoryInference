import numpy as np
import sys
from gp_code.kernels import create_kernel_matrix

# Takes as an input a matrix of kernels. Exports the parameters, line by line
def write_parameters(matrix,rows,columns,fileName,kernelType='linePriorCombined'):
    f = open(fileName,"w")
    f.write('%d %d %s\n' % (rows,columns,kernelType))
    for i in range(rows):
        for j in range(columns):
            ker = matrix[i][j]
            if ker!=None:
                f.write('{:d} {:d} '.format(i,j))
                parameters = ker.get_parameters()
                for k in range(len(parameters)):
                    f.write('{:07.4f} '.format(parameters[k]))
                skip = "\n"
                f.write(skip)
    f.close()

# Read a parameter file and return the matrix of kernels corresponding to this file
def read_and_set_parameters(file_name, nParameters):
    try:
        file = open(file_name,'r')
    except OSError:
        print("[ERR] Could not open/read file: ",file_name)
        sys.exit()

    firstline = file.readline()
    header    = firstline.split()
    if len(header)<3:
        print("[ERR] Problem with the header of file: ",file_name)
        sys.exit()
    # Get rows, columns, kernelType from the header
    rows      = int(header[0])
    columns   = int(header[1])
    kernelType= header[2]
    print("[INF] Opening ",file_name," to read parameters of ",rows,"x",columns," kernels of type: ",kernelType)
    matrix, parametersMat = create_kernel_matrix(kernelType, rows, columns)

    for line in file:
        parameters = []
        parameters_str = line.split()
        i = int(parameters_str[0])
        j = int(parameters_str[1])
        for k in range(2,len(parameters_str)):
            parameters.append(float(parameters_str[k]))
        print("[INF] From goal ",i," to ", j, " parameters: ",parameters)
        matrix[i][j].set_parameters(parameters)
    file.close()
    return matrix
