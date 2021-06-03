import numpy as np
import logging
import sys
from gp_code.kernels import create_kernel_matrix

# Takes as an input a matrix of kernels. Exports the parameters, line by line
def write_parameters(kernels,fileName,kernelType='linePriorCombined'):
    s = len(kernels)
    f = open(fileName,"w")
    f.write('%d %d %s\n' % (s,s,kernelType))
    for i in range(s):
        for j in range(s):
            ker = kernels[i][j]
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
        logging.error("Could not open/read file: ",file_name)
        sys.exit()

    firstline = file.readline()
    header    = firstline.split()
    if len(header)<3:
        logging.error("Problem with the header of file: ",file_name)
        sys.exit()
    # Get rows, columns, kernelType from the header
    rows      = int(header[0])
    columns   = int(header[1])
    kernelType= header[2]
    logging.info("Opening {:s} to read parameters of {:d}x{:d} kernels of type {}".format(file_name,rows,columns,kernelType))
    kernels = create_kernel_matrix(kernelType, rows, columns)
    # Read the parameters for the matrix entries
    for line in file:
        parameters = []
        parameters_str = line.split()
        i = int(parameters_str[0])
        j = int(parameters_str[1])
        for k in range(2,len(parameters_str)):
            parameters.append(float(parameters_str[k]))
        logging.info("From goal {:d} to {:d} parameters {}".format(i,j,parameters))
        kernels[i][j].set_parameters(parameters)
        kernels[i][j].optimized = True
    file.close()
    return kernels
