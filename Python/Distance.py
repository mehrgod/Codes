# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 17:23:15 2020

@author: mirza
"""
import numpy as np


def create_matrix(path):
    
    p = []
    
    with open(path + "pattern.txt") as file:
        for line in file:
            p.append(line.strip())
    
    #print p
    
    fw = open(path + 'mp.txt', "w")
    
    for i in range(len(p)):
        row = ''
        for j in range(len(p)):
            row = row + ',' + str(nmlv(p[i].replace('_',''), p[j].replace('_','')))
            
        #print row
        fw.write(row[1:] + '\n')
        
    fw.close()
    
def nmlv(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in xrange(size_x):
        matrix [x, 0] = x
    for y in xrange(size_y):
        matrix [0, y] = y

    for x in xrange(1, size_x):
        for y in xrange(1, size_y):
            d = dis(seq1[x-1], seq2[y-1])
            if seq1[x-1] == seq2[y-1]:
                if (x-2 > -1) and (y-2 > -1):
                    if seq1[x-2] == seq2[y-2]:
                        matrix [x,y] = min(
                    matrix[x-2, y-1] + d,
                    matrix[x-1, y] + d,
                    
                    matrix[x-1, y] + d,
                    matrix[x-1, y-2] + d,
                    
                    matrix[x-1, y-1],
                    matrix[x-2, y-2],
                    
                    matrix[x, y-1] + d,
                    matrix[x, y-2] + d
                    )
                        
                matrix [x,y] = min(
                    matrix[x-1, y] + d,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + d
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + d,
                    matrix[x-1,y-1] + d,
                    matrix[x,y-1] + d
                )
    print (matrix)
    return (matrix[size_x - 1, size_y - 1])


def dis(x, y):
    d = 1
    if (x.lower() == y.lower()
    or (x.islower() and y.islower())
    or (x.isupper() and y.isupper())
    ):
        d = 0.5
    
    return d
    
def normal_s(path):
    with open(path + 'mp.txt') as file:
        arr = [[float(digit) for digit in line.split(',')] for line in file]
    
    A = np.array(arr)
    
    maxA = np.amax(A)
    
    B = maxA - A
    
    lower = 0
    upper = 0.25
    
    m, n = B.shape
    
    print m, n
    
    l = np.ndarray.flatten(B)
    
    minl = float(np.amin(l))
    maxl = float(np.amax(l))
    
    l_norm = [ (upper - lower) * (x - minl) / (maxl - minl) + lower for x in l]
    
    nB = np.reshape(l_norm, (m, n))
    
    np.savetxt(path + "/nmp.csv", nB, delimiter=",")
    
    print nB

if __name__ == "__main__":
    
    path = "C:/Project/EDU/OLI_175318/update/step/separateh/"
    
    #create_matrix(path)
    normal_s(path)