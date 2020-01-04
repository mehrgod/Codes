# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 12:22:00 2019

@author: mirza
"""

import numpy as np
#import Levenshtein as lv
import os
import matplotlib
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import csv
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans 


def normal_s(path):
    with open(path + 'mp.txt') as file:
        arr = [[float(digit) for digit in line.split(',')] for line in file]
    
    A = np.array(arr)
    
    maxA = np.amax(A)
    
    B = maxA - A
    
    lower = 0
    upper = 0.23
    
    m, n = B.shape
    
    print m, n
    
    l = np.ndarray.flatten(B)
    
    minl = float(np.amin(l))
    maxl = float(np.amax(l))
    
    l_norm = [ (upper - lower) * (x - minl) / (maxl - minl) + lower for x in l]
    
    nB = np.reshape(l_norm, (m, n))
    
    np.savetxt(path + "/nms.csv", nB, delimiter=",")
    
    print nB


def normal_p(path):
    with open(path + 'p.txt') as file:
        arr = [[float(digit) for digit in line.split(',')] for line in file]
    
    A = np.array(arr)
    
    lower = 0
    upper = 0.3
    
    m, n = A.shape
    
    print m, n
    
    l = np.ndarray.flatten(A)
    
    minl = float(np.amin(l))
    maxl = float(np.amax(l))
    
    l_norm = [ (upper - lower) * (x - minl) / (maxl - minl) + lower for x in l]
    
    nA = np.reshape(l_norm, (m, n))
    
    np.savetxt(path + "/normal_p.csv", nA, delimiter=",")
    
    print nA

def create_matrix(path):
    
    #path = "C:/Project/EDU/files/2013/example/Topic/similarity/"
    
    p = []
    
    with open(path + "pattern.txt") as file:
        for line in file:
            p.append(line.strip())
    
    print p
    
    fw = open(path + 'mp.txt', "w")
    
    for i in range(len(p)):
        row = ''
        for j in range(len(p)):
            #row = row + ',' + str(levenshtein(p[i], p[j]))
            row = row + ',' + str(mlv(p[i], p[j]))
            
        print row
        fw.write(row[1:] + '\n')
        
    fw.close()
        
def levenshtein(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in xrange(size_x):
        matrix [x, 0] = x
    for y in xrange(size_y):
        matrix [0, y] = y

    for x in xrange(1, size_x):
        for y in xrange(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    #print (matrix)
    return (matrix[size_x - 1, size_y - 1])

def mlv(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in xrange(size_x):
        matrix [x, 0] = x
    for y in xrange(size_y):
        matrix [0, y] = y

    for x in xrange(1, size_x):
        for y in xrange(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                d = dis(seq1[x-1], seq2[y-1])
                matrix [x,y] = min(
                    matrix[x-1,y] + d,
                    matrix[x-1,y-1] + d,
                    matrix[x,y-1] + d
                )
    print (matrix)
    return (matrix[size_x - 1, size_y - 1])

def dis(x, y):
    #x = ""
    #y = ""
    d = 1
    if (x.lower() == y.lower()
    or (x.islower() and y.islower())
    or (x.isupper() and y.isupper())
    ):
        d = 0.5
    
    
    return d

def test_lv():
    path = "C:/Project/EDU/files/2013/example/Topic/similarity/pattern.txt"
    
    p = []
    
    with open(path) as file:
        for line in file:
            p.append(line.strip())
            
    for ptrn in p:
        print p[0], ptrn
        print lv.distance(p[0], ptrn)
        print levenshtein(p[0], ptrn)
        print mlv(p[0], ptrn)
    
def column_center(X):
    avg = np.mean(X, axis=0)
    
    new_row = []
    
    for row in X:
        new_row.append(row - avg)
        
    return np.vstack(new_row)

def normal_similarity(path):
    #path = "C:/Project/EDU/files/2013/example/Topic/similarity/"
    
    with open(path + 'mp.txt') as file:
        array = [[float(digit) for digit in line.split('\t')] for line in file]
        
    p = np.array(array)
    
    n = column_center(p)
    
    np.savetxt(path + "/nmp.csv", n, delimiter=",")

def merge_errors(path, k):
    #path = "C:/Project/EDU/files/2013/example/Topic/similarity/it/10/k" + str(k) + "/"
    path = path + "k" + str(k) + "/"
    
    errorsX1 = []
    errorsX2 = []
    errorsS1 = []
    errorsS2 = []
    
    for i in range(1,k):
        x = 'c' + str(i) + 'd' + str(k-i)
        pathn = path + x
        with open(pathn + "/err.txt") as file:
                array2d = [line for line in file]
                errorsX1.append(array2d[0].split()[2])
                errorsX2.append(array2d[1].split()[2])
                errorsS1.append(array2d[2].split()[2])
                errorsS2.append(array2d[3].split()[2])
            
        
    fw1 = open(path + "/errsX1.txt", "w")
    for e in errorsX1:
        fw1.write(e.strip()+",")
    
    fw2 = open(path + "/errsX2.txt", "w")
    for e in errorsX2:
        fw2.write(e.strip()+",")
    
    fws1 = open(path + "/errsS1.txt", "w")
    for e in errorsS1:
        fws1.write(e.strip()+",")
    
    fws2 = open(path + "/errsS2.txt", "w")
    for e in errorsS2:
        fws2.write(e.strip()+",")
    
    fw1.close()
    fw2.close()
    fws1.close()
    fws2.close()


def take_avg_easy(path, value):
    #path = "C:/Project/EDU/files/2013/example/Topic/similarity/it/10/"
    
    errorX1 = []
    errorX2 = []
    errorS1 = []
    errorS2 = []
    
    for k in os.listdir(path):
        if k == "k" + str(value):
            pathnn = path + k
                        
            with open(pathnn + "/errsX1.txt") as X1:
                for line in X1:
                    errorX1.append(line)
                
            with open(pathnn + "/errsX2.txt") as X2:
                for line in X2:
                    errorX2.append(line)
                
            with open(pathnn + "/errsS1.txt") as S1:
                for line in S1:
                    errorS1.append(line)
            
            with open(pathnn + "/errsS2.txt") as S2:
                for line in S2:
                    errorS2.append(line)            
    
    fw1 = open(path + 'k' + str(value) + 'errX1.txt', "w")
    fw2 = open(path + 'k' + str(value) + 'errX2.txt', "w")
    fwS1 = open(path + 'k' + str(value) + 'errS1.txt', "w")
    fwS2 = open(path + 'k' + str(value) + 'errS2.txt', "w")
    
    
    for i in errorX1:
        fw1.write(i + "\n")
        print i + "\n"
    
    for i in errorX2:
        fw2.write(i + "\n")
        print i + "\n"
    
    for i in errorS1:
        fwS1.write(i + "\n")
        print i + "\n"
    
    for i in errorS2:
        fwS2.write(i + "\n")
        print i + "\n"
    
    
    fw1.close()
    fw2.close()
    fwS1.close()
    fwS2.close()    

def to_plot(path, m):
    #path = "C:/Project/EDU/files/2013/example/Topic/similarity/it/10/"
    
    
    #fw = open(path + 'AllplotX1.txt', 'w')
    #fw = open(path + 'AllplotX2.txt', 'w')
    #fw = open(path + 'AllplotS1.txt', 'w')
    #fw = open(path + 'AllplotS2.txt', 'w')
    fw = open(path + 'Allplot' + m + '.txt', 'w')
    s = ''
    for i in range(5,21):
        for j in range(1,i):
            s = s + str(i) + "," + str(j) + "\t"
    fw.write(s[0:-1] + '\n')
    s = ''
    for i in range(5,21):
        #fname = path + 'k' + str(i) + 'errX1.txt'
        #fname = path + 'k' + str(i) + 'errX2.txt'
        #fname = path + 'k' + str(i) + 'errS1.txt'
        #fname = path + 'k' + str(i) + 'errS2.txt'
        fname = path + 'k' + str(i) + 'err' + m + '.txt'
        with open(fname) as f:
            for line in f:
                s += line.strip()
    fw.write(s[0:-1])
    fw.close()

def ave_error_ci(path, m):
    
    vecs = []
    k1 = 5
    k2 = 20
    
    
    for i in range(1,11):
        pathn = "C:/Project/EDU/files/2013/example/Topic/similarity/it/" + str(i) + "/"
        #with open(path + 'AllplotX1.txt') as f:
        #with open(path + 'AllplotX2.txt') as f:
        #with open(pathn + 'AllplotS1.txt') as f:
        #with open(path + 'AllplotS2.txt') as f:
        with open(pathn + 'Allplot' + m + '.txt') as f:
            data = f.readlines()
            vecs.append(data[1])
    
    vecs_avg = take_avg_vec(vecs)
    vecs_std = take_std_vec(vecs)
    
    l = len(vecs_avg)
    
    vecs_l = np.zeros(l)
    vecs_h = np.zeros(l)
    
    for i in range(l):
        vecs_l[i] = vecs_avg[i] - (1.96 * vecs_std[i])/np.sqrt(10)
        vecs_h[i] = vecs_avg[i] + (1.96 * vecs_std[i])/np.sqrt(10)
    
    print vecs_l
    print vecs_h
    
    #fw1 = open('C:/Project/EDU/files/2013/example/Topic/similarity/it/S2_avg_d_std_95.txt','w')
    #fw1.write(list_to_str(vecs_avg) + '\n' + list_to_str(vecs_std))
    
    #fw2 = open('C:/Project/EDU/files/2013/example/Topic/similarity/it/S2_ci_d_95.txt','w')
    #fw2.write(list_to_str(vecs_l) + '\n' + list_to_str(vecs_h))
    
    #fw3= open(path + '/S1_avg_label.txt','w')
    fw3= open(path + '/' + m +'_avg_label.txt','w')
    fw3.write(print_label(k1, k2) + '\n' + list_to_str_tab(vecs_avg))
    
    #fw1.close()
    #fw2.close()
    fw3.close()

def list_to_str(l):
    s = ''
    for i in l:
        s = s + ',' + str(i)
    return s[1:]

def list_to_str_tab(l):
    s = ''
    for i in l:
        s = s + '\t' + str(i)
    return s[1:]

def take_avg_vec(l):
    vecs = []
    for line in l:
        vec = [float(digit) for digit in line.split(',')]
        vecs.append(vec)
        
    return np.average(vecs, axis=0)

def take_std_vec(l):
    vecs = []
    for line in l:
        vec = [float(digit) for digit in line.split(',')]
        vecs.append(vec)
        
    return np.std(vecs, axis=0)

def print_label(k1, k2):
    s = ""
    for i in range(k1, k2+1):
        for j in range(1, i):
            #print i, j
            s = s + "\t" + str(i) + "," + str(j)
    return s[1:]

def plot_avg(path, m):
    #path = "C:/Project/EDU/files/2013/example/Topic/similarity/it/"
    
    #f = open(path + "S1_avg_label.txt")
    f = open(path + m + "_avg_label.txt")
    
    line = f.readlines()
    
    l = [token.strip() for token in line[0].split('\t')]
    l1 = [float(token.strip()) for token in line[1].split('\t')]
    #l2 = [float(token.strip()) for token in line[2].split(',')]
    plt.rcParams.update({'font.size': 10})
    #plt.subplots(figsize=(50, 10))
    figure(figsize = (80, 10))
    plt.errorbar(l,l1)
    #plt.errorbar(l,l2)
    
    plt.show()

def concateWcWd(path):
    #path = 'C:/Project/EDU/files/2013/example/Topic/60/LG/6040i10-2/3/k14/c8d6/'
    
    with open(path + 'W1c.csv') as fw1c:
        w1c = list(csv.reader(fw1c, quoting=csv.QUOTE_NONNUMERIC))
    
    with open(path + 'W2c.csv') as fw2c:
        w2c = list(csv.reader(fw2c, quoting=csv.QUOTE_NONNUMERIC))
    
    with open(path + 'W1d.csv') as fw1d:
        w1d = list(csv.reader(fw1d, quoting=csv.QUOTE_NONNUMERIC))
        
    with open(path + 'W2d.csv') as fw2d:
        w2d = list(csv.reader(fw2d, quoting=csv.QUOTE_NONNUMERIC))
        
    w1cA = np.array(w1c)
    w2cA = np.array(w2c)
    w1dA = np.array(w1d)
    w2dA = np.array(w2d)
    
    w12cA = (w1cA + w2cA) /2
    
    w = np.concatenate((w12cA, w1dA, w2dA), axis = 1)
    
    np.savetxt(path + "W1cW2cW1dW2d.csv", w, delimiter=",")
    
    fw1c.close()
    fw2c.close()
    fw1d.close()
    fw2d.close()  

def spectral(path, n):
    #path = 'C:/Project/EDU/files/2013/example/Topic/60/LG/6040i10-2/1/k15/c10d5/'
    
    with open(path + 'W1cW2cW1dW2d.csv') as f:
        array2d = list(csv.reader(f, quoting=csv.QUOTE_NONNUMERIC))
        
    X = np.array(array2d)
    
    print (X.shape)
    
    ptrn = []
    with open('C:/Project/EDU/OLI_175318/pattern.txt') as patterns:
        for p in patterns:
            ptrn.append(p.strip())

    clustering = SpectralClustering(n_clusters=n, assign_labels="discretize", random_state=0).fit(X)
    
    pathc = path + "Spectral/"
    
    if (os.path.isdir(pathc) == False):
        os.mkdir(pathc)
        
    fw = open(pathc + "Spectral_" + str(n) + ".txt", "w")
    fwc = open(pathc + str(n) + ".txt", "w")
    #fwm = open(pathc + "silhouette_" + str(n) + ".txt", "w")
    
    #labels = clustering.labels_
    #metric_score = metrics.silhouette_score(X, labels, metric = 'euclidean')
    #print metric_score
    #fwm.write(str(metric_score))
        
    for l in range(len(clustering.labels_)):
        fw.write(ptrn[l] + "\t" + str(clustering.labels_[l]) + "\n")
        fwc.write(str(clustering.labels_[l]) + "\n")

    fw.close()
    fwc.close()
    #fwm.close()
    f.close()

def kmeans(path, n):
    #path = 'C:/Project/EDU/files/2013/example/Topic/60/LG/6040i10-2/1/k15/c10d5/'
    
    with open(path + 'W1cW2cW1dW2d.csv') as f:
    #with open(path + 'H1H2.csv') as f:
        array2d = list(csv.reader(f, quoting=csv.QUOTE_NONNUMERIC))
        
    X = np.array(array2d)
    
    print (X.shape)
    
    ptrn = []
    with open('C:/Project/EDU/OLI_175318/pattern.txt') as patterns:
        for p in patterns:
            ptrn.append(p.strip())

    clustering = KMeans(n_clusters=n, random_state=0).fit(X)
    
    pathc = path + "kmeans/"
    #pathc = path + "kmeans_H1H2/"
    
    if (os.path.isdir(pathc) == False):
        os.mkdir(pathc)
        
    fw = open(pathc + "kmeans_" + str(n) + ".txt", "w")
    #fwc = open(pathc + str(n) + ".txt", "w")
    #fwm = open(pathc + "silhouette_" + str(n) + ".txt", "w")
    
    #labels = clustering.labels_
    #metric_score = metrics.silhouette_score(X, labels, metric = 'euclidean')
    #print metric_score
    #fwm.write(str(metric_score))
        
    for l in range(len(clustering.labels_)):
        fw.write(ptrn[l] + "\t" + str(clustering.labels_[l]) + "\n")
        #fwc.write(str(clustering.labels_[l]) + "\n")

    fw.close()
    #fwc.close()
    #fwm.close()
    f.close()
    

def prepare_cluster(path, num_of_cluster):
    
    Ws = []
    
    record = [[] for i in range(3)]
    
    counter = 0
    
    with open(path + 'W1cW2cW1dW2d.csv') as wf:
        for line in wf:
            counter = counter +1
            Ws.append(line)
            arr = np.fromstring(line.strip(), dtype = float, sep = ',')
            record[2].append(arr)
            

    with open(path + 'Spectral/Spectral_' + num_of_cluster + '.txt') as clf:
        for line in clf:
            pattern, cl = line.split()
            record[0].append(pattern.strip())
            record[1].append(cl.strip())
            
    
    clusters = list(set(record[1]))
    
    l = len(clusters)
    
    r = [[] for i in range(l)]
    
    for i in range(len(record[0])):

        r[int(record[1][i])].append(record[2][i])
        
    list_of_matrices = []
    
    for i in range(l):
        list_of_matrices.append(np.vstack(r[i]))
    
    avg = []
    stv = []
    
    new_path = path + 'Spectral/' + num_of_cluster + '/'
    os.mkdir(new_path)
        
    for i in range(len(list_of_matrices)):
        raw = list_of_matrices[i]
        avg.append(np.mean(raw, axis=0))
        stv.append(np.std(raw, axis=0))
        
        #if (os.path.isdir(new_path) == False):
        np.savetxt(new_path + clusters[i] + '.csv', raw, delimiter=",")
    
    avgs = np.vstack(avg)
    stvs = np.vstack(stv)
    
    avgstv = np.vstack((avgs, stvs))
    
    np.savetxt(new_path + 'all.csv' , avgstv, delimiter=",")
    
    fw = open(new_path + '/cluster_order.txt', "w")
    for i in range(l):
        m, n = list_of_matrices[i].shape
        fw.write(clusters[i] + '\t' + str(m) + '\n')
    
    fw.close()
    
    
    '''
    for i in range(len(list_of_matrices)):
        avg = np.mean(temp, axis=0)
        std = np.std(temp, axis=0)
        avgstd = np.vstack((avg, std))
        temp = np.vstack((temp, avgstd))
        
        new_path = path + 'Spectral/' + num_of_cluster + '/'
        if (os.path.isdir(new_path) == False):
            os.mkdir(new_path)
        np.savetxt(new_path + clusters[i] + '.csv', temp, delimiter=",")
        
    '''
    
def W_to_plot(path, num_of_cluster):
    
    Ws = []
    
    record = [[] for i in range(3)]
    
    counter = 0
    
    with open(path + 'W1cW2cW1dW2d.csv') as wf:
        for line in wf:
            counter = counter +1
            Ws.append(line)
            arr = np.fromstring(line.strip(), dtype = float, sep = ',')
            record[2].append(arr)
            

    with open(path + 'Spectral/Spectral_' + num_of_cluster + '.txt') as clf:
        for line in clf:
            pattern, cl = line.split()
            record[0].append(pattern.strip())
            record[1].append(cl.strip())
            
    
    clusters = list(set(record[1]))
    
    l = len(clusters)
    
    r = [[] for i in range(l)]
    
    for i in range(len(record[0])):

        r[int(record[1][i])].append(record[2][i])
        
    list_of_matrices = []
    
    for i in range(l):
        list_of_matrices.append(np.vstack(r[i]))
    
    avg = []
    stv = []
    
    new_path = path + 'Spectral/' + num_of_cluster + '/'
    os.mkdir(new_path)
        
    for i in range(len(list_of_matrices)):
        raw = list_of_matrices[i]
        avg.append(np.mean(raw, axis=0))
        stv.append(np.std(raw, axis=0))
        
        np.savetxt(new_path + clusters[i] + '.csv', raw, delimiter=",")
    
    avgs = np.vstack(avg)
    stvs = np.vstack(stv)
    
    avgstv = np.vstack((avgs, stvs))
    
    np.savetxt(new_path + 'all.csv' , avgstv, delimiter=",")
    
    fw = open(new_path + '/cluster_order.txt', "w")
    
    plt.figure(figsize = (20,10))
    
    for i in range(l):
        m, n = list_of_matrices[i].shape
        ci = []
        index = []
        ix = 0
        for j in range(n):
            ci.append(1.96 * stvs[i][j] / math.sqrt(m))
            index.append(ix)
            ix = ix + 1
        
        plt.errorbar(index, 
                     stv[i], 
                     ci, 
                     capsize = 5,
                     #linestyle='None',
                     label = 'Cluster' + clusters[i],
                     capthick=1, linewidth=2, elinewidth=1)
        
        plt.legend()
        fw.write(clusters[i] + '\t' + str(m) + '\n')
    
    plt.savefig(new_path + 'clusters.pdf')
    
    
    fw.close()
    
    

if __name__ == "__main__":
    #s1 = 'ssS_'
    #s2 = '_ssss'
    #print (s1, s2)
    #print (lv.distance(s1, s2))
    #print levenshtein(s1, s2)
    #print mlv(s1, s2)
    
    #create_matrix()
    #normal_similarity()
    #test_lv()
    
    path = "C:/Project/EDU/files/2013/example/Topic/similarity/"
    #path = "C:/Project\EDU/OLI_175318/"
    
    #create_matrix(path)
    normal_s(path)
    
    
    #normal_p(path)
    
    
    #path = 'C:/Project/EDU/OLI_175318/lg/mp_con/0/k100/c50d50/'
    #path = 'C:/Project/EDU/files/2013/example/Topic/similarity/mp_con_k15_kc10/k15/c10d5/'
    
    path = 'C:/Project/EDU/files/2013/example/Topic/similarity/grid_mp_con/a0.9b0.9d0.5-k20-c7d13/1/k20/c7d13/'
    path = 'C:/Project/EDU/files/2013/example/Topic/similarity/grid_mp_con/a0.1b0.1d0.9-k20-c18d2/1/k20/c18d2/'
    path = 'C:/Project/EDU/files/2013/example/Topic/similarity/grid_mp_sep/a0.1b0.1d0.9-k17-c16d1/1/k17/c16d1/'
    
    '''
    concateWcWd(path)
    for i in range(2,7):
        spectral(path, i)
        kmeans(path, i)
    '''
    '''
    for i in range(1, 11):
        pathn = path + str(i) + "/"
        for j in range(5,21):
            merge_errors(pathn, j)
            take_avg_easy(pathn, j)
    '''
    
    
    Ms = ["X1", "X2", "S1", "S2"]
    '''
    for M in Ms:    
        for i in range(1, 11):
            pathn = path + str(i) + "/"
            to_plot(pathn, M)
    '''    
    
    '''
    for M in Ms:
        ave_error_ci(path, M)
    '''
    
    '''
    for M in Ms:
        print M
        plot_avg(path, M)
    '''
    #k1 = 5
    #k2 = 20
    #print_label(k1, k2)
    #number_of_cluster = '6'
    #prepare_cluster(path, number_of_cluster)
    #W_to_plot(path, number_of_cluster)