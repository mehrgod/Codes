# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 17:23:15 2020

@author: mirza
"""

import os
import csv
import math
import matplotlib
import numpy as np
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering

def create_matrix(path):
    
    p = []
    
    #with open(path + "pattern.txt") as file:
    with open(path + "patternsTranslateFilterTFIDF.txt") as file:
        for line in file:            
            p.append(line.split('\t')[0].strip())
    
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
    upper = 0.2
    
    m, n = B.shape
    
    print m, n
    
    l = np.ndarray.flatten(B)
    
    minl = float(np.amin(l))
    maxl = float(np.amax(l))
    
    l_norm = [ (upper - lower) * (x - minl) / (maxl - minl) + lower for x in l]
    
    nB = np.reshape(l_norm, (m, n))
    
    np.savetxt(path + "/nmp.csv", nB, delimiter=",")
    
    print nB

def W_to_plot(path, num_of_cluster, clustering):
    
    #clustering = 'kmeans'
    #clustering = 'Spectral'
    
    record = [[] for i in range(3)]
        
    with open(path + 'W1cW2cW1dW2d.csv') as wf:
        for line in wf:
            arr = np.fromstring(line.strip(), dtype = float, sep = ',')
            record[2].append(arr)
            

    #with open(path + 'Spectral/Spectral_' + num_of_cluster + '.txt') as clf:
    #with open(path + 'kmeans/kmeans_' + num_of_cluster + '.txt') as clf:
    with open(path + clustering + '/' + clustering + '_' + num_of_cluster + '.txt') as clf:
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
    
    #new_path = path + 'Spectral/' + num_of_cluster + '/'
    #new_path = path + 'kmeans/' + num_of_cluster + '/'
    new_path = path + clustering + '/' + num_of_cluster + '/'
    if (os.path.isdir(new_path) == False):
        os.mkdir(new_path)
        
    for i in range(len(list_of_matrices)):
        raw = list_of_matrices[i]
        avg_val = np.mean(raw, axis=0)
        stv_val = np.std(raw, axis=0)
        avg.append(avg_val)
        stv.append(stv_val)
        
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
                     avg[i], 
                     ci, 
                     capsize = 5,
                     #linestyle='None',
                     label = 'Cluster' + clusters[i],
                     capthick=1, linewidth=2, elinewidth=1)
        
        plt.legend()
        fw.write(clusters[i] + '\t' + str(m) + '\n')
    
    plt.savefig(new_path + 'clusters.pdf')
    plt.savefig(new_path + 'clusters.png')
    
    fw.close()

def W_to_plot_f(path, num_of_cluster, clustering):
    
    record = [[] for i in range(3)]
        
    with open(path + 'W1cW2cW1dW2d.csv') as wf:
        for line in wf:
            arr = np.fromstring(line.strip(), dtype = float, sep = ',')
            record[2].append(arr)
            
    with open(path + clustering + '_' + num_of_cluster + '.txt') as clf:
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
    
    #new_path = path + clustering + '/' + num_of_cluster + '/'
    new_path = path
    if (os.path.isdir(new_path) == False):
        os.mkdir(new_path)
        
    for i in range(len(list_of_matrices)):
        raw = list_of_matrices[i]
        avg_val = np.mean(raw, axis=0)
        stv_val = np.std(raw, axis=0)
        avg.append(avg_val)
        stv.append(stv_val)
        
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
                     avg[i], 
                     ci, 
                     capsize = 5,
                     #linestyle='None',
                     #label = 'Cluster' + clusters[i],
                     label = 'Cluster' + str(i),
                     capthick=1, linewidth=2, elinewidth=1)
        
        plt.legend()
        fw.write(clusters[i] + '\t' + str(m) + '\n')
    
    plt.savefig(new_path + 'clusters.pdf')
    plt.savefig(new_path + 'clusters.png')
    
    fw.close()

def W_to_heatmap(path, num_of_cluster, clustering, c, d):
        
    intensity = 0.2
    X = [intensity]
        
    record = [[] for i in range(3)]
    
    with open(path + 'W1cW2cW1dW2d.csv') as wf:
        for line in wf:
            arr = np.fromstring(line.strip(), dtype = float, sep = ',')
            B = np.concatenate((arr[:c], X, arr[c:c+d], X, arr[c+d:]))
            record[2].append(B)
            
    with open(path + clustering + '/' + clustering + '_' + num_of_cluster + '.txt') as clf:
        for line in clf:
            pattern, cl = line.split()
            record[0].append(pattern.strip())
            record[1].append(cl.strip())
            
    clusters = list(set(record[1]))
    
    l = len(clusters)
    
    y_axis_label = []
    r = [[] for i in range(l)]
    out = []
    
    for i in range(l):
        for j in range(len(record[0])):
            if record[1][j] == str(i):
                y_axis_label.append(record[0][j])
                r[int(record[1][j])].append(record[2][j])
                out.append(record[2][j])
        out.append(np.full((1, len(record[2][0])), intensity))
        y_axis_label.append('')
    
    HM = np.vstack(out)
    
    m, n = HM.shape
    
    np.savetxt(path + clustering + '/' + num_of_cluster + '/heatmap.csv', HM[:m-1], delimiter=",")
    fw = open(path + '/ptrn.txt', "w")
    for p in y_axis_label:
        fw.write(p + '\n')
    fw.close()
    
    sns.set()
    plt.figure(figsize = (10,20))

    sns.set(font_scale=1.1)
    cmap = sns.cm.rocket_r
    sns_plot = sns.heatmap(HM[:m-1], yticklabels=y_axis_label, cmap = cmap)
    
    sns_plot.figure.savefig(path + clustering + '/' + num_of_cluster + '/heatmap.pdf')
    sns_plot.figure.savefig(path + clustering + '/' + num_of_cluster + '/heatmap.png')
    
    plt.show()

def W_to_heatmap_f(path, num_of_cluster, clustering, c, d):
        
    intensity = 0.2
    X = [intensity]
        
    record = [[] for i in range(3)]
    
    with open(path + 'W1cW2cW1dW2d.csv') as wf:
        for line in wf:
            arr = np.fromstring(line.strip(), dtype = float, sep = ',')
            B = np.concatenate((arr[:c], X, arr[c:c+d], X, arr[c+d:]))
            record[2].append(B)
            
    with open(path + clustering + '_' + num_of_cluster + '.txt') as clf:
        for line in clf:
            pattern, cl = line.split()
            record[0].append(pattern.strip())
            record[1].append(cl.strip())
            
    clusters = list(set(record[1]))
    
    l = len(clusters)
    
    y_axis_label = []
    r = [[] for i in range(l)]
    out = []
    
    for i in range(l):
        for j in range(len(record[0])):
            if record[1][j] == str(i):
                y_axis_label.append(record[0][j])
                r[int(record[1][j])].append(record[2][j])
                out.append(record[2][j])
        out.append(np.full((1, len(record[2][0])), intensity))
        y_axis_label.append('')
    
    HM = np.vstack(out)
    
    m, n = HM.shape
    '''
    np.savetxt(path + clustering + '/' + num_of_cluster + '/heatmap.csv', HM[:m-1], delimiter=",")
    fw = open(path + '/ptrn.txt', "w")
    for p in y_axis_label:
        fw.write(p + '\n')
    fw.close()
    '''
    sns.set()
    plt.figure(figsize = (10,20))

    sns.set(font_scale=1.1)
    cmap = sns.cm.rocket_r
    sns_plot = sns.heatmap(HM[:m-1], yticklabels=y_axis_label, cmap = cmap)
    
    sns_plot.figure.savefig(path + '/heatmap.pdf')
    sns_plot.figure.savefig(path + '/heatmap.png')
    
    plt.show()

def W_to_heatmap_simple(path, num_of_cluster, clustering, c, d):
        
    intensity = 0.2
    X = [intensity]
        
    #record = [[] for i in range(3)]
    
    with open(path + 'W1cW2cW1dW2d.csv') as wf:
        W = list(csv.reader(wf, quoting=csv.QUOTE_NONNUMERIC))
    '''
    with open(path + 'W1cW2cW1dW2d.csv') as wf:
        for line in wf:
            arr = np.fromstring(line.strip(), dtype = float, sep = ',')
            B = np.concatenate((arr[:c], X, arr[c:c+d], X, arr[c+d:]))
            record[2].append(B)
    '''        
    
    y_axis_label = []
    
    with open(path + 'ptrn.txt') as ptrnfile:
        for line in ptrnfile:
            y_axis_label.append(line)
    
        
    '''
    with open(path + clustering + '_' + num_of_cluster + '.txt') as clf:
        for line in clf:
            pattern, cl = line.split()
            record[0].append(pattern.strip())
            record[1].append(cl.strip())
            
    clusters = list(set(record[1]))
    
    l = len(clusters)
    
    y_axis_label = []
    r = [[] for i in range(l)]
    out = []
    
    for i in range(l):
        for j in range(len(record[0])):
            if record[1][j] == str(i):
                y_axis_label.append(record[0][j])
                r[int(record[1][j])].append(record[2][j])
                out.append(record[2][j])
        out.append(np.full((1, len(record[2][0])), intensity))
        y_axis_label.append('')
    
    HM = np.vstack(out)
    
    m, n = HM.shape
    
    np.savetxt(path + clustering + '/' + num_of_cluster + '/heatmap.csv', HM[:m-1], delimiter=",")
    fw = open(path + '/ptrn.txt', "w")
    for p in y_axis_label:
        fw.write(p + '\n')
    fw.close()
    '''
    sns.set()
    plt.figure(figsize = (10,20))

    sns.set(font_scale=1.1)
    cmap = sns.cm.rocket_r
    #sns_plot = sns.heatmap(HM[:m-1], yticklabels=y_axis_label, cmap = cmap)
    sns_plot = sns.heatmap(W, yticklabels=y_axis_label, cmap = cmap)
    
    sns_plot.figure.savefig(path + '/heatmap_f.pdf')
    sns_plot.figure.savefig(path + '/heatmap_f.png')
    
    plt.show()

def concateWcWd(path):
    
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
    
    fw = open(path + "train-W.txt", "w")
    
    for row in w:
        out = ''
        for i in row:
            out = out + ',' + "{:.8f}".format(i)
        fw.write(out[1:] + '\n')
            
    
    fw1c.close()
    fw2c.close()
    fw1d.close()
    fw2d.close()
    fw.close()
    
def concateW(path):
    
    with open(path + 'W1.csv') as fw1:
        w1 = list(csv.reader(fw1, quoting=csv.QUOTE_NONNUMERIC))
    
    with open(path + 'W2.csv') as fw2:
        w2 = list(csv.reader(fw2, quoting=csv.QUOTE_NONNUMERIC))
    
        
    w1A = np.array(w1)
    w2A = np.array(w2)
    
    #w12cA = (w1cA + w2cA) /2
    
    w = np.concatenate((w1A, w2A), axis = 1)
    
    np.savetxt(path + "W1cW2cW1dW2d.csv", w, delimiter=",")
    
    fw = open(path + "train-W.txt", "w")
    
    for row in w:
        out = ''
        for i in row:
            out = out + ',' + "{:.8f}".format(i)
        fw.write(out[1:] + '\n')
            
    
    fw1.close()
    fw2.close()
    fw.close()

def spectral(path, ppath, n):
    
    with open(path + 'W1cW2cW1dW2d.csv') as f:
        array2d = list(csv.reader(f, quoting=csv.QUOTE_NONNUMERIC))
        
    X = np.array(array2d)
    
    print (X.shape)
    
    ptrn = []

    #with open('C:/Project/EDU/OLI_175318/update/step/sep/tfidf/pattern.txt') as patterns:
    with open(ppath + 'pattern.txt') as patterns:
        for p in patterns:
            ptrn.append(p.strip())

    clustering = SpectralClustering(n_clusters=n, assign_labels="discretize", random_state=0).fit(X)
    
    #sil = round(metrics.silhouette_score(X, clustering.labels_), 4)
    sil = metrics.silhouette_score(X, clustering.labels_)
    
    print sil
    
    pathc = path + "Spectral/"
    
    if (os.path.isdir(pathc) == False):
        os.mkdir(pathc)
        
    fw = open(pathc + "Spectral_" + str(n) + ".txt", "w")
    fwc = open(pathc + str(n) + ".txt", "w")
        
    for l in range(len(clustering.labels_)):
        fw.write(ptrn[l] + "\t" + str(clustering.labels_[l]) + "\n")
        fwc.write(str(clustering.labels_[l]) + "\n")

    fw.close()
    fwc.close()
    f.close()

def stat_attempts_histogram():
    path = "C:/Project/EDU/OLI_175318/update/step/sep/"
    
    attempt_length = []
    
    with open(path + 'Sequence.txt') as lines:
        for line in lines:
            l = str(line).replace("_", "").strip()
            #print len(l)
            attempt_length.append(len(l))
            
    plt.hist(attempt_length, bins = 100)
    #plt.title('Frequency of attempt numbers')
    plt.xlabel('Number of Attempts')
    plt.ylabel('Frequency')
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    #matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')
    #plt.rcParams.update({'font.size': 24})
    plt.savefig(path + 'histogram.pdf')
    plt.show()
            
def stat_hint_histogram():
    path = "C:/Project/EDU/OLI_175318/update/step/sep/"
    
    hints = []
    
    with open(path + 'Sequence.txt') as lines:
        for line in lines:
            counter1 = str(line).count('h')
            counter2 = str(line).count('H')
            counter = counter1 + counter2
            hints.append(counter)
            
    plt.hist(hints, bins = 'auto')
    #plt.title('Frequency of attempt numbers')
    plt.xlabel('Number of Hints')
    plt.ylabel('Frequency')
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    plt.savefig(path + 'histogram-hints.pdf')
    plt.show()
    
def stat_sf_histogram():
    path = "C:/Project/EDU/OLI_175318/update/step/sep/"
    
    hints = []
    
    with open(path + 'Sequence.txt') as lines:
        for line in lines:
            counter1 = str(line).count('f')
            counter2 = str(line).count('F')
            counter = counter1 + counter2
            hints.append(counter)
            
    plt.hist(hints, bins = 'auto')
    #plt.title('Frequency of attempt numbers')
    plt.xlabel('Number of Failure')
    plt.ylabel('Frequency')
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    plt.savefig(path + 'histogram-failure.pdf')
    plt.show()


if __name__ == "__main__":
    
    #path = "C:/Project/EDU/OLI_175318/update/step/sep/lg/k20c10d10/k20/c10d10/"
    #path = "C:/Project/EDU/OLI_175318/update/step/sep/tfidf/"
    #path = "C:/Project/EDU/OLI_175318/update/step/sep/tfidf/lg/k20c10d10/k20/c10d10/"
    path = "C:/Project/EDU/OLI_175318/update/step/sep/tfidf/lg/grid/a1.0b1.0d1.4-k10kc6-1000/k10/c6d4/"
    path = "C:/Project/EDU/OLI_175318/update/step/sep/tfidf/pre/grid/a1.4b0.5d1.4-k10kc6-1000/k10/c6d4/"
    path = "C:/Project/EDU/OLI_175318/update/step/sep/tfidf/avg/grid/a1.0b1.0d1.4-k10-kc7kd3-1000/k10/c7d3/"
    path = "C:/Project/EDU/OLI_175318/update/step/sep/tfidf/post/grid/a1.0b1.4d1.4-k10-c5d5-1000/k10/c5d5/"
    path = "C:/Project/EDU/OLI_175318/update/step/sep/train-test/0.3/lg/grid/a0.1b1.0d1.4-k18-c7d11/k18/c7d11/"
    #path = "C:/Project/EDU/OLI_175318/update/step/sep/train-test/0.4\lg/grid/a0.1b1.0d1.4-k20-c13d7/k20/c13d7/"
    #path = "C:/Project/EDU/OLI_175318/update/step/sep/train-test/0.5/lg/grid/a0.1b0.1d0.1-k4-c3d1/k4/c3d1/"
    #path = "C:/Project/EDU/OLI_175318/update/step/sep/train-test/0.5/lg/grid/a0.1b0.1d1.0-k14-c3d11/k14/c3d11/"
    #path = "C:/Project/EDU/OLI_175318/update/step/sep/train-test/0.4/lg/grid/a0.1b1.0d1.4-k8-c3d5/k8/c3d5/"
    path = "C:/Project/EDU/OLI_175318/update/step/sep/train-test/0.2/lg/grid/a0.5b1.0d1.4-k10-c3d7/k18/c7d11/"
    path = "C:/Project/EDU/OLI_175318/update/step/sep/train-test/0.1/lg/grid/a0.1b1.4d1.4-k12-c3d9/k12/c3d9/"
    path = "C:/Project/EDU/files/2013/example/Topic/60/fix/"
    
    #stat_attempts_histogram()
    #stat_hint_histogram()
    #stat_sf_histogram()
    
    #create_matrix(path)
    #normal_s(path)
    
    
    '''
    path = 'C:/Project/EDU/OLI_175318/update/step/sep/train-test/method1/'
    path = 'C:/Project/EDU/files/2013/example/Topic/60/predictionFilter2/'
    
    dir1 = ['1/','2/','3/','4/']
    dir2 = ['0.1/','0.2/','0.3/','0.4/','0.5/']
    for d1 in dir1:
        for d2 in dir2:
            pathn = path + d1 + d2
            print pathn
            create_matrix(pathn)
            normal_s(pathn)
    '''
    
    
    '''
    path = "C:/Project/EDU/OLI_175318/update/step/sep/train-test/method1/4/0.5/lg/grid/a0.1b0.1d1.4-k10-c5d5/k10/c5d5/"
    
    '''
    path = 'C:/Project/EDU/OLI_175318/update/step/sep/train-test/method1/1/0.1/MuParamTest/k20/c10d10/'
    path = 'C:/Project/EDU/OLI_175318/update/step/sep/train-test/method1/1/0.1/MuUpdate/k20/c10d10/'
    path = 'C:/Project/EDU/files/2013/example/Topic/60/fix/lg/NMF/k20/c12d8/'
    path = 'C:/Project/EDU/OLI_175318/update/step/sep/lg/DNMF/k26/c19d7/'
    #concateWcWd(path)
    #concateW(path)
    path = 'C:/Project/EDU/OLI_175318/update/step/sep/fix/lg/DNMF/k12/c3d9/'
    ppath = 'C:/Project/EDU/OLI_175318/update/step/sep/fix/'
    
    path = "C:/Project/EDU/Statistics-ds1139/fix/post/DNMF/k13/c4d9/"
    ppath = "C:/Project/EDU/Statistics-ds1139/fix/"
    #concateWcWd(path)
    
    #for i in range(2,7):
    #for i in range(6,7):
    #    spectral(path, ppath, i)
    
    
    #clustering = 'kmeans'
    clustering = ['Spectral']
    
    #clustering = ['Spectral', 'kmeans']
    
    #path = 'C:/Project/EDU/OLI_175318/update/step/sep/train-test/method1/'
    
    
    c = 4
    d = 9
    '''
    #cluster = ['2','3','4','5','6']
    cluster = ['6']
    for number_of_cluster in cluster:
        for cluster in clustering:
            print cluster
            W_to_plot(path, number_of_cluster, cluster)
            W_to_heatmap(path, number_of_cluster, cluster, c, d)
    '''
    
    #path = 'C:/Project/EDU/OLI_175318/update/step/sep/fix/lg/DNMF/k12/c3d9/Spectral/6/'
    path = "C:/Project/EDU/Statistics-ds1139/fix/post/DNMF/k13/c4d9/Spectral/6/"
    
    cluster = ['6']
    for number_of_cluster in cluster:
        for cluster in clustering:
            print cluster
            #W_to_plot_f(path, number_of_cluster, cluster)
            W_to_heatmap_simple(path, number_of_cluster, cluster, c, d)
    
    
    