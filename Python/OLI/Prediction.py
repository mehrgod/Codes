# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 17:45:29 2020

@author: mirza
"""
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KN
from sklearn import preprocessing
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
import csv
import os



def split_sk(path, per, n):
    with open(path + "LabelSequence.txt", "rb") as f:
        data = f.read().split('\n')
        data = np.array(data)  #convert array to numpy type array
        
    x_train ,x_test = train_test_split(data,test_size=per)
        
    dir = path + str(n) + '/' + str(per) + '/'
    
    if not os.path.exists(dir):
        os.mkdir(dir)
    
    #np.savetxt(dir + 'train.txt', x_train, fmt="%s")
    #np.savetxt(dir + 'test-100.txt', x_test, fmt="%s")
    
    fwtr = open(dir + 'train.txt', 'w')
    fwte = open(dir + 'test-100.txt', 'w')
    
    
    for x in x_train:
        if len(x) > 0:
            fwtr.write(x.strip() + '\n')
            #fwtr.write(x.strip())
            
    for x in x_test:
        if len(x) > 0:
            fwte.write(x.strip() + '\n')
            #fwte.write(x.strip())
            
    fwtr.close()
    fwte.close()
        
    
def split_perf(path, filename):
    
    dic = {}
    
    with open("C:/Project/EDU/OLI_175318/update/step/sep/train-test/perfAll.txt") as file:
        for line in file:
            id, pre, post, lg, avg = line.split('\t')
            dic[id] = lg
            
    
    fwl = open(path + filename + '-lo.txt', "w")
    fwh = open(path + filename + '-hi.txt', "w")
    
    with open(path + filename + '.txt') as file:
        for line in file:
            id, seq = line.split('\t')
            if (dic[id] == 'l'):
                fwl.write(line)
            elif (dic[id] == 'h'):
                fwh.write(line)
                
    fwl.close
    fwh.close


def split_test(path, per, n):
    
    fw1 = open(path + str(n) + '/' + str(per) + '/test-10.txt', "w")
    fw2 = open(path + str(n) + '/' + str(per) + '/test-20.txt', "w")
    fw3 = open(path + str(n) + '/' + str(per) + '/test-30.txt', "w")
    fw4 = open(path + str(n) + '/' + str(per) + '/test-40.txt', "w")
    fw5 = open(path + str(n) + '/' + str(per) + '/test-50.txt', "w")
    fw6 = open(path + str(n) + '/' + str(per) + '/test-60.txt', "w")
    fw7 = open(path + str(n) + '/' + str(per) + '/test-70.txt', "w")
    fw8 = open(path + str(n) + '/' + str(per) + '/test-80.txt', "w")
    fw9 = open(path + str(n) + '/' + str(per) + '/test-90.txt', "w")
    
    with open(path + str(n) + '/' + str(per) + '/test-100.txt') as file:
        for line in file:
            #if len(line) == 0:
            #    continue
            print line
            id, seq = line.split('\t')
            x = seq.split('_')
            l = len(x)
            a = np.split(x, [int(l*0.1), int(l*0.2),int(l*0.3), int(l*0.4),int(l*0.5), int(l*0.6),int(l*0.7), int(l*0.8), int(l*0.9) ])
            
            out = id + '\t' + array_to_seq(a[0]) + '_\n'
            if (len(out) > 50):
                fw1.write(out)
            
            out = id + '\t' + array_to_seq(a[0]) + '_' + array_to_seq(a[1]) + '_\n'
            if (len(out) > 50):
                fw2.write(out)
            
            out = id + '\t' + array_to_seq(a[0]) + '_' + array_to_seq(a[1]) + '_' + array_to_seq(a[2]) + '_\n'
            if (len(out) > 50):
                fw3.write(out)
            
            out = id + '\t' + array_to_seq(a[0]) + '_' + array_to_seq(a[1]) + '_' + array_to_seq(a[2]) + '_' + array_to_seq(a[3]) + '_\n'
            if (len(out) > 50):
                fw4.write(out)
            
            out = id + '\t' + array_to_seq(a[0]) + '_' + array_to_seq(a[1]) + '_' + array_to_seq(a[2]) + '_' + array_to_seq(a[3]) + '_' + array_to_seq(a[4]) + '_\n'
            if (len(out) > 50):
                fw5.write(out)
            
            out = id + '\t' + array_to_seq(a[0]) + '_' + array_to_seq(a[1]) + '_' + array_to_seq(a[2]) + '_' + array_to_seq(a[3]) + '_' + array_to_seq(a[4]) + '_' + array_to_seq(a[5]) + '_\n'
            if (len(out) > 50):
                fw6.write(out)
            
            out = id + '\t' + array_to_seq(a[0]) + '_' + array_to_seq(a[1]) + '_' + array_to_seq(a[2]) + '_' + array_to_seq(a[3]) + '_' + array_to_seq(a[4]) + '_' + array_to_seq(a[5]) + '_' + array_to_seq(a[6]) + '_\n'
            if (len(out) > 50):
                fw7.write(out)
            
            out = id + '\t' + array_to_seq(a[0]) + '_' + array_to_seq(a[1]) + '_' + array_to_seq(a[2]) + '_' + array_to_seq(a[3]) + '_' + array_to_seq(a[4]) + '_' + array_to_seq(a[5]) + '_' + array_to_seq(a[6]) +'_' + array_to_seq(a[7]) + '_\n'
            if (len(out) > 50):
                fw8.write(out)
            
            out = id + '\t' + array_to_seq(a[0]) + '_' + array_to_seq(a[1]) + '_' + array_to_seq(a[2]) + '_' + array_to_seq(a[3]) + '_' + array_to_seq(a[4]) + '_' + array_to_seq(a[5]) + '_' + array_to_seq(a[6]) +'_' + array_to_seq(a[7]) +'_' + array_to_seq(a[8]) + '_\n'
            if (len(out) > 50):
                fw9.write(out)

            
    fw1.close()
    fw2.close()
    fw3.close()
    fw4.close()
    fw5.close()
    fw6.close()
    fw7.close()
    fw8.close()
    fw9.close()
    
        
def array_to_seq(arr):
    output = ''
    for a in arr:
        output = output + '_' + a
    return output[1:]

def array_to_str(arr):
    output = ''
    for a in arr:
        output = output + ',' + str(a)
    return output[1:]

def normal_test(path, n):
    
    with open(path + 'SequenceVector-' + str(n) + '.txt') as file:
        array2d = [[int(digit) for digit in line.split('\t')[1].split(',')] for line in file]
    
    X = np.array(array2d)
    
    X_normalized = preprocessing.normalize(X, norm='l1')
    
    np.savetxt(path + 'VectorNormal1-' + str(n) + '.csv', X_normalized, delimiter=",")
    
    file.close()

def normal(path):
    
    with open(path + 'SequenceVector.txt') as file:
        array2d = [[int(digit) for digit in line.split('\t')[1].split(',')] for line in file]
    
    X = np.array(array2d)
    
    X_normalized = preprocessing.normalize(X, norm='l1')
    
    np.savetxt(path + 'VectorNormal1.csv', X_normalized, delimiter=",")
    
    file.close()

    
def predict(path, n , c , d):
    with open(path + 'train-W.txt') as file:
        array = [[float(digit) for digit in line.split(',')] for line in file]
    train = np.array(array)
    
    dic = {}
    
    with open(path + "perfAll.txt") as file:
        for line in file:
            id, pre, post, lg, avg = line.split('\t')
            dic[id] = lg
    
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    
    file_name = "VectorNormal1-" + str(n)
    
    with open(path + file_name + ".txt") as file:
        for line in file:
            id = line.split('\t')[0]
            array = [float(digit) for digit in line.split('\t')[1].split(',')]
            test = np.array(array)
            res = np.dot(test, train)
            perf = dic.get(id)
            lo = 0
            hi = 0
            #for i in range(3, 14):
            for i in range(c, c+d):
                lo = lo + res[i]
            #for i in range(14, 25):
            for i in range(c+d, c+d+d):
                hi = hi + res[i]
            if perf == 'h':
                if hi >= lo:
                    TP = TP + 1
                else:
                    FP = FP + 1
            if perf == 'l':
                if lo >= hi:
                    TN = TN + 1
                else:
                    FN = FN + 1
            #print array_to_str(res)
    print "TP: ", TP
    print "TN: ", TN
    print "FP: ", FP
    print "FN: ", FN
    
    acc = float(TP + TN)/(TP + TN + FP + FN)
    print "Acc: ", acc
    print str(TP + TN + FP + FN)

    fw = open(path + file_name + '_res.txt', "w")
    '''
    fw.write("TP: " + str(TP) + "\n")
    fw.write("TN: " + str(TN) + "\n")
    fw.write("FP: " + str(FP) + "\n")
    fw.write("FN: " + str(FN) + "\n")
    fw.write("Acc: " + str(acc))
    '''
    fw.write(str(TP) + "\n")
    fw.write(str(TN) + "\n")
    fw.write(str(FP) + "\n")
    fw.write(str(FN) + "\n")
    fw.write(str(acc))

        
    fw.close()
    

def count_performance(path ,n):
    dic = {}
    
    with open(path + "perfAll.txt") as file:
        for line in file:
            id, pre, post, lg, avg = line.split('\t')
            dic[id] = lg.strip()
            #print lg
    
    l = 0
    m = 0
    h = 0    
    with open(path + "test-" + str(n) + ".txt") as file:
        for line in file:
            id, seq = line.split('\t')
            #print dic[id]
            if dic[id] == 'l':
                l = l + 1
            if dic[id] == 'm':
                m = m + 1
            if dic[id] == 'h':
                h = h + 1
           
    #print 'low: ', l
    #print 'med: ', m
    #print 'hi:' , h
    print l
    print m
    print h
    fw = open(path + "stat-" + str(n) + ".txt", "w")
    #fw.write("low\t" + str(l) + '\nmed\t' + str(m) + "\nhi\t" + str(h))
    fw.write(str(l) + '\n' + str(m) + "\n" + str(h))
    fw.close()

def knn(path, n):
    with open(path + 'lg/low.txt') as file:
        array = [[float(digit) for digit in line.split(',')] for line in file]
    L = np.array(array)
    
    with open(path + 'lg/hi.txt') as file:
        array = [[float(digit) for digit in line.split(',')] for line in file]
    H = np.array(array)
    
    lx, ly = L.shape
    hx, hy = H.shape
    
    y_train = []
    for i in range (1, lx + 1):
        y_train.append(1)
    for i in range (lx + 1, lx + hx + 1):
        y_train.append(0)
    
    X_train = np.concatenate((L, H), axis = 0)
    
    neigh = KN(n_neighbors = 3)
    neigh.fit(X_train, y_train)
    
    print X_train.shape

    with open(path + 'KNN/test-' + str(n) + '-features.txt') as file:
        array = [[float(digit) for digit in line.split(',')] for line in file]
    X_test = np.array(array)
    
    label = []
    with open(path + 'KNN/test-' + str(n) + '-labels.txt') as file:
        for line in file:
            label.append(int(line.strip()))
    
    y_label = np.array(label)
    #l = len(label)
    
    scores = {}
    scores_list = []
    
    k_range = range(1, 21)
    
    mxm = 0
    #max_acc = 0
    
    for k in k_range:
        knn = KN(n_neighbors = k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        score = metrics.accuracy_score(y_label, y_pred)
        scores[k] = score
        scores_list.append(metrics.accuracy_score(y_label, y_pred))
        if score >= mxm :
            mxm = score
            print y_label
            print '----'
            print y_pred
            tn, fp, fn, tp = metrics.confusion_matrix(y_label, y_pred).ravel()
            
            
    fw = open(path + 'KNN/knn-' + str(n) + '.txt' ,'w')
    fw.write(str(tp) + '\n' + str(tn) + '\n' + str(fp) + '\n' + str(fn) + '\n' + str(mxm))
    
    plt.plot(k_range, scores_list)
    #plt.savefig(path + 'KNN/acc-' + str(n) + '.pdf')
    plt.show()
    
    fw.close()
    
    
def find_perf_test(path, n):
    
    dic = {}
            
    with open(path[:-6] + "perfAll.txt") as file:
        for line in file:
            #id, pre, post, lg, avg = line.split('\t')
            id, pre, post, lg = line.split('\t')
            dic[id] = lg.strip()
    
    dir = path + 'KNN/'
    
    if not os.path.exists(dir):
        os.mkdir(dir)
    
    fwf = open(path + 'KNN/test-' + str(n) + '-features.txt', "w")
    fwl = open(path + 'KNN/test-' + str(n) + '-labels.txt', "w")
    
    with open(path + 'VectorNormal1-' + str(n) + '.txt') as file:
        for line in file:
            id, seq = line.split('\t')
            if (id not in dic):
                continue
            if (dic[id] == 'h'):
                print 'h=h'
                fwf.write(seq)
                fwl.write('0\n')
            if (dic[id] == 'l'):
                fwf.write(seq)
                fwl.write('1\n')
            
    fwf.close()
    fwl.close()
    

def H_avg_var(pathin, pathout, filename):
    #dir = path + 'KNN-latent'
    dir = pathout
    if not os.path.exists(dir):
        os.mkdir(dir)
    
    with open(pathin + filename + '.csv') as fw1c:
        harr = list(csv.reader(fw1c, quoting=csv.QUOTE_NONNUMERIC))
    H = np.array(harr)
    
    avg = np.mean(H, axis = 0)
    var = np.var(H, axis = 0)
    
    #np.savetxt(path + 'KNN-latent/' + filename + '-avg.txt', avg, delimiter = ',', newline = " ")
    #np.savetxt(path + 'KNN-latent/' + filename + '-var.txt', var, delimiter = ',', newline = " ")
    np.savetxt(pathout + filename + '-avg.txt', avg, delimiter = ',', newline = " ")
    np.savetxt(pathout + filename + '-var.txt', var, delimiter = ',', newline = " ")
        
    
def predict_knn_latent(path, n):
    
    
    with open(path + 'W1.csv') as fw1:
        array = list(csv.reader(fw1, quoting=csv.QUOTE_NONNUMERIC))
    W1 = np.array(array)
    
    with open(path + 'W2.csv') as fw2:
        array = list(csv.reader(fw2, quoting=csv.QUOTE_NONNUMERIC))
    W2 = np.array(array)
    
    dic = {}
    
    with open(path[:-6] + "perfAll.txt") as file:
        for line in file:
            id, pre, post, lg, avg = line.split('\t')
            dic[id] = lg
    
    label = []
    predict = []


    with open(path + 'KNN-latent/H1-avg.txt') as file:
        for line in file:
            seq = line.split()
            float_seq = [float(i) for i in seq]
    H1 = np.array(float_seq)
    
    with open(path + 'KNN-latent/H2-avg.txt') as file:
        for line in file:
            seq = line.split()
            float_seq = [float(i) for i in seq]
    H2 = np.array(float_seq)
    
    file_name = "VectorNormal1-" + str(n)
    
    with open(path + file_name + ".txt") as file:
        for line in file:
            id = line.split('\t')[0]
            perf = dic.get(id)
            
            if perf == 'h':
                lbl = 0
            elif perf == 'l':
                lbl = 1
            else:
                continue
            
            label.append(lbl)
            
            array = [float(digit) for digit in line.split('\t')[1].split(',')]
            test = np.array(array)
            
            #resW1 = np.dot(test, W1)
            #resW2 = np.dot(test, W2)
            
            resW1 = np.dot(np.linalg.pinv(W1, rcond = 0.0), test)
            resW2 = np.dot(np.linalg.pinv(W2, rcond = 0.0), test)
            '''
            print 'resW1 ', resW1.shape
            print 'H1 ', H1.shape
            '''
            
            dis1 = distance_euc(resW1, H1)
            #print dis1
            dis2 = distance_euc(resW2, H2)
            #print dis2
            
            
            if (dis1 > dis2):
                pred = 0
            else:
                pred = 1
                
            predict.append(pred)
            #print pred
            
            
    pred = np.array(predict)
    labl = np.array(label)
    
    score = metrics.accuracy_score(labl, pred)
    tn, fp, fn, tp = metrics.confusion_matrix(labl, pred).ravel()
        
    dir = path + 'KNN-latent-dis/5/'
    if not os.path.exists(dir):
        os.mkdir(dir)
        
    fw = open(path + 'KNN-latent-dis/5/con_' + str(n) + '_res.txt', "w")


    fw.write(str(tp) + "\n")
    fw.write(str(tn) + "\n")
    fw.write(str(fp) + "\n")
    fw.write(str(fn) + "\n")
    fw.write(str(score))

        
    fw.close()

def predict_knn_latent_test(path, pathn, n):
    
    with open(path + 'W1.csv') as fw1:
        array = list(csv.reader(fw1, quoting=csv.QUOTE_NONNUMERIC))
    W1 = np.array(array)
    
    with open(path + 'W2.csv') as fw2:
        array = list(csv.reader(fw2, quoting=csv.QUOTE_NONNUMERIC))
    W2 = np.array(array)
    
    dic = {}
    
    #with open(path[:-6] + "perfAll.txt") as file:
    with open(path + "perfAll.txt") as file:
        for line in file:
            id, pre, post, lg, avg = line.split('\t')
            #id, pre, post, lg = line.split('\t')
            dic[id] = lg
    
    label = []
    predict = []


    with open(pathn + 'H1-avg.txt') as file:
        for line in file:
            seq = line.split()
            float_seq = [float(i) for i in seq]
    H1 = np.array(float_seq)
    
    with open(pathn + 'H2-avg.txt') as file:
        for line in file:
            seq = line.split()
            float_seq = [float(i) for i in seq]
    H2 = np.array(float_seq)
    
    with open(pathn + 'H1-var.txt') as file:
        for line in file:
            seq = line.split()
            float_seq = [float(i) for i in seq]
    H1v = np.array(float_seq)
    
    with open(pathn + 'H2-var.txt') as file:
        for line in file:
            seq = line.split()
            float_seq = [float(i) for i in seq]
    H2v = np.array(float_seq)
    
    file_name = "VectorNormal1-" + str(n)
    
    with open(path + file_name + ".txt") as file:
        for line in file:
            id = line.split('\t')[0]
            perf = dic.get(id)
            
            if perf == 'h':
                lbl = 0
            elif perf == 'l':
                lbl = 1
            else:
                continue
            
            label.append(lbl)
            
            array = [float(digit) for digit in line.split('\t')[1].split(',')]
            test = np.array(array)
            
            #resW1 = np.dot(test, W1)
            #resW2 = np.dot(test, W2)
            
            resW1 = np.dot(np.linalg.pinv(W1, rcond = 0.0), test)
            resW2 = np.dot(np.linalg.pinv(W2, rcond = 0.0), test)
            '''
            print 'resW1 ', resW1.shape
            print 'H1 ', H1.shape
            '''
            
            temp = np.subtract(resW1, H1)
            dis1 = 0
            for i in range(0, len(temp)):
                dis1 = dis1 + np.square(temp[i] / H1v[i])
            temp = np.subtract(resW2, H2)
            
            dis2 = 0
            for i in range(0, len(temp)):
                dis2 = dis2 + np.square(temp[i] / H2v[i])
            
            #dis1 = distance_euc(resW1, H1)
            #print dis1
            #dis2 = distance_euc(resW2, H2)
            #print dis2
            
            
            if (dis1 > dis2):
                pred = 0
            else:
                pred = 1
                
            predict.append(pred)
            #print pred
            
            
    pred = np.array(predict)
    labl = np.array(label)
    
    score = metrics.accuracy_score(labl, pred)
    tn, fp, fn, tp = metrics.confusion_matrix(labl, pred).ravel()
        
    #dir = pathn + 'KNN-latent-dis/8/'
    #if not os.path.exists(dir):
    #    os.mkdir(dir)
        
    fw = open(pathn + 'con_' + str(n) + '_res.txt', "w")


    fw.write(str(tp) + "\n")
    fw.write(str(tn) + "\n")
    fw.write(str(fp) + "\n")
    fw.write(str(fn) + "\n")
    fw.write(str(score))

        
    fw.close()


def predict_knn_latent_var(path, n):
    
    
    with open(path + 'W1.csv') as fw1:
        array = list(csv.reader(fw1, quoting=csv.QUOTE_NONNUMERIC))
    W1 = np.array(array)
    
    with open(path + 'W2.csv') as fw2:
        array = list(csv.reader(fw2, quoting=csv.QUOTE_NONNUMERIC))
    W2 = np.array(array)
    
    dic = {}
    
    with open(path[:-6] + "perfAll.txt") as file:
        for line in file:
            id, pre, post, lg, avg = line.split('\t')
            dic[id] = lg
    
    label = []
    predict = []


    with open(path + 'KNN-latent-dis/H1-avg.txt') as file:
        for line in file:
            seq = line.split()
            float_seq = [float(i) for i in seq]
    H1 = np.array(float_seq)
    
    with open(path + 'KNN-latent-dis/H2-avg.txt') as file:
        for line in file:
            seq = line.split()
            float_seq = [float(i) for i in seq]
    H2 = np.array(float_seq)
    
    with open(path + 'KNN-latent-dis/H1-var.txt') as file:
        for line in file:
            seq = line.split()
            float_seq = [float(i) for i in seq]
    V1 = np.array(float_seq)
    sv1 = np.sum(V1)
    
    with open(path + 'KNN-latent-dis/H2-var.txt') as file:
        for line in file:
            seq = line.split()
            float_seq = [float(i) for i in seq]
    V2 = np.array(float_seq)
    sv2 = np.sum(V2)
    
    
    fwd = open(path + 'KNN-latent-dis/dis_' + str(n) + '.txt', "w")
    
    file_name = "VectorNormal1-" + str(n)
    
    with open(path + file_name + ".txt") as file:
        for line in file:
            id = line.split('\t')[0]
            perf = dic.get(id)
            
            if perf == 'h':
                lbl = 0
            elif perf == 'l':
                lbl = 1
            else:
                continue
            
            label.append(lbl)
            
            array = [float(digit) for digit in line.split('\t')[1].split(',')]
            test = np.array(array)
            resW1 = np.dot(test, W1)
            resW2 = np.dot(test, W2)
            
            dis1 = distance_cos(resW1, H1)
            
            disv1 = dis1/sv1
            
            dis2 = distance_cos(resW2, H2)
            
            disv2 = dis1/sv2
            
            #print dis1, dis2
            
            #pred = np.sign(dis1 - dis2)
            
            fwd.write(str(dis1) + '\t' + str(dis2) + '\t' + str(disv1) + '\t' + str(disv2) + '\n')
            
            
            if (dis1 > dis2):
                pred = 1
            else:
                pred = 0
            
            
            predict.append(pred)
            #print pred
            #print lbl
            
            
    pred = np.array(predict)
    labl = np.array(label)
    
    score = metrics.accuracy_score(labl, pred)
    tn, fp, fn, tp = metrics.confusion_matrix(labl, pred).ravel()
        
    
    fw = open(path + 'KNN-latent-dis/con_' + str(n) + '_res.txt', "w")
    '''
    fw.write("TP: " + str(TP) + "\n")
    fw.write("TN: " + str(TN) + "\n")
    fw.write("FP: " + str(FP) + "\n")
    fw.write("FN: " + str(FN) + "\n")
    fw.write("Acc: " + str(acc))
    '''
    fw.write(str(tp) + "\n")
    fw.write(str(tn) + "\n")
    fw.write(str(fp) + "\n")
    fw.write(str(fn) + "\n")
    fw.write(str(score))

        
    fw.close()
    fwd.close()

def predict_knn_latent_var_1(path, n):
    
    distance_method = 'cosine'
    distance_method = 'euclidean'
    
    with open(path + 'W1.csv') as fw1:
        array = list(csv.reader(fw1, quoting=csv.QUOTE_NONNUMERIC))
    W1 = np.array(array)
    
    with open(path + 'W2.csv') as fw2:
        array = list(csv.reader(fw2, quoting=csv.QUOTE_NONNUMERIC))
    W2 = np.array(array)
    
    dic = {}
    
    with open(path[:-6] + "perfAll.txt") as file:
        for line in file:
            id, pre, post, lg, avg = line.split('\t')
            dic[id] = lg
    
    label = []
    predict = []


    with open(path + 'KNN-latent-dis/H1-avg.txt') as file:
        for line in file:
            seq = line.split()
            float_seq = [float(i) for i in seq]
    H1 = np.array(float_seq)
    
    with open(path + 'KNN-latent-dis/H2-avg.txt') as file:
        for line in file:
            seq = line.split()
            float_seq = [float(i) for i in seq]
    H2 = np.array(float_seq)
    
    fwd = open(path + 'KNN-latent-dis/dis_' + str(n) + '.txt', "w")
    
    file_name = "VectorNormal1-" + str(n)
    
    with open(path + file_name + ".txt") as file:
        for line in file:
            id = line.split('\t')[0]
            perf = dic.get(id)
            
            if perf == 'h':
                lbl = 0
            elif perf == 'l':
                lbl = 1
            else:
                continue
            
            label.append(lbl)
            
            array = [float(digit) for digit in line.split('\t')[1].split(',')]
            test = np.array(array)
            
            inv1 = np.dot(np.linalg.pinv(W1, rcond = 0.0), test)
            inv2 = np.dot(np.linalg.pinv(W2, rcond = 0.0), test)
            
            if distance_method.startswith('c'):
                dis1 = distance_cos(inv1, H1)
                dis2 = distance_cos(inv2, H2)
            elif distance_method.startswith('e'):
                dis1 = distance_euc(inv1, H1)
                dis2 = distance_euc(inv2, H2)
            else:
                print 'Invalid Distance Measure'
                break
            
            fwd.write(str(dis1) + '\t' + str(dis2) + '\n')
            
            if (dis1 < dis2):
                pred = 1
            else:
                pred = 0
            
            predict.append(pred)
            
    pred = np.array(predict)
    labl = np.array(label)
    
    dir = path + 'KNN-latent-dis/1/'
    if not os.path.exists(dir):
        os.mkdir(dir)
        
    fw = open(path + 'KNN-latent-dis/1/pred_' + str(n) + '_1_' + distance_method + '.txt', "w")
    l = len(labl)
    for i in range(0, l):
        fw.write(str(pred[i]) + '\t' + str(labl[i]) + '\n')
        
    score = metrics.accuracy_score(labl, pred)
    tn, fp, fn, tp = metrics.confusion_matrix(labl, pred).ravel()
    
    fw = open(path + 'KNN-latent-dis/1/conf_' + str(n) + '_1_' + distance_method + '.txt', "w")

    fw.write(str(tp) + "\n")
    fw.write(str(tn) + "\n")
    fw.write(str(fp) + "\n")
    fw.write(str(fn) + "\n")
    fw.write(str(score))
        
    fw.close()
    fwd.close()

def predict_knn_latent_var_sum_1(path, n):
    
    distance_method = 'cosine'
    distance_method = 'euclidean'
    
    with open(path + 'W1.csv') as fw1:
        array = list(csv.reader(fw1, quoting=csv.QUOTE_NONNUMERIC))
    W1 = np.array(array)
    
    with open(path + 'W2.csv') as fw2:
        array = list(csv.reader(fw2, quoting=csv.QUOTE_NONNUMERIC))
    W2 = np.array(array)
    
    with open(path + 'H1.csv') as fh1:
        array = list(csv.reader(fh1, quoting=csv.QUOTE_NONNUMERIC))
    H1 = np.array(array)
    
    with open(path + 'H2.csv') as fh2:
        array = list(csv.reader(fh2, quoting=csv.QUOTE_NONNUMERIC))
    H2 = np.array(array)
    
    dic = {}
    
    with open(path[:-6] + "perfAll.txt") as file:
        for line in file:
            id, pre, post, lg, avg = line.split('\t')
            dic[id] = lg
    
    label = []
    predict = []
    
    lst_dis1 = []
    lst_dis2 = []
    
    '''
   with open(path + 'KNN-latent-dis/H1-avg.txt') as file:
        for line in file:
            seq = line.split()
            float_seq = [float(i) for i in seq]
    H1 = np.array(float_seq)
    
    with open(path + 'KNN-latent-dis/H2-avg.txt') as file:
        for line in file:
            seq = line.split()
            float_seq = [float(i) for i in seq]
    H2 = np.array(float_seq)
    '''
    
    fwd = open(path + 'KNN-latent-dis/dis_' + str(n) + '.txt', "w")
    
    file_name = "VectorNormal1-" + str(n)
    
    with open(path + file_name + ".txt") as file:
        for line in file:
            id = line.split('\t')[0]
            perf = dic.get(id)
            
            if perf == 'h':
                lbl = 0
            elif perf == 'l':
                lbl = 1
            else:
                continue
            
            label.append(lbl)
            
            array = [float(digit) for digit in line.split('\t')[1].split(',')]
            test = np.array(array)
            
            inv1 = np.dot(np.linalg.pinv(W1, rcond = 0.0), test)
            inv2 = np.dot(np.linalg.pinv(W2, rcond = 0.0), test)
            
            dis1 = 0
            dis2 = 0
            
            for row in H1:
                dis1 = dis1 + distance_euc(inv1, row)
            for row in H2:
                dis2 = dis2 + distance_euc(inv2, row)
                
            lst_dis1.append(dis1)
            lst_dis2.append(dis2)
            
            '''
            if distance_method.startswith('c'):
                dis1 = distance_cos(inv1, H1)
                dis2 = distance_cos(inv2, H2)
            elif distance_method.startswith('e'):
                dis1 = distance_euc(inv1, H1)
                dis2 = distance_euc(inv2, H2)
            else:
                print 'Invalid Distance Measure'
                break
            '''
            
            fwd.write(str(dis1) + '\t' + str(dis2) + '\n')
            
            if (dis1 < dis2):
                pred = 1
            else:
                pred = 0
            
            predict.append(pred)
            
    pred = np.array(predict)
    labl = np.array(label)
    
    dir = path + 'KNN-latent-dis/1-sum/'
    if not os.path.exists(dir):
        os.mkdir(dir)
        
    fw = open(path + 'KNN-latent-dis/1-sum/pred_' + str(n) + '_1_' + distance_method + '.txt', "w")
    l = len(labl)
    for i in range(0, l):
        fw.write(str(pred[i]) + '\t' + str(labl[i]) + '\n')
        
    score = metrics.accuracy_score(labl, pred)
    tn, fp, fn, tp = metrics.confusion_matrix(labl, pred).ravel()
    
    fw = open(path + 'KNN-latent-dis/1-sum/conf_' + str(n) + '_1_' + distance_method + '.txt', "w")

    fw.write(str(tp) + "\n")
    fw.write(str(tn) + "\n")
    fw.write(str(fp) + "\n")
    fw.write(str(fn) + "\n")
    fw.write(str(score))
    
    fw = open(path + 'KNN-latent-dis/1-sum/dist1_' + str(n) + '_1_' + distance_method + '.txt', "w")
    
    for l in lst_dis1:
        fw.write(str(l) + '\n')
        
    fw = open(path + 'KNN-latent-dis/1-sum/dist2_' + str(n) + '_1_' + distance_method + '.txt', "w")
    
    for l in lst_dis2:
        fw.write(str(l) + '\n')

    
    fw.close()
    fwd.close()
    
def predict_knn_latent_var_sum_2(path, n):
    
    distance_method = 'cosine'
    distance_method = 'euclidean'
    
    with open(path + 'W1.csv') as fw1:
        array = list(csv.reader(fw1, quoting=csv.QUOTE_NONNUMERIC))
    W1 = np.array(array)
    
    with open(path + 'W2.csv') as fw2:
        array = list(csv.reader(fw2, quoting=csv.QUOTE_NONNUMERIC))
    W2 = np.array(array)
    
    with open(path + 'H1.csv') as fh1:
        array = list(csv.reader(fh1, quoting=csv.QUOTE_NONNUMERIC))
    H1 = np.array(array)
    
    with open(path + 'H2.csv') as fh2:
        array = list(csv.reader(fh2, quoting=csv.QUOTE_NONNUMERIC))
    H2 = np.array(array)
    
    with open(path + 'KNN-latent-dis/H1-var.txt') as file:
        for line in file:
            seq = line.split()
            float_seq = [float(i) for i in seq]
    H1v = np.array(float_seq)
    
    with open(path + 'KNN-latent-dis/H2-var.txt') as file:
        for line in file:
            seq = line.split()
            float_seq = [float(i) for i in seq]
    H2v = np.array(float_seq)
    
    sh1 = np.sum(H1v)
    sh2 = np.sum(H2v)
    
    
    dic = {}
    
    with open(path[:-6] + "perfAll.txt") as file:
        for line in file:
            id, pre, post, lg, avg = line.split('\t')
            dic[id] = lg
    
    label = []
    predict = []
    
    lst_dis1 = []
    lst_dis2 = []
    
    
    fwd = open(path + 'KNN-latent-dis/dis_' + str(n) + '.txt', "w")
    
    file_name = "VectorNormal1-" + str(n)
    
    with open(path + file_name + ".txt") as file:
        for line in file:
            id = line.split('\t')[0]
            perf = dic.get(id)
            
            if perf == 'h':
                lbl = 0
            elif perf == 'l':
                lbl = 1
            else:
                continue
            
            label.append(lbl)
            
            array = [float(digit) for digit in line.split('\t')[1].split(',')]
            test = np.array(array)
            
            inv1 = np.dot(np.linalg.pinv(W1, rcond = 0.0), test)
            inv2 = np.dot(np.linalg.pinv(W2, rcond = 0.0), test)
            
            dis1 = 0
            dis2 = 0
            
            for row in H1:
                dis1 = dis1 + distance_euc(inv1, row) / sh1
            for row in H2:
                dis2 = dis2 + distance_euc(inv2, row) / sh2
                
            lst_dis1.append(dis1)
            lst_dis2.append(dis2)
            
            '''
            if distance_method.startswith('c'):
                dis1 = distance_cos(inv1, H1)
                dis2 = distance_cos(inv2, H2)
            elif distance_method.startswith('e'):
                dis1 = distance_euc(inv1, H1)
                dis2 = distance_euc(inv2, H2)
            else:
                print 'Invalid Distance Measure'
                break
            '''
            
            fwd.write(str(dis1) + '\t' + str(dis2) + '\n')
            
            if (dis1 < dis2):
                pred = 1
            else:
                pred = 0
            
            predict.append(pred)
            
    pred = np.array(predict)
    labl = np.array(label)
    
    dir = path + 'KNN-latent-dis/2-sum/'
    if not os.path.exists(dir):
        os.mkdir(dir)
        
    fw = open(path + 'KNN-latent-dis/2-sum/pred_' + str(n) + '_2_' + distance_method + '.txt', "w")
    l = len(labl)
    for i in range(0, l):
        fw.write(str(pred[i]) + '\t' + str(labl[i]) + '\n')
        
    score = metrics.accuracy_score(labl, pred)
    tn, fp, fn, tp = metrics.confusion_matrix(labl, pred).ravel()
    
    fw = open(path + 'KNN-latent-dis/2-sum/conf_' + str(n) + '_2_' + distance_method + '.txt', "w")

    fw.write(str(tp) + "\n")
    fw.write(str(tn) + "\n")
    fw.write(str(fp) + "\n")
    fw.write(str(fn) + "\n")
    fw.write(str(score))
    
    fw = open(path + 'KNN-latent-dis/2-sum/dist1_' + str(n) + '_2_' + distance_method + '.txt', "w")
    
    for l in lst_dis1:
        fw.write(str(l) + '\n')
        
    fw = open(path + 'KNN-latent-dis/2-sum/dist2_' + str(n) + '_2_' + distance_method + '.txt', "w")
    
    for l in lst_dis2:
        fw.write(str(l) + '\n')
    
    fw.close()
    fwd.close()

def predict_knn_latent_var_sum_3(path, n):
    
    distance_method = 'cosine'
    distance_method = 'euclidean'
    
    with open(path + 'W1.csv') as fw1:
        array = list(csv.reader(fw1, quoting=csv.QUOTE_NONNUMERIC))
    W1 = np.array(array)
    
    with open(path + 'W2.csv') as fw2:
        array = list(csv.reader(fw2, quoting=csv.QUOTE_NONNUMERIC))
    W2 = np.array(array)
    
    with open(path + 'H1.csv') as fh1:
        array = list(csv.reader(fh1, quoting=csv.QUOTE_NONNUMERIC))
    H1 = np.array(array)
    
    with open(path + 'H2.csv') as fh2:
        array = list(csv.reader(fh2, quoting=csv.QUOTE_NONNUMERIC))
    H2 = np.array(array)
    
    with open(path + 'KNN-latent-dis/H1-var.txt') as file:
        for line in file:
            seq = line.split()
            float_seq = [float(i) for i in seq]
    H1v = np.array(float_seq)
    
    with open(path + 'KNN-latent-dis/H2-var.txt') as file:
        for line in file:
            seq = line.split()
            float_seq = [float(i) for i in seq]
    H2v = np.array(float_seq)
    
    #sh1 = np.sum(H1)
    #sh2 = np.sum(H2)
    
    
    dic = {}
    
    with open(path[:-6] + "perfAll.txt") as file:
        for line in file:
            id, pre, post, lg, avg = line.split('\t')
            dic[id] = lg
    
    label = []
    predict = []
    
    lst_dis1 = []
    lst_dis2 = []
    
    
    fwd = open(path + 'KNN-latent-dis/dis_' + str(n) + '.txt', "w")
    
    file_name = "VectorNormal1-" + str(n)
    
    with open(path + file_name + ".txt") as file:
        for line in file:
            id = line.split('\t')[0]
            perf = dic.get(id)
            
            if perf == 'h':
                lbl = 0
            elif perf == 'l':
                lbl = 1
            else:
                continue
            
            label.append(lbl)
            
            array = [float(digit) for digit in line.split('\t')[1].split(',')]
            test = np.array(array)
            
            inv1 = np.dot(np.linalg.pinv(W1, rcond = 0.0), test)
            inv2 = np.dot(np.linalg.pinv(W2, rcond = 0.0), test)
            
            dis1 = 0
            dis2 = 0
            
            x1, y1 = H1.shape
            x2, y2 = H2.shape
            
            for i in range(0, y1):
                dis1 = dis1 + distance_euc(inv1, H1[i]) / H1v[i]
            
            for i in range(0, y2):
                dis2 = dis2 + distance_euc(inv2, H2[i]) / H2v[i]
            
            '''
            for row in H1:
                dis1 = dis1 + distance_euc(inv1, row) / sh1
            for row in H2:
                dis2 = dis2 + distance_euc(inv2, row) / sh2
            '''
            
            lst_dis1.append(dis1)
            lst_dis2.append(dis2)
            
            '''
            if distance_method.startswith('c'):
                dis1 = distance_cos(inv1, H1)
                dis2 = distance_cos(inv2, H2)
            elif distance_method.startswith('e'):
                dis1 = distance_euc(inv1, H1)
                dis2 = distance_euc(inv2, H2)
            else:
                print 'Invalid Distance Measure'
                break
            '''
            
            fwd.write(str(dis1) + '\t' + str(dis2) + '\n')
            
            if (dis1 < dis2):
                pred = 1
            else:
                pred = 0
            
            predict.append(pred)
            
    pred = np.array(predict)
    labl = np.array(label)
    
    dir = path + 'KNN-latent-dis/3-sum/'
    if not os.path.exists(dir):
        os.mkdir(dir)
        
    fw = open(path + 'KNN-latent-dis/3-sum/pred_' + str(n) + '_3_' + distance_method + '.txt', "w")
    l = len(labl)
    for i in range(0, l):
        fw.write(str(pred[i]) + '\t' + str(labl[i]) + '\n')
        
    score = metrics.accuracy_score(labl, pred)
    tn, fp, fn, tp = metrics.confusion_matrix(labl, pred).ravel()
    
    fw = open(path + 'KNN-latent-dis/3-sum/conf_' + str(n) + '_3_' + distance_method + '.txt', "w")

    fw.write(str(tp) + "\n")
    fw.write(str(tn) + "\n")
    fw.write(str(fp) + "\n")
    fw.write(str(fn) + "\n")
    fw.write(str(score))
    
    fw = open(path + 'KNN-latent-dis/3-sum/dist1_' + str(n) + '_3_' + distance_method + '.txt', "w")
    
    for l in lst_dis1:
        fw.write(str(l) + '\n')
        
    fw = open(path + 'KNN-latent-dis/3-sum/dist2_' + str(n) + '_3_' + distance_method + '.txt', "w")
    
    for l in lst_dis2:
        fw.write(str(l) + '\n')
    
    fw.close()
    fwd.close()


def predict_knn_latent_var_2(path, n):
    
    distance_method = 'cosine'
    distance_method = 'euclidean'
    
    with open(path + 'W1.csv') as fw1:
        array = list(csv.reader(fw1, quoting=csv.QUOTE_NONNUMERIC))
    W1 = np.array(array)
    
    with open(path + 'W2.csv') as fw2:
        array = list(csv.reader(fw2, quoting=csv.QUOTE_NONNUMERIC))
    W2 = np.array(array)
    
    dic = {}
    
    with open(path[:-6] + "perfAll.txt") as file:
        for line in file:
            id, pre, post, lg, avg = line.split('\t')
            dic[id] = lg
    
    label = []
    predict = []


    with open(path + 'KNN-latent-dis/H1-avg.txt') as file:
        for line in file:
            seq = line.split()
            float_seq = [float(i) for i in seq]
    H1 = np.array(float_seq)
    
    with open(path + 'KNN-latent-dis/H2-avg.txt') as file:
        for line in file:
            seq = line.split()
            float_seq = [float(i) for i in seq]
    H2 = np.array(float_seq)
    
    with open(path + 'KNN-latent-dis/H1-var.txt') as file:
        for line in file:
            seq = line.split()
            float_seq = [float(i) for i in seq]
    V1 = np.array(float_seq)
    
    with open(path + 'KNN-latent-dis/H2-var.txt') as file:
        for line in file:
            seq = line.split()
            float_seq = [float(i) for i in seq]
    V2 = np.array(float_seq)
    
    sv1 = np.sum(V1)
    sv2 = np.sum(V2)
    
    fwd = open(path + 'KNN-latent-dis/dis_' + str(n) + '.txt', "w")
    
    file_name = "VectorNormal1-" + str(n)
    
    with open(path + file_name + ".txt") as file:
        for line in file:
            id = line.split('\t')[0]
            perf = dic.get(id)
            
            if perf == 'h':
                lbl = 0
            elif perf == 'l':
                lbl = 1
            else:
                continue
            
            label.append(lbl)
            
            array = [float(digit) for digit in line.split('\t')[1].split(',')]
            test = np.array(array)
            
            inv1 = np.dot(np.linalg.pinv(W1, rcond = 0.0), test)
            inv2 = np.dot(np.linalg.pinv(W2, rcond = 0.0), test)
            
            if distance_method.startswith('c'):
                dis1 = distance_cos(inv1, H1)/sv1
                dis2 = distance_cos(inv2, H2)/sv2
            elif distance_method.startswith('e'):
                dis1 = distance_euc(inv1, H1)/sv1
                dis2 = distance_euc(inv2, H2)/sv2
            else:
                print 'Invalid Distance Measure'
                break
            
            fwd.write(str(dis1) + '\t' + str(dis2) + '\n')
            
            if (dis1 < dis2):
                pred = 1
            else:
                pred = 0
            
            predict.append(pred)
            
    pred = np.array(predict)
    labl = np.array(label)
    
    dir = path + 'KNN-latent-dis/2/'
    if not os.path.exists(dir):
        os.mkdir(dir)
        
    fw = open(path + 'KNN-latent-dis/2/pred_' + str(n) + '_2_' + distance_method + '.txt', "w")
    l = len(labl)
    for i in range(0, l):
        fw.write(str(pred[i]) + '\t' + str(labl[i]) + '\n')
        
    score = metrics.accuracy_score(labl, pred)
    tn, fp, fn, tp = metrics.confusion_matrix(labl, pred).ravel()
    
    fw = open(path + 'KNN-latent-dis/2/conf_' + str(n) + '_2_' + distance_method + '.txt', "w")

    fw.write(str(tp) + "\n")
    fw.write(str(tn) + "\n")
    fw.write(str(fp) + "\n")
    fw.write(str(fn) + "\n")
    fw.write(str(score))
        
    fw.close()
    fwd.close()

def predict_knn_latent_var_multi(path, n):
    
    with open(path + 'W1.csv') as fw1:
        array = list(csv.reader(fw1, quoting=csv.QUOTE_NONNUMERIC))
    W1 = np.array(array)
    
    with open(path + 'W2.csv') as fw2:
        array = list(csv.reader(fw2, quoting=csv.QUOTE_NONNUMERIC))
    W2 = np.array(array)
    
    dic = {}
    
    with open(path[:-6] + "perfAll.txt") as file:
        for line in file:
            id, pre, post, lg, avg = line.split('\t')
            dic[id] = lg
    
    label = []
    predict = []


    with open(path + 'KNN-latent-dis/H1-avg.txt') as file:
        for line in file:
            seq = line.split()
            float_seq = [float(i) for i in seq]
    H1 = np.array(float_seq)
    
    with open(path + 'KNN-latent-dis/H2-avg.txt') as file:
        for line in file:
            seq = line.split()
            float_seq = [float(i) for i in seq]
    H2 = np.array(float_seq)
    
    with open(path + 'KNN-latent-dis/H1-var.txt') as file:
        for line in file:
            seq = line.split()
            float_seq = [float(i) for i in seq]
    V1 = np.array(float_seq)
    sv1 = np.sum(V1)
    
    with open(path + 'KNN-latent-dis/H2-var.txt') as file:
        for line in file:
            seq = line.split()
            float_seq = [float(i) for i in seq]
    V2 = np.array(float_seq)
    sv2 = np.sum(V2)
    
    
    fwd = open(path + 'KNN-latent-dis/dis_' + str(n) + '.txt', "w")
    
    file_name = "VectorNormal1-" + str(n)
    
    fw = open(path + 'KNN-latent-dis/alaki_' + str(n) + '_res.txt', "w")
    
    
    with open(path + file_name + ".txt") as file:
        for line in file:
            id = line.split('\t')[0]
            perf = dic.get(id)
            
            if perf == 'h':
                lbl = 0
            elif perf == 'l':
                lbl = 1
            else:
                continue
            
            label.append(lbl)
            
            array = [float(digit) for digit in line.split('\t')[1].split(',')]
            test = np.array(array)
            resW1 = np.dot(test, W1)
            resW2 = np.dot(test, W2)
            
            print resW1
            for i in resW1:
                fw.write(str(i) + '\t')
            
            fw.write('\n')
            
            print resW2
            for i in resW2:
                fw.write(str(i) + '\t')
            
            fw.write('\n')
            
            print resW2
            
            
            
            inv1 = np.dot(np.linalg.pinv(W1, rcond = 0.0), test)
            inv2 = np.dot(np.linalg.pinv(W2, rcond = 0.0), test)
            
            print inv1
            print inv2
            
            print inv1
            for i in inv1:
                fw.write(str(i) + '\t')
            
            fw.write('\n')
            
            print inv2
            for i in inv2:
                fw.write(str(i) + '\t')
            
            '''
            print 'W1 pinv'
            print np.linalg.pinv(W1, rcond = 0.0)
            print 'W2 pinv'
            print np.linalg.pinv(W2, rcond = 0.0)
            '''
            
            
            
            dis1 = distance_cos(inv1, H1)
            #distance_euc
            
            print dis1
            
            disv1 = dis1/sv1
            
            print disv1
            
            dis2 = distance_cos(inv2, H2)
            
            print dis2
            
            disv2 = dis1/sv2
            
            print disv2

            fwd.write(str(dis1) + '\t' + str(dis2) + '\t' + str(disv1) + '\t' + str(disv2) + '\n')
            
            if (dis1 > dis2):
                pred = 1
            else:
                pred = 0
            
            print pred
            predict.append(pred)
            
            break
            
    '''        
    pred = np.array(predict)
    labl = np.array(label)
    
    score = metrics.accuracy_score(labl, pred)
    tn, fp, fn, tp = metrics.confusion_matrix(labl, pred).ravel()
    '''    
    
    
    
        
    fw.close()
    fwd.close()
    

def predict_knn_latent_individual(path, n):
    
    with open(path + 'W1.csv') as fw1:
        array = list(csv.reader(fw1, quoting=csv.QUOTE_NONNUMERIC))
    W1 = np.array(array)
    
    with open(path + 'W2.csv') as fw2:
        array = list(csv.reader(fw2, quoting=csv.QUOTE_NONNUMERIC))
    W2 = np.array(array)
    
    dic = {}
    
    with open(path[:-6] + "perfAll.txt") as file:
        for line in file:
            id, pre, post, lg, avg = line.split('\t')
            dic[id] = lg
    
    label = []
    predict = []
    
    
    with open(path + 'H1.csv') as h1f:
        H1 = list(csv.reader(h1f, quoting=csv.QUOTE_NONNUMERIC))
    
    with open(path + 'H2.csv') as h2f:
        H2 = list(csv.reader(h2f, quoting=csv.QUOTE_NONNUMERIC))
    
    
    dir = path + 'KNN-latent-individual/'
    if not os.path.exists(dir):
        os.mkdir(dir)
    
    fwd = open(path + 'KNN-latent-individual/dis_' + str(n) + '.txt', "w")
    
    file_name = "VectorNormal1-" + str(n)
    
    with open(path + file_name + ".txt") as file:
        for line in file:
            id = line.split('\t')[0]
            perf = dic.get(id)
            
            if perf == 'h':
                lbl = 0
            elif perf == 'l':
                lbl = 1
            else:
                continue
            
            label.append(lbl)
            
            array = [float(digit) for digit in line.split('\t')[1].split(',')]
            
            test = np.array(array)
            resW1 = np.dot(test, W1)
            resW2 = np.dot(test, W2)
            
            dis1 = 0
            
            for H in H1:
                d = distance_cos(resW1, H)
                dis1 = dis1 + d
            
            dis2 = 0
            
            for H in H2:
                d = distance_cos(resW2, H)
                dis2 = dis2 + d
            
            
            fwd.write(str(dis1) + '\t' + str(dis2) + '\n')
            
            
            if (dis1 > dis2):
                pred = 0
            else:
                pred = 1
              
            predict.append(pred)
            
            
    pred = np.array(predict)
    labl = np.array(label)
    
    score = metrics.accuracy_score(labl, pred)
    tn, fp, fn, tp = metrics.confusion_matrix(labl, pred).ravel()
        
    
    fw = open(path + 'KNN-latent-individual/con_' + str(n) + '_res.txt', "w")

    fw.write(str(tp) + "\n")
    fw.write(str(tn) + "\n")
    fw.write(str(fp) + "\n")
    fw.write(str(fn) + "\n")
    fw.write(str(score))
        
    fw.close()
    fwd.close()
    

def distance_cos(a, b):
    return np.dot(a, b)/(norm(a) * norm(b))

    
def distance_euc(a, b):
    return np.linalg.norm(a-b)
    

def average_knn_latent(path, per, n):
    #tn, fp, fn, tp
    '''
    tn = 0
    fp = 0
    fn = 0
    tp = 0
    acc = 0
    '''
    
    tns = []
    fps = []
    fns = []
    tps = []
    accs = []
    
    
    for i in range(1, 5):
        pathn = path + str(i) + '/' + str(per)
        
        #dir = pathn + '/KNN-latent/con_' + str(n) + '_res.txt'
        dir = pathn + '/KNN-latent-dis/con_' + str(n) + '_res.txt'
        with open(dir) as file:
            print dir
            
            lines = file.readlines()
            line = [float(x.strip()) for x in lines]
            
            '''
            tns.append(line[0])
            fps.append(line[1])
            fns.append(line[2])
            tps.append(line[3])
            accs.append(line[4])
            
            tn = tn + line[0]
            fp = fp + line[1]
            fn = fn + line[2]
            tp = tp + line[3]
            acc = acc + line[4]
            '''
            
            tps.append(line[0])
            tns.append(line[1])
            fps.append(line[2])
            fns.append(line[3])
            accs.append(line[4])
            
            
    print np.mean(tps)
    print np.mean(tns)
    print np.mean(fps)
    print np.mean(fns)
    print np.mean(accs)
    
    print np.var(tps)
    print np.var(tns)
    print np.var(fps)
    print np.var(fns)
    print np.var(accs)
    
def average_knn_latent_1(path, per, n):

    tns = []
    fps = []
    fns = []
    tps = []
    accs = []
    
    
    for i in range(1, 5):
        pathn = path + str(i) + '/' + str(per)
        
        #dir = pathn + '/KNN-latent-dis/3-sum/conf_' + str(n) + '_3_euclidean.txt'
        dir = pathn + '/KNN-latent-dis/7/con_' + str(n) + '_res.txt'
        with open(dir) as file:
            print dir
            
            lines = file.readlines()
            line = [float(x.strip()) for x in lines]
            
            tps.append(line[0])
            tns.append(line[1])
            fps.append(line[2])
            fns.append(line[3])
            accs.append(line[4])
            
            
    print np.mean(tps)
    print np.mean(tns)
    print np.mean(fps)
    print np.mean(fns)
    print np.mean(accs)
    print
    print np.var(tps)
    print np.var(tns)
    print np.var(fps)
    print np.var(fns)
    print np.var(accs)
    print

def average_knn(path, per, n):
    #tn, fp, fn, tp
    '''
    tn = 0
    fp = 0
    fn = 0
    tp = 0
    acc = 0
    '''
    
    tns = []
    fps = []
    fns = []
    tps = []
    accs = []
    
    
    for i in range(1, 5):
        pathn = path + str(i) + '/' + str(per)
        
        #dir = pathn + '/KNN-latent/con_' + str(n) + '_res.txt'
        dir = pathn + '/KNN/knn-' + str(n) + '.txt'
        print dir
        with open(dir) as file:
            
            lines = file.readlines()
            line = [float(x.strip()) for x in lines]
            
            '''
            tns.append(line[0])
            fps.append(line[1])
            fns.append(line[2])
            tps.append(line[3])
            accs.append(line[4])
            
            tn = tn + line[0]
            fp = fp + line[1]
            fn = fn + line[2]
            tp = tp + line[3]
            acc = acc + line[4]
            '''
            
            tps.append(line[0])
            tns.append(line[1])
            fps.append(line[2])
            fns.append(line[3])
            accs.append(line[4])
            
            
    print np.mean(tps)
    print np.mean(tns)
    print np.mean(fps)
    print np.mean(fns)
    print np.mean(accs)
    
    print np.var(tps)
    print np.var(tns)
    print np.var(fps)
    print np.var(fns)
    print np.var(accs)
    
if __name__ == "__main__":
    #path = 'C:/Project/EDU/OLI_175318/update/step/sep/train-test/'
    #path = 'C:/Project/EDU/OLI_175318/update/step/sep/train-test/0.1/'
    path = 'C:/Project/EDU/OLI_175318/update/step/sep/train-test/method1/'
    
    path = 'C:/Project/EDU/files/2013/example/Topic/60/predictionFilter2/'
    
    '''
    for i in range(1,5):
        dir = path + str(i)
        if not os.path.exists(dir):
            os.mkdir(dir)
        for per in np.arange(0.1, 0.6, 0.1):
            j = round(per, 1)
            print i, j
            split_sk(path, j, i)
            split_test(path, j, i)
    '''       
    
    
    '''
    filename = 'test'
    split_perf(path, filename)
    
    filename = 'train'
    split_perf(path, filename)
    '''
    
    #split_test(path)
    #normal(path)
    
    
    path = 'C:/Project/EDU/OLI_175318/update/step/sep/train-test/method1/4/0.5/'
    print path[:-6]
    
    path = 'C:/Project/EDU/OLI_175318/update/step/sep/train-test/method1/1/0.1/MuParamTest/k20/c10d10/'
    
    c = 10
    d = 10
    for i in range(10,101,10):
        print i
        #normal_test(path, i)
        #predict(path, i, c, d)
        #count_performance(path ,i)
        
    
    #predict(path)
    
    
    '''
    a = np.array([1,2,3])
    array_to_str(a)
    '''
    
    
    path = 'C:/Project/EDU/OLI_175318/update/step/sep/train-test/method1/'
    path = 'C:/Project/EDU/files/2013/example/Topic/60/predictionFilter2/'
    
    '''
    dir1 = ['1/','2/','3/','4/']
    dir2 = ['0.1/','0.2/','0.3/','0.4/','0.5/']
    for d1 in dir1:
        for d2 in dir2:
            pathn = path + d1 + d2
            print pathn
            for i in range(50, 101, 10):
                find_perf_test(pathn, i)
                knn(pathn, i)
    '''

    '''
    for i in range(10, 101, 10):
        find_perf_test(path, i)
        knn(path, i)
    
    
    path = 'C:/Project/EDU/OLI_175318/update/step/sep/train-test/method1/1/0.1/'
    
    '''
    
    '''
    path = 'C:/Project/EDU/OLI_175318/update/step/sep/train-test/method1/'
    dir1 = ['1/','2/','3/','4/']
    dir2 = ['0.1/','0.2/','0.3/','0.4/','0.5/']
    for d1 in dir1:
        for d2 in dir2:
            pathn = path + d1 + d2
            print pathn
            #H = ['H1', 'H2']
            #for h in H:
            #    H_avg_var(pathn, h)
            for i in range(10, 101, 10):
                #predict_knn_latent_var(pathn, i)
                predict_knn_latent_individual(pathn, i)
    '''
    path = 'C:/Project/EDU/OLI_175318/update/step/sep/train-test/method1/1/0.1/'
    path = 'C:/Project/EDU/files/2013/example/Topic/60/predictionFilter2/1/0.1/testNew/k20/c10d10/'
    n = 100
    #predict_knn_latent_var_multi(path, n)
    #predict_knn_latent_var_1(path, n)
    #find_perf_test(path)
    #predict_knn_latent_var_sum_3(path, n)
    
    
    #path = 'C:/Project/EDU/OLI_175318/update/step/sep/train-test/method1/'
    '''
    dir1 = ['1/','2/','3/','4/']
    dir2 = ['0.1/','0.2/','0.3/','0.4/','0.5/']
    for d1 in dir1:
        for d2 in dir2:
            pathn = path + d1 + d2
            print pathn
            for i in range(10, 101, 10):
                print i
                #predict_knn_latent_var_2(pathn, i)
                #predict_knn_latent_var_sum_3(pathn, i)
                #predict_knn_latent_test(pathn, i)
    '''
    
    '''
    for i in range(10, 101, 10):
        predict_knn_latent(path, i)
    '''    
    
    
    path = 'C:/Project/EDU/OLI_175318/update/step/sep/train-test/method1/1/0.1/MuParamTest/k20/c10d10/'
    pathn = path + 'KNN-latent-mu/'
    
    '''
    H = ['H1', 'H2']
    for h in H:
        H_avg_var(path, pathn, h)
    '''
    
    
    for i in range(10, 101, 10):
        #predict_knn_latent(path, i)
        predict_knn_latent_test(path, pathn, i)
    
    
    
    '''
    
    path = 'C:/Project/EDU/OLI_175318/update/step/sep/train-test/method1/'
    #path = 'C:/Project/EDU/files/2013/example/Topic/60/predictionFilter2/'
    per = 0.5
    for n in range(10, 101, 10):
        print n
        #average_knn_latent(path, per, n)
        #average_knn(path, per, n)
        #average_knn_latent_1(path, per, n)
    '''