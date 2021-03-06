# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 18:30:30 2019

@author: mirza
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import os
import matplotlib
from numpy import linalg as LA
import sys
import seaborn as sns

#mode = 'write'
#mode = 'test'

def gd_simple(path, alpha, beta):
    
    with open('C:/Project/EDU/files/2013/example/Topic/similarity/dual/l.txt') as file:
        array2dX1 = [[float(digit) for digit in line.split(',')] for line in file]
        
    X1 = np.array(array2dX1)
    
    with open('C:/Project/EDU/files/2013/example/Topic/similarity/dual/h.txt') as file:
        array2dX2 = [[float(digit) for digit in line.split(',')] for line in file]
    
    X2 = np.array(array2dX2)
    
    
    epoc = 1000
    
    m1,n1 = X1.shape
    m2,n2 = X2.shape
    
    
    W1 = np.random.rand(m1,k)
    H1 = np.random.rand(n1,k)
    
    W2 = np.random.rand(m2,k)
    H2 = np.random.rand(n2,k)
    
    
    W1 = W1 / 10.0
    H1 = H1 / 10.0
    
    W2 = W2 / 10.0
    H2 = H2 / 10.0
    
    
    index = []
    errX1 = []
    errX2 = []
    errX1X2 = []
    
    gama = 2.0 - (alpha + beta)

    
    for e in range(epoc):
        learning_rate = 0.01/np.sqrt(e+1)
    
        grad_W1 = -2 * gama * np.dot(X1 - np.dot(W1, H1.T), H1)
        W1n = W1 - learning_rate * grad_W1
        
        grad_H1 = -2 * gama * np.dot(W1.T, X1 - np.dot(W1, H1.T))
        H1Tn = H1.T - learning_rate * grad_H1
        
        grad_W2 = -2 * gama * np.dot(X2 - np.dot(W2, H2.T), H2)
        W2n = W2 - learning_rate * grad_W2
        
        grad_H2 = -2 * gama * np.dot(W2.T, X2 - np.dot(W2, H2.T))
        H2Tn = H2.T - learning_rate * grad_H2
        
        
        W1n[W1n<0] = 0
        H1Tn[H1Tn<0] = 0

        W2n[W2n<0] = 0
        H2Tn[H2Tn<0] = 0

        errorX1 = error(X1, np.dot(W1, H1.T))
        
        errorX2 = error(X2, np.dot(W2, H2.T))
        
        index.append(e)
        
        errX1.append(errorX1)

        errX2.append(errorX2)
        
        errX1X2.append((errorX1 + errorX2)/2)
        
        W1 = W1n
        H1 = H1Tn.T
        
        W2 = W2n
        H2 = H2Tn.T
        
        
        if (e % 10 == 0):
            print (e)
        
    
    #mode = 'test'
    mode = 'write'
    
    if (mode == 'write'):        
        if (os.path.isdir(path) == False):
            os.mkdir(path)
        pathk = path + "k" + str(k)
        if (os.path.isdir(pathk) == False):
            os.mkdir(pathk)
        
        pathkc = pathk + "/c" + str(kc) + "d" + str(k-kc)    
        if (os.path.isdir(pathkc) == False):
            os.mkdir(pathkc)
        
        
        if (mode == 'write'):
            
            np.savetxt(pathkc + "/W1.csv", W1, delimiter=",")
            np.savetxt(pathkc + "/W2.csv", W2, delimiter=",")
            
            np.savetxt(pathkc + "/H1.csv", H1, delimiter=",")
            np.savetxt(pathkc + "/H2.csv", H2, delimiter=",")
            
            
            fw = open(pathkc + '/err.txt', "w")
            
            fwx1 = open(pathkc + '/errors-X1.txt', "w")
            fwx2 = open(pathkc + '/errors-X2.txt', "w")
            
            for i in errX1:
                fwx1.write(str(i) + '\n')
                
            for i in errX2:
                fwx2.write(str(i) + '\n')
            
            
            errX1mae = errorMAE(X1, np.dot(W1, H1.T))
            errX2mae = errorMAE(X2, np.dot(W2, H2.T))
            
            fw.write("Error X1 RMSE: " + str(errX1[-1]) + " MAE: " + str(errX1mae) + "\n")
            fw.write("Error X2 RMSE: " + str(errX2[-1]) + " MAE: " + str(errX2mae) +  "\n")
            fw.write("Error X1+X2: " + str(errX1X2[-1]) + "\n")
            
            errorX1 = np.abs(X1 - np.dot(W1,np.transpose(H1)))
            errorX2 = np.abs(X2 - np.dot(W2,np.transpose(H2)))
            
                
            fw.close()
            fwx1.close()
            fwx2.close()
            
        pathk = path + str(k)
        
        err1 = error(X1, np.dot(W1, H1.T))
        #errmae1 = errorMAE(X1, np.dot(W1, H1.T))
        fw1 = open(pathk + 'err1.txt', "w")
        #fw1.write('RMSE: ' + str(err1) + '\nMAE: ' + str(errmae1))
        fw1.write(str(err1))
        fw1.close()
        
        err2 = error(X2, np.dot(W2, H2.T))
        #errmae2 = errorMAE(X2, np.dot(W2, H2.T))
        fw2 = open(pathk + 'err2.txt', "w")
        #fw2.write('RMSE: ' + str(err2) + '\nMAE: ' + str(errmae2))
        fw2.write(str(err2))
        fw2.close()
        
        
    pathk = path + "k" + str(k)
    pathkc = pathk + "/c" + str(kc) + "d" + str(k-kc)
    print (pathkc)
    
    
    #mode = 'test'
    mode = 'write'
    
    
    
    plt.figure()
    plt.plot(index,errX1)
    plt.title('Error X1')
    plt.xlabel('Iteration')
    if mode == 'write':
        plt.savefig(pathkc + "/ErrorX1.png")
        plt.clf()
        plt.close()
    
    plt.figure()
    plt.plot(index,errX2)
    plt.title('Error X2')
    plt.xlabel('Iteration')
    if mode == 'write':
        plt.savefig(pathkc + "/ErrorX2.png")
        plt.clf()
        plt.close()
    
    plt.figure()
    plt.plot(index,errX1X2)
    plt.title('Error X1X2')
    plt.xlabel('Iteration')
    if mode == 'write':
        plt.savefig(pathkc + "/ErrorX1X2.png")
        plt.clf()
        plt.close()
                
    print ('plot')
    plt.show()
    plt.rcParams.update({'font.size': 10})
    


    '''
    plt.figure(4)
    plt.plot(index,errS2)
    plt.title('Error S2')
    plt.xlabel('Iteration')
    if mode == 'write':
        plt.savefig(pathkc + "/ErrorS2.png")
    '''

    '''
    #fig = plt.figure()
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    #matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')
    plt.rcParams.update({'font.size': 24})
    
    
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize = (5,8))
    fig.subplots_adjust(hspace = .3)
    
    #plt.figure(figsize = (10,15))
    
    ax[0].plot(index,errSqrX1,label = 'X1 Error')
    ax[0].set_xlabel('Iteration')
    ax[0].set_ylabel('Reconstruction Error')
    ax[0].legend()
    
    ax[1].plot(index,errSqrX2,label = 'X2 Error')
    ax[1].set_xlabel('Iteration')
    ax[1].set_ylabel('Reconstruction Error')
    ax[1].legend()
    
    plt.savefig(pathkc + "/X1X2V_large.pdf")
    '''


def gd_eps_dual(typ, k, kc, path, alpha, beta, delta, sigma):
    
    with open('C:/Project/EDU/files/2013/example/Topic/similarity/dual/l.txt') as file:
        array2dX1 = [[float(digit) for digit in line.split(',')] for line in file]
        
    X1 = np.array(array2dX1)
    
    with open('C:/Project/EDU/files/2013/example/Topic/similarity/dual/h.txt') as file:
        array2dX2 = [[float(digit) for digit in line.split(',')] for line in file]
    
    X2 = np.array(array2dX2)
    
    with open('C:/Project/EDU/files/2013/example/Topic/similarity/dual/newmp.csv') as file:
        arrayS = [[float(digit) for digit in line.split(',')] for line in file]
        
    S = np.array(arrayS)
    
    epoc = 1000
    
    m1,n1 = X1.shape
    m2,n2 = X2.shape
    
    W1 = np.random.rand(m1,k)
    H1 = np.random.rand(n1,k)
    
    W2 = np.random.rand(m2,k)
    H2 = np.random.rand(n2,k)
    
    
    '''
    print 'W1: ', W1.shape
    print 'W2: ', W2.shape
    print 'H1: ', H1.shape
    print 'H2: ', H2.shape
    '''
    
    W1 = W1 / 10.0
    H1 = H1 / 10.0
    
    W2 = W2 / 10.0
    H2 = H2 / 10.0
    
    '''
    print 'W1', W1
    print 'W2', W2
    print 'H1', H1
    print 'H2', H2
    '''
    
    index = []
    errX1 = []
    errX2 = []
    errX1X2 = []
    errSqrC = []
    errD = []
    errCD1 = []
    errCD2 = []
        
    errS1 = []
    errS2 = []
    errS = []
    
    
    #C alpha
    #D beta
    
    #X1X2
    gama = 2.0 - (alpha + beta)
    #gama
    eps = 1
    #S delta
    
    #W1cW1d
    #sigma = 1.0
    
    reg = 1

    for e in range(epoc):
        learning_rate = 0.01/np.sqrt(e+1)
    
        W1c = W1[:,:kc]
        W1d = W1[:,kc:]
        
        H1c = H1[:,:kc]
        H1d = H1[:,kc:]
        
        W2c = W2[:,:kc]
        W2d = W2[:,kc:]
        
        H2c = H2[:,:kc]
        H2d = H2[:,kc:]
        
        #Con_sim II        
        if (typ == 'con'):
            W = np.concatenate((W1, W2), axis = 1)
        
        
        if (typ == 'con'):
            grad_w1c = (2 * gama * np.dot((np.dot(W1, H1.T) - X1), H1c)
            + 2 * alpha * (W1c - W2c)
            + 2 * reg * W1c
            - 4 * delta * np.dot((S - eps * np.dot(W, W.T)), W1c)
            + 2 * sigma * np.dot(np.dot(W1c, W1d.T), W1d)
            ) 
        else:
            if (typ == 'sep'):
                grad_w1c = (2 * gama * np.dot((np.dot(W1, H1.T) - X1), H1c)
                + 2 * alpha * (W1c - W2c)
                + 2 * reg * W1c
                - 4 * delta * np.dot((S - eps * np.dot(W1, W1.T)), W1c)
                )
        
        W1cn = W1c - learning_rate * grad_w1c
        
        
        if (typ == 'con'):
            grad_w2c = (2 * gama * np.dot((np.dot(W2, H2.T) - X2), H2c)
            - 2 * alpha * (W1c - W2c)
            + 2 * reg * W2c
            - 4 * delta * np.dot((S - eps * np.dot(W, W.T)), W2c)
            + 2 * sigma * np.dot(np.dot(W2c, W2d.T), W2d)
            )
        else:
            if (typ == 'sep'):
                grad_w2c = (2 * gama * np.dot((np.dot(W2, H2.T) - X2), H2c)
                - 2 * alpha * (W1c - W2c)
                + 2 * reg * W2c
                - 4 * delta * np.dot((S - eps * np.dot(W2, W2.T)), W2c)
                )
            
        W2cn = W2c - learning_rate * grad_w2c
        
        
        if (typ == 'con'):    
            grad_w1d = (2 * gama * np.dot((np.dot(W1, H1.T) - X1), H1d) 
            + 2 * beta * np.dot(W2d, np.dot(W1d.T,W2d))
            + 2 * reg * W1d
            - 4 * delta * np.dot((S - eps * np.dot(W, W.T)), W1d)
            + 2 * sigma * np.dot(np.dot(W1c, W1d.T), W1c)
            )
        else:
            if (typ == 'sep'):
                grad_w1d = (2 * np.dot((np.dot(W1, H1.T) - X1), H1d) 
                + 2 * beta * np.dot(W2d, np.dot(W1d.T,W2d))
                + 2 * reg * W1d
                - 4 * delta * np.dot((S - eps * np.dot(W1, W1.T)), W1d)
                )
        
        W1dn = W1d - learning_rate * grad_w1d
        
        
        if (typ == 'con'):
            grad_w2d = (2 * gama * np.dot((np.dot(W2, H2.T) - X2), H2d) 
            + 2 * beta * np.dot(W1d, np.dot(W1d.T,W2d))
            + 2 * reg * W2d
            - 4 * delta * np.dot((S - eps * np.dot(W, W.T)), W2d)
            + 2 * sigma * np.dot(np.dot(W2c, W2d.T), W2c)
            )
        else:
            if (typ == 'sep'):
                grad_w2d = (2 * np.dot((np.dot(W2, H2.T) - X2), H2d) 
                + 2 * beta * np.dot(W1d, np.dot(W1d.T,W2d))
                + 2 * reg * W2d
                - 4 * delta * np.dot((S - eps * np.dot(W2, W2.T)), W2d)
                )
                
        
        W2dn = W2d - learning_rate * grad_w2d
        
        grad_h1 = -2 * gama * np.dot(np.transpose(X1 - np.dot(W1, H1.T)), W1) + 2 * reg * H1
        H1n = H1 - learning_rate * grad_h1
        
        grad_h2 = -2 * gama * np.dot(np.transpose(X2 - np.dot(W2, H2.T)), W2) + 2 * reg * H2
        H2n = H2 - learning_rate * grad_h2
        
        if (typ == 'con'):
            grad_eps = delta * np.sum(np.multiply( (-2 * np.dot(W, W.T)) , (S - eps * np.dot(W, W.T)) ))
        else:
            if (typ == 'sep'):
                grad_eps = delta * (np.sum(np.multiply( (-2 * np.dot(W1, W1.T)) , (S - eps * np.dot(W1, W1.T)) ))
                + np.sum(np.multiply( (-2 * np.dot(W2, W2.T)) , (S - eps * np.dot(W2, W2.T)) ))
                )
        
        eps = eps - learning_rate * grad_eps
        
        
        W1n = np.concatenate((W1cn,W1dn), axis = 1)
        W2n = np.concatenate((W2cn,W2dn), axis = 1)
        
        
        W1n[W1n<0] = 0
        H1n[H1n<0] = 0

        W2n[W2n<0] = 0
        H2n[H2n<0] = 0

        errorX1 = error(X1, np.dot(W1, H1.T))
        
        errorX2 = error(X2, np.dot(W2, H2.T))
        
        errorSqrC = lossfuncSqr(W1cn, W2cn)
        
        errorD = lossfuncD(np.transpose(W1dn), W2dn)
        errorCD1 = lossfuncD(W1c, W1d.T)
        errorCD2 = lossfuncD(W1c, W2d.T)
        
        if (typ == 'con'):
            errorS = error(S, eps * np.dot(W, W.T))
        else:
            if (typ == 'sep'):
                errorS1 = error(S, eps * np.dot(W1,W1.T))
                errorS2 = error(S, eps * np.dot(W2,W2.T))
        
        index.append(e)
        
        errX1.append(errorX1)

        errX2.append(errorX2)
        
        errX1X2.append((errorX1 + errorX2)/2)
        
        errSqrC.append(errorSqrC)
        
        errD.append(errorD)
        
        errCD1.append(errorCD1)
        
        errCD2.append(errorCD2)
        
        if (typ == 'con'):
            errS.append(errorS)
        else:
            if (typ == 'sep'):
                errS1.append(errorS1)
                errS2.append(errorS2)

        W1 = W1n
        H1 = H1n
        
        W2 = W2n
        H2 = H2n
        
        
        if (e % 10 == 0):
            print (e)
            
        
    
    #mode = 'test'
    mode = 'write'
    
    if (mode == 'write'):        
        if (os.path.isdir(path) == False):
            os.mkdir(path)
        pathk = path + "k" + str(k)
        if (os.path.isdir(pathk) == False):
            os.mkdir(pathk)
        
        pathkc = pathk + "/c" + str(kc) + "d" + str(k-kc)    
        if (os.path.isdir(pathkc) == False):
            os.mkdir(pathkc)
        
        print 'Mode write'
        
        
        np.savetxt(pathkc + "/W1.csv", W1, delimiter=",")
        np.savetxt(pathkc + "/W2.csv", W2, delimiter=",")
        
        np.savetxt(pathkc + "/W1c.csv", W1c, delimiter=",")
        np.savetxt(pathkc + "/W2c.csv", W2c, delimiter=",")
        np.savetxt(pathkc + "/W1d.csv", W1d, delimiter=",")
        np.savetxt(pathkc + "/W2d.csv", W2d, delimiter=",")
        
        np.savetxt(pathkc + "/H1.csv", H1, delimiter=",")
        np.savetxt(pathkc + "/H2.csv", H2, delimiter=",")
        
        np.savetxt(pathkc + "/H1c.csv", H1c, delimiter=",")
        np.savetxt(pathkc + "/H2c.csv", H2c, delimiter=",")
        np.savetxt(pathkc + "/H1d.csv", H1d, delimiter=",")
        np.savetxt(pathkc + "/H2d.csv", H2d, delimiter=",")
        
        
        fw = open(pathkc + '/err.txt', "w")
        
        fw.write("Error X1: " + str(errX1[-1]) + "\n")
        fw.write("Error X2: " + str(errX2[-1]) + "\n")
        fw.write("Error X1+X2: " + str(errX1X2[-1]) + "\n")
        fw.write("Error C: " + str(errSqrC[-1]) + "\n")
        fw.write("Error D: " + str(errD[-1]) + "\n")
        fw.write("Error CD1: " + str(errCD1[-1]) + "\n")
        fw.write("Error CD2: " + str(errCD2[-1]) + "\n")
        if (typ == 'con'):
            fw.write("Error S: " + str(errS[-1]) + "\n")
        else:
            if (typ == 'sep'):
                fw.write("Error S1: " + str(errS1[-1]) + "\n")
                fw.write("Error S2: " + str(errS2[-1]) + "\n")
        fw.write("Eps: " + str(eps))
        
        errorX1 = np.abs(X1 - np.dot(W1,np.transpose(H1)))
        errorX2 = np.abs(X2 - np.dot(W2,np.transpose(H2)))
        
            
        fw.close()
        
        pathk = path + str(k)
        
        err1 = error(X1, np.dot(W1, H1.T))
        fw1 = open(pathk + 'err1.txt', "w")
        fw1.write(str(err1))
        fw1.close()
        
        err2 = error(X2, np.dot(W2, H2.T))
        fw2 = open(pathk + 'err2.txt', "w")
        fw2.write(str(err2))
        fw2.close()
        
        if (typ == 'con'):
            errs = error(S, np.dot(W,W.T))
            fw3 = open(pathk + 'errs.txt', "w")
            fw3.write(str(errs))
            fw3.close()
        else:
            if (typ == 'sep'):
                errs1 = error(S, np.dot(W1,W1.T))
                fw3 = open(pathk + 'errs2.txt', "w")
                fw3.write(str(errs1))
                fw3.close()
                
                errs2 = error(S, np.dot(W2,W2.T))
                fw4 = open(pathk + 'errs1.txt', "w")
                fw4.write(str(errs2))
                fw4.close()
        

    pathk = path + "k" + str(k)
    pathkc = pathk + "/c" + str(kc) + "d" + str(k-kc)
    print (pathkc)
    
    
    #mode = 'test'
    mode = 'write'
    
    plt.figure()
    plt.plot(index,errX1)
    plt.title('Error X1')
    plt.xlabel('Iteration')
    if mode == 'write':
        plt.savefig(pathkc + "/ErrorX1.png")
        plt.clf()
        plt.close()
    
    plt.figure()
    plt.plot(index,errX2)
    plt.title('Error X2')
    plt.xlabel('Iteration')
    if mode == 'write':
        plt.savefig(pathkc + "/ErrorX2.png")
        plt.clf()
        plt.close()
    
    plt.figure()
    plt.plot(index,errSqrC)
    plt.title('Error C')
    plt.xlabel('Iteration')
    if mode == 'write':
        plt.savefig(pathkc + "/ErrorC.png")
        plt.clf()
        plt.close()
    
    plt.figure()
    plt.plot(index,errD)
    plt.title('Error D')
    plt.xlabel('Iteration')
    if mode == 'write':
        plt.savefig(pathkc + "/ErrorD.png")
        plt.clf()
        plt.close()
    
    plt.figure()
    plt.plot(index,errCD1)
    plt.title('Error CD1')
    plt.xlabel('Iteration')
    if mode == 'write':
        plt.savefig(pathkc + "/ErrorCD1.png")
        plt.clf()
        plt.close()
    
    plt.figure()
    plt.plot(index,errCD2)
    plt.title('Error CD2')
    plt.xlabel('Iteration')
    if mode == 'write':
        plt.savefig(pathkc + "/ErrorCD2.png")
        plt.clf()
        plt.close()
    
    
    if (typ == 'con'):
        plt.figure()
        plt.plot(index,errS)
        plt.title('Error S')
        plt.xlabel('Iteration')
        if mode == 'write':
            plt.savefig(pathkc + "/ErrorS.png")
            plt.clf()
            plt.close()
    else:
        if (typ == 'sep'):
            plt.figure()
            plt.plot(index,errS1)
            plt.title('Error S1')
            plt.xlabel('Iteration')
            if mode == 'write':
                plt.savefig(pathkc + "/ErrorS1.png")
                plt.clf()
                plt.close()
            
            plt.figure()
            plt.plot(index,errS2)
            plt.title('Error S2')
            plt.xlabel('Iteration')
            if mode == 'write':
                plt.savefig(pathkc + "/ErrorS2.png")
                plt.clf()
                plt.close()
                
    print ('plot')
    print eps
    #plt.show()
    plt.rcParams.update({'font.size': 10})
    


    '''
    plt.figure(4)
    plt.plot(index,errS2)
    plt.title('Error S2')
    plt.xlabel('Iteration')
    if mode == 'write':
        plt.savefig(pathkc + "/ErrorS2.png")
    '''

    '''
    #fig = plt.figure()
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    #matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')
    plt.rcParams.update({'font.size': 24})
    
    
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize = (5,8))
    fig.subplots_adjust(hspace = .3)
    
    #plt.figure(figsize = (10,15))
    
    ax[0].plot(index,errSqrX1,label = 'X1 Error')
    ax[0].set_xlabel('Iteration')
    ax[0].set_ylabel('Reconstruction Error')
    ax[0].legend()
    
    ax[1].plot(index,errSqrX2,label = 'X2 Error')
    ax[1].set_xlabel('Iteration')
    ax[1].set_ylabel('Reconstruction Error')
    ax[1].legend()
    
    plt.savefig(pathkc + "/X1X2V_large.pdf")
    '''

def gd_eps(typ, k, kc, path, alpha, beta, delta):
    
    #path = 'C:/Project/EDU/files/2013/example/Topic/similarity/grid_mp_con/'
    
    #with open('C:/Project/EDU/OLI_175318/hint/lg/l.txt') as file:
    with open('C:/Project/EDU/files/2013/example/Topic/similarity/grid_nmp_con/l.txt') as file:
    #with open(path + 'l.txt') as file:
    #with open(path[0:-2] + 'l.txt') as file:
        array2dX1 = [[float(digit) for digit in line.split(',')] for line in file]
        #for line in file:
        #    for digit in line.split(','):
        #        print float(digit)
    
    X1 = np.array(array2dX1)
    
    #with open('C:/Project/EDU/OLI_175318/hint/lg/h.txt') as file:
    with open('C:/Project/EDU/files/2013/example/Topic/similarity/grid_nmp_con/h.txt') as file:
    #with open(path + 'h.txt') as file:
    #with open(path[0:-2] + 'h.txt') as file:
        array2dX2 = [[float(digit) for digit in line.split(',')] for line in file]
    
    X2 = np.array(array2dX2)
    
    #with open('C:/Project/EDU/OLI_175318/hint/lg/nms.csv') as file:
    with open('C:/Project/EDU/files/2013/example/Topic/similarity/grid_nmp_con/newmp.csv') as file:
    #with open(path + 'normal_mp.csv') as file:
    #with open(path[0:-2] + 'normal_mp.csv') as file:
    #with open(path[0:-2] + 'nms.csv') as file:
        arrayS = [[float(digit) for digit in line.split(',')] for line in file]
        
    S = np.array(arrayS)
    
    epoc = 1000
    
    m1,n1 = X1.shape
    m2,n2 = X2.shape
    
    W1 = np.random.rand(m1,k)
    H1 = np.random.rand(n1,k)
    
    W2 = np.random.rand(m2,k)
    H2 = np.random.rand(n2,k)
    
    
    '''
    print 'W1: ', W1.shape
    print 'W2: ', W2.shape
    print 'H1: ', H1.shape
    print 'H2: ', H2.shape
    '''
    
    W1 = W1 / 10.0
    H1 = H1 / 10.0
    
    W2 = W2 / 10.0
    H2 = H2 / 10.0
    
    '''
    print 'W1', W1
    print 'W2', W2
    print 'H1', H1
    print 'H2', H2
    '''
    
    index = []
    errX1 = []
    errX2 = []
    errX1X2 = []
    errSqrC = []
    errD = []
        
    errS1 = []
    errS2 = []
    errS = []
    
    eps_list = []
    
    
    #C
    #alpha = 0.5
    #D
    #beta = 0.5
    
    #X1X2
    gama = 2.0 - (alpha + beta)
    #gama = 0.8
    eps = 1
    #S
    #delta = 0.5
    
    reg = 1

    for e in range(epoc):
        learning_rate = 0.02/np.sqrt(e+1)
    
        W1c = W1[:,:kc]
        W1d = W1[:,kc:]
        
        H1c = H1[:,:kc]
        H1d = H1[:,kc:]
        
        W2c = W2[:,:kc]
        W2d = W2[:,kc:]
        
        H2c = H2[:,:kc]
        H2d = H2[:,kc:]
        
        #Con_sim II        
        if (typ == 'con'):
            W = np.concatenate((W1, W2), axis = 1)
        
        
        if (typ == 'con'):
            grad_w1c = (2 * gama * np.dot((np.dot(W1, H1.T) - X1), H1c)
            + 2 * alpha * (W1c - W2c)
            + 2 * reg * W1c
            - 4 * delta * np.dot((S - eps * np.dot(W, W.T)), W1c)
            ) 
        else:
            if (typ == 'sep'):
                grad_w1c = (2 * gama * np.dot((np.dot(W1, H1.T) - X1), H1c)
                + 2 * alpha * (W1c - W2c)
                + 2 * reg * W1c
                - 4 * delta * np.dot((S - eps * np.dot(W1, W1.T)), W1c)
                )
        '''
        print 'line1', 2 * gama * np.dot((np.dot(W1, H1.T) - X1), H1c)
        print 'line2', 2 * alpha * (W1c - W2c)
        print 'line3', 2 * reg * W1c
        print 'line4', 4 * delta * np.dot((S - eps * np.dot(W, W.T)), W1c)
        print 'X1', X1
        print 'W1, H1.t', np.dot(W1, H1.T)
        '''
        W1cn = W1c - learning_rate * grad_w1c
        #print 'W1cn', grad_w1c[0]
        
        if (typ == 'con'):
            grad_w2c = (2 * gama * np.dot((np.dot(W2, H2.T) - X2), H2c)
            - 2 * alpha * (W1c - W2c)
            + 2 * reg * W2c
            - 4 * delta * np.dot((S - eps * np.dot(W, W.T)), W2c)
            )
        else:
            if (typ == 'sep'):
                grad_w2c = (2 * gama * np.dot((np.dot(W2, H2.T) - X2), H2c)
                - 2 * alpha * (W1c - W2c)
                + 2 * reg * W2c
                - 4 * delta * np.dot((S - eps * np.dot(W2, W2.T)), W2c)
                )
            
        W2cn = W2c - learning_rate * grad_w2c
        #print 'W2cn', grad_w2c[0]
        
        if (typ == 'con'):    
            grad_w1d = (2 * gama * np.dot((np.dot(W1, H1.T) - X1), H1d) 
            + 2 * beta * np.dot(W2d, np.dot(W1d.T,W2d))
            + 2 * reg * W1d
            - 4 * delta * np.dot((S - eps * np.dot(W, W.T)), W1d)
            )
        else:
            if (typ == 'sep'):
                grad_w1d = (2 * np.dot((np.dot(W1, H1.T) - X1), H1d) 
                + 2 * beta * np.dot(W2d, np.dot(W1d.T,W2d))
                + 2 * reg * W1d
                - 4 * delta * np.dot((S - eps * np.dot(W1, W1.T)), W1d)
                )
        
        W1dn = W1d - learning_rate * grad_w1d
        
        
        if (typ == 'con'):
            grad_w2d = (2 * gama * np.dot((np.dot(W2, H2.T) - X2), H2d) 
            + 2 * beta * np.dot(W1d, np.dot(W1d.T,W2d))
            + 2 * reg * W2d
            - 4 * delta * np.dot((S - eps * np.dot(W, W.T)), W2d)
            )
        else:
            if (typ == 'sep'):
                grad_w2d = (2 * np.dot((np.dot(W2, H2.T) - X2), H2d) 
                + 2 * beta * np.dot(W1d, np.dot(W1d.T,W2d))
                + 2 * reg * W2d
                - 4 * delta * np.dot((S - eps * np.dot(W2, W2.T)), W2d)
                )
                
        
        W2dn = W2d - learning_rate * grad_w2d
        
        grad_h1 = -2 * gama * np.dot(np.transpose(X1 - np.dot(W1, H1.T)), W1) + 2 * reg * H1
        H1n = H1 - learning_rate * grad_h1
        
        grad_h2 = -2 * gama * np.dot(np.transpose(X2 - np.dot(W2, H2.T)), W2) + 2 * reg * H2
        H2n = H2 - learning_rate * grad_h2
        
        #grad_eps = np.square(LA.norm(np.dot(W, W.T)))
        if (typ == 'con'):
            grad_eps = delta * np.sum(np.multiply( (-2 * np.dot(W, W.T)) , (S - eps * np.dot(W, W.T)) ))
        else:
            if (typ == 'sep'):
                grad_eps = delta * (np.sum(np.multiply( (-2 * np.dot(W1, W1.T)) , (S - eps * np.dot(W1, W1.T)) ))
                + np.sum(np.multiply( (-2 * np.dot(W2, W2.T)) , (S - eps * np.dot(W2, W2.T)) ))
                )
        
        #print 'W.Wt', np.dot(W, W.T)
        #print 'grad eps', grad_eps
        #print 'eps', eps
        #print 'W', W[0]
        #print 'H', H1[0]
        eps = eps - learning_rate * grad_eps
        
        eps_list.append(eps)
        
        
        W1n = np.concatenate((W1cn,W1dn), axis = 1)
        W2n = np.concatenate((W2cn,W2dn), axis = 1)
        
        #print 'W1n', W1n[0]
        #print 'W2n', W2n[0]
        
        
        W1n[W1n<0] = 0
        H1n[H1n<0] = 0

        W2n[W2n<0] = 0
        H2n[H2n<0] = 0

        #print ('---------------------------------------------------------')
        
        #errorSqrX1 = lossfuncSqr(X1, np.dot(W1n, H1n.T) )
        #errorAbsX1 = lossfuncAbs(X1, np.dot(W1n, H1n.T) )
        errorX1 = error(X1, np.dot(W1, H1.T))
        
        #errorSqrX2 = lossfuncSqr(X2, np.dot(W2n, np.transpose(H2n)))
        #errorAbsX2 = lossfuncAbs(X2, np.dot(W2n, np.transpose(H2n)))
        errorX2 = error(X2, np.dot(W2, H2.T))
        
        #errorAbsC = lossfuncAbs(W1cn, W2cn)
        errorSqrC = lossfuncSqr(W1cn, W2cn)
        
        errorD = lossfuncD(np.transpose(W1dn), W2dn)
        
        #errorS1 = lossfuncS(S,W1)
        #errorS2 = lossfuncS(S,W2)
        
        if (typ == 'con'):
            errorS = error(S, eps * np.dot(W, W.T))
        else:
            if (typ == 'sep'):
                errorS1 = error(S, eps * np.dot(W1,W1.T))
                errorS2 = error(S, eps * np.dot(W2,W2.T))
        
        #print "Iteration %s" %(e)
        #print "X1: %s, X2: %s, S1: %s, S2: %s" %(errorSqrX1, errorSqrX2, errorS1, errorS2)
        #print "-----"*10
        
        
        index.append(e)
        
        #errSqrX1.append(errorSqrX1)
        #errAbsX1.append(errorAbsX1)
        errX1.append(errorX1)

        #errSqrX2.append(errorSqrX2)
        #errAbsX2.append(errorAbsX2)
        errX2.append(errorX2)
        
        errX1X2.append((errorX1 + errorX2)/2)
        
        #errAbsC.append(errorAbsC)
        errSqrC.append(errorSqrC)
        
        errD.append(errorD)
        
        if (typ == 'con'):
            errS.append(errorS)
        else:
            if (typ == 'sep'):
                errS1.append(errorS1)
                errS2.append(errorS2)

        W1 = W1n
        H1 = H1n
        
        W2 = W2n
        H2 = H2n
        
        #print e
        
        if (e % 10 == 0):
            print (e)
            
        
    
    #mode = 'test'
    mode = 'write'
    
    if (mode == 'write'):        
        if (os.path.isdir(path) == False):
            os.mkdir(path)
        pathk = path + "k" + str(k)
        if (os.path.isdir(pathk) == False):
            os.mkdir(pathk)
        
        pathkc = pathk + "/c" + str(kc) + "d" + str(k-kc)    
        if (os.path.isdir(pathkc) == False):
            os.mkdir(pathkc)
        
        print 'Mode write'
        
        np.savetxt(pathkc + "/W1.csv", W1, delimiter=",")
        np.savetxt(pathkc + "/W2.csv", W2, delimiter=",")
        
        np.savetxt(pathkc + "/W1c.csv", W1c, delimiter=",")
        np.savetxt(pathkc + "/W2c.csv", W2c, delimiter=",")
        np.savetxt(pathkc + "/W1d.csv", W1d, delimiter=",")
        np.savetxt(pathkc + "/W2d.csv", W2d, delimiter=",")
        
        np.savetxt(pathkc + "/H1.csv", H1, delimiter=",")
        np.savetxt(pathkc + "/H2.csv", H2, delimiter=",")
        
        np.savetxt(pathkc + "/H1c.csv", H1c, delimiter=",")
        np.savetxt(pathkc + "/H2c.csv", H2c, delimiter=",")
        np.savetxt(pathkc + "/H1d.csv", H1d, delimiter=",")
        np.savetxt(pathkc + "/H2d.csv", H2d, delimiter=",")
        
        
        fw = open(pathkc + '/err.txt', "w")
        
        fwx1 = open(pathkc + '/errors-X1.txt', "w")
        fwx2 = open(pathkc + '/errors-X2.txt', "w")
        
        fwe = open(pathkc + '/eps.txt', "w")
            
        for i in errX1:
            fwx1.write(str(i) + '\n')
                
        for i in errX2:
            fwx2.write(str(i) + '\n')
            
        for i in eps_list:
            fwe.write(str(i) + '\n')
            
        errX1mae = errorMAE(X1, np.dot(W1, H1.T))
        errX2mae = errorMAE(X2, np.dot(W2, H2.T))
            
        #fw.write("Error X1 RMSE: " + str(errX1[-1]) + " MAE: " + str(errX1mae) + "\n")
        #fw.write("Error X2 RMSE: " + str(errX2[-1]) + " MAE: " + str(errX2mae) +  "\n")    
        
        fw.write("Error X1 RMSE: " + str(errX1[-1]) + " MAE: " + str(errX1mae) + "\n")
        fw.write("Error X2 RMSE: " + str(errX2[-1]) + " MAE: " + str(errX2mae) + "\n")
        fw.write("Error X1+X2: " + str(errX1X2[-1]) + "\n")
        fw.write("Error C: " + str(errSqrC[-1]) + "\n")
        fw.write("Error D: " + str(errD[-1]) + "\n")
        if (typ == 'con'):
            fw.write("Error S: " + str(errS[-1]) + "\n")
        else:
            if (typ == 'sep'):
                fw.write("Error S1: " + str(errS1[-1]) + "\n")
                fw.write("Error S2: " + str(errS2[-1]) + "\n")
        fw.write("Eps: " + str(eps))
        
        errorX1 = np.abs(X1 - np.dot(W1,np.transpose(H1)))
        errorX2 = np.abs(X2 - np.dot(W2,np.transpose(H2)))
        
        #np.savetxt(pathkc + "/ErrorX1.csv", errorX1, delimiter=",")
        #np.savetxt(pathkc + "/ErrorX2.csv", errorX2, delimiter=",")
            
        fw.close()
        fwx1.close()
        fwx2.close()
        fwe.close()
        
        pathk = path + str(k)
        
        err1 = error(X1, np.dot(W1, H1.T))
        fw1 = open(pathk + 'err1.txt', "w")
        fw1.write(str(err1))
        fw1.close()
        
        err2 = error(X2, np.dot(W2, H2.T))
        fw2 = open(pathk + 'err2.txt', "w")
        fw2.write(str(err2))
        fw2.close()
        
        if (typ == 'con'):
            errs = error(S, np.dot(W,W.T))
            fw3 = open(pathk + 'errs.txt', "w")
            fw3.write(str(errs))
            fw3.close()
        else:
            if (typ == 'sep'):
                errs1 = error(S, np.dot(W1,W1.T))
                fw3 = open(pathk + 'errs2.txt', "w")
                fw3.write(str(errs1))
                fw3.close()
                
                errs2 = error(S, np.dot(W2,W2.T))
                fw4 = open(pathk + 'errs1.txt', "w")
                fw4.write(str(errs2))
                fw4.close()
        

    pathk = path + "k" + str(k)
    pathkc = pathk + "/c" + str(kc) + "d" + str(k-kc)
    print (pathkc)
    
    
    #mode = 'test'
    mode = 'write'
    
    sns.set()
    
    plt.figure()
    plt.plot(index,eps_list)
    #plt.title('Error eps')
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    if mode == 'write':
        plt.savefig(pathkc + "/Eps.png")
        plt.savefig(pathkc + "/Eps.pdf")
        #plt.clf()
        #plt.close()
    
    plt.figure()
    plt.plot(index,errX1)
    #plt.title('Error X1')
    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction Error')
    if mode == 'write':
        plt.savefig(pathkc + "/ErrorX1.png")
        plt.savefig(pathkc + "/ErrorX1.pdf")
        #plt.clf()
        #plt.close()
    
    plt.figure()
    plt.plot(index,errX2)
    #plt.title('Error X2')
    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction Error')
    if mode == 'write':
        plt.savefig(pathkc + "/ErrorX2.png")
        plt.savefig(pathkc + "/ErrorX2.pdf")
        #plt.clf()
        #plt.close()
    
    plt.figure()
    plt.plot(index,errSqrC)
    #plt.title('Error C')
    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction Error')
    if mode == 'write':
        plt.savefig(pathkc + "/ErrorC.png")
        plt.savefig(pathkc + "/ErrorC.pdf")
        #plt.clf()
        #plt.close()
    
    plt.figure()
    plt.plot(index,errD)
    #plt.title('Error D')
    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction Error')
    if mode == 'write':
        plt.savefig(pathkc + "/ErrorD.png")
        plt.savefig(pathkc + "/ErrorD.pdf")
        #plt.clf()
        #plt.close()
    
    
    if (typ == 'con'):
        plt.figure()
        plt.plot(index,errS)
        #plt.title('Error S')
        plt.xlabel('Iteration')
        plt.ylabel('Reconstruction Error')
        if mode == 'write':
            plt.savefig(pathkc + "/ErrorS.png")
            plt.savefig(pathkc + "/ErrorS.pdf")
            #plt.clf()
            #plt.close()
    else:
        if (typ == 'sep'):
            plt.figure()
            plt.plot(index,errS1)
            #plt.title('Error S1')
            plt.xlabel('Iteration')
            plt.ylabel('Reconstruction Error')
            if mode == 'write':
                plt.savefig(pathkc + "/ErrorS1.png")
                plt.clf()
                plt.close()
            
            plt.figure()
            plt.plot(index,errS2)
            #plt.title('Error S2')
            plt.xlabel('Iteration')
            plt.ylabel('Reconstruction Error')
            if mode == 'write':
                plt.savefig(pathkc + "/ErrorS2.png")
                plt.clf()
                plt.close()
                
    print ('plot')
    print eps
    plt.show()
    plt.rcParams.update({'font.size': 10})
    


    '''
    plt.figure(4)
    plt.plot(index,errS2)
    plt.title('Error S2')
    plt.xlabel('Iteration')
    if mode == 'write':
        plt.savefig(pathkc + "/ErrorS2.png")
    '''

    '''
    #fig = plt.figure()
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    #matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')
    plt.rcParams.update({'font.size': 24})
    
    
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize = (5,8))
    fig.subplots_adjust(hspace = .3)
    
    #plt.figure(figsize = (10,15))
    
    ax[0].plot(index,errSqrX1,label = 'X1 Error')
    ax[0].set_xlabel('Iteration')
    ax[0].set_ylabel('Reconstruction Error')
    ax[0].legend()
    
    ax[1].plot(index,errSqrX2,label = 'X2 Error')
    ax[1].set_xlabel('Iteration')
    ax[1].set_ylabel('Reconstruction Error')
    ax[1].legend()
    
    plt.savefig(pathkc + "/X1X2V_large.pdf")
    '''

def gd_eps_no_structure(k, kc, path, alpha, beta):
    
    with open('C:/Project/EDU/files/2013/example/Topic/similarity/grid_nmp_con/l.txt') as file:
        array2dX1 = [[float(digit) for digit in line.split(',')] for line in file]
    
    X1 = np.array(array2dX1)
    
    with open('C:/Project/EDU/files/2013/example/Topic/similarity/grid_nmp_con/h.txt') as file:
        array2dX2 = [[float(digit) for digit in line.split(',')] for line in file]
    
    X2 = np.array(array2dX2)
    
    epoc = 1000
    
    m1,n1 = X1.shape
    m2,n2 = X2.shape
    
    W1 = np.random.rand(m1,k)
    H1 = np.random.rand(n1,k)
    
    W2 = np.random.rand(m2,k)
    H2 = np.random.rand(n2,k)
    
    W1 = W1 / 10.0
    H1 = H1 / 10.0
    
    W2 = W2 / 10.0
    H2 = H2 / 10.0
    
    index = []
    errX1 = []
    errX2 = []
    errX1X2 = []
    errSqrC = []
    errD = []
    
    #C
    #alpha = 0.5
    #D
    #beta = 0.5
    
    #X1X2
    gama = 2.0 - (alpha + beta)
    #gama = 0.8
    #eps = 1
    #S
    #delta = 0.5
    
    reg = 1

    for e in range(epoc):
        learning_rate = 0.02/np.sqrt(e+1)
    
        W1c = W1[:,:kc]
        W1d = W1[:,kc:]
        
        H1c = H1[:,:kc]
        H1d = H1[:,kc:]
        
        W2c = W2[:,:kc]
        W2d = W2[:,kc:]
        
        H2c = H2[:,:kc]
        H2d = H2[:,kc:]
        
        grad_w1c = (2 * gama * np.dot((np.dot(W1, H1.T) - X1), H1c)
            + 2 * alpha * (W1c - W2c)
            + 2 * reg * W1c
            )
        
        
        W1cn = W1c - learning_rate * grad_w1c
        
        grad_w2c = (2 * gama * np.dot((np.dot(W2, H2.T) - X2), H2c)
            - 2 * alpha * (W1c - W2c)
            + 2 * reg * W2c
            )
        
        W2cn = W2c - learning_rate * grad_w2c
        
        grad_w1d = (2 * gama * np.dot((np.dot(W1, H1.T) - X1), H1d) 
            + 2 * beta * np.dot(W2d, np.dot(W1d.T,W2d))
            + 2 * reg * W1d
            )
        
        W1dn = W1d - learning_rate * grad_w1d
        
        grad_w2d = (2 * gama * np.dot((np.dot(W2, H2.T) - X2), H2d) 
            + 2 * beta * np.dot(W1d, np.dot(W1d.T,W2d))
            + 2 * reg * W2d
            )
        
        W2dn = W2d - learning_rate * grad_w2d
        
        grad_h1 = -2 * gama * np.dot(np.transpose(X1 - np.dot(W1, H1.T)), W1) + 2 * reg * H1
        H1n = H1 - learning_rate * grad_h1
        
        grad_h2 = -2 * gama * np.dot(np.transpose(X2 - np.dot(W2, H2.T)), W2) + 2 * reg * H2
        H2n = H2 - learning_rate * grad_h2
        
        W1n = np.concatenate((W1cn,W1dn), axis = 1)
        W2n = np.concatenate((W2cn,W2dn), axis = 1)
        
        W1n[W1n<0] = 0
        H1n[H1n<0] = 0

        W2n[W2n<0] = 0
        H2n[H2n<0] = 0

        errorX1 = error(X1, np.dot(W1, H1.T))
        errorX2 = error(X2, np.dot(W2, H2.T))
        errorSqrC = lossfuncSqr(W1cn, W2cn)
        errorD = lossfuncD(np.transpose(W1dn), W2dn)
                
        index.append(e)
        
        errX1.append(errorX1)
        errX2.append(errorX2)
        errX1X2.append((errorX1 + errorX2)/2)
        errSqrC.append(errorSqrC)
        errD.append(errorD)
        
        W1 = W1n
        H1 = H1n
        
        W2 = W2n
        H2 = H2n
        
        if (e % 10 == 0):
            print (e)
            
    #mode = 'test'
    mode = 'write'
    
    if (mode == 'write'):        
        if (os.path.isdir(path) == False):
            os.mkdir(path)
        pathk = path + "k" + str(k)
        if (os.path.isdir(pathk) == False):
            os.mkdir(pathk)
        
        pathkc = pathk + "/c" + str(kc) + "d" + str(k-kc)    
        if (os.path.isdir(pathkc) == False):
            os.mkdir(pathkc)
        
        print 'Mode write'
        
        np.savetxt(pathkc + "/W1.csv", W1, delimiter=",")
        np.savetxt(pathkc + "/W2.csv", W2, delimiter=",")
        
        np.savetxt(pathkc + "/W1c.csv", W1c, delimiter=",")
        np.savetxt(pathkc + "/W2c.csv", W2c, delimiter=",")
        np.savetxt(pathkc + "/W1d.csv", W1d, delimiter=",")
        np.savetxt(pathkc + "/W2d.csv", W2d, delimiter=",")
        
        np.savetxt(pathkc + "/H1.csv", H1, delimiter=",")
        np.savetxt(pathkc + "/H2.csv", H2, delimiter=",")
        
        np.savetxt(pathkc + "/H1c.csv", H1c, delimiter=",")
        np.savetxt(pathkc + "/H2c.csv", H2c, delimiter=",")
        np.savetxt(pathkc + "/H1d.csv", H1d, delimiter=",")
        np.savetxt(pathkc + "/H2d.csv", H2d, delimiter=",")
        
        
        fw = open(pathkc + '/err.txt', "w")
        
        fwx1 = open(pathkc + '/errors-X1.txt', "w")
        fwx2 = open(pathkc + '/errors-X2.txt', "w")
        
        for i in errX1:
            fwx1.write(str(i) + '\n')
                
        for i in errX2:
            fwx2.write(str(i) + '\n')
        
        errX1mae = errorMAE(X1, np.dot(W1, H1.T))
        errX2mae = errorMAE(X2, np.dot(W2, H2.T))
        
        fw.write("Error X1 RMSE: " + str(errX1[-1]) + " MAE: " + str(errX1mae) + "\n")
        fw.write("Error X2 RMSE: " + str(errX2[-1]) + " MAE: " + str(errX2mae) + "\n")
        fw.write("Error X1+X2: " + str(errX1X2[-1]) + "\n")
        fw.write("Error C: " + str(errSqrC[-1]) + "\n")
        fw.write("Error D: " + str(errD[-1]) + "\n")
        
        errorX1 = np.abs(X1 - np.dot(W1,np.transpose(H1)))
        errorX2 = np.abs(X2 - np.dot(W2,np.transpose(H2)))
            
        fw.close()
        fwx1.close()
        fwx2.close()
        
        pathk = path + str(k)
        
        err1 = error(X1, np.dot(W1, H1.T))
        fw1 = open(pathk + 'err1.txt', "w")
        fw1.write(str(err1))
        fw1.close()
        
        err2 = error(X2, np.dot(W2, H2.T))
        fw2 = open(pathk + 'err2.txt', "w")
        fw2.write(str(err2))
        fw2.close()


    pathk = path + "k" + str(k)
    pathkc = pathk + "/c" + str(kc) + "d" + str(k-kc)
    print (pathkc)
    
    
    #mode = 'test'
    mode = 'write'
    
    sns.set()
    
    plt.figure()
    plt.plot(index,errX1)
    plt.title('Error X1')
    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction Error')
    if mode == 'write':
        plt.savefig(pathkc + "/ErrorX1.png")
        plt.savefig(pathkc + "/ErrorX1.pdf")
        #plt.clf()
        #plt.close()
    
    plt.figure()
    plt.plot(index,errX2)
    plt.title('Error X2')
    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction Error')
    if mode == 'write':
        plt.savefig(pathkc + "/ErrorX2.png")
        plt.savefig(pathkc + "/ErrorX2.pdf")
        #plt.clf()
        #plt.close()
    
    plt.figure()
    plt.plot(index,errSqrC)
    plt.title('Error C')
    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction Error')
    if mode == 'write':
        plt.savefig(pathkc + "/ErrorC.png")
        plt.savefig(pathkc + "/ErrorC.pdf")
        #plt.clf()
        #plt.close()
    
    plt.figure()
    plt.plot(index,errD)
    plt.title('Error D')
    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction Error')
    if mode == 'write':
        plt.savefig(pathkc + "/ErrorD.png")
        plt.savefig(pathkc + "/ErrorD.pdf")
        #plt.clf()
        #plt.close()
    
    
    print ('plot')
    
    plt.show()
    plt.rcParams.update({'font.size': 10})
    

def gd(typ, k, kc, path, alpha, beta, delta):
    
    
    with open(path[0:-2] + 'l.txt') as file:
        array2dX1 = [[float(digit) for digit in line.split(',')] for line in file]
    
    X1 = np.array(array2dX1)
    
    
    with open(path[0:-2] + 'h.txt') as file:
        array2dX2 = [[float(digit) for digit in line.split(',')] for line in file]
    
    X2 = np.array(array2dX2)
    
    
    with open(path[0:-2] + 'nms.csv') as file:
        arrayS = [[float(digit) for digit in line.split(',')] for line in file]
        
    S = np.array(arrayS)
    
    epoc = 200
    
    m1,n1 = X1.shape
    m2,n2 = X2.shape
    
    W1 = np.random.rand(m1,k)
    H1 = np.random.rand(n1,k)
    
    W2 = np.random.rand(m2,k)
    H2 = np.random.rand(n2,k)
    
    
    W1 = W1 / 10.0
    H1 = H1 / 10.0
    
    W2 = W2 / 10.0
    H2 = H2 / 10.0
    
    index = []
    errX1 = []
    errX2 = []
    errX1X2 = []
    errSqrC = []
    errD = []
        
    errS1 = []
    errS2 = []
    errS = []
    
    
    #C
    #alpha = 0.5
    #D
    #beta = 0.5
    
    #X1X2
    gama = 2.0 - (alpha + beta)
    #gama = 0.8
    
    #S
    #delta = 0.5
    
    reg = 1

    for e in range(epoc):
        learning_rate = 0.01/np.sqrt(e+1)
    
        W1c = W1[:,:kc]
        W1d = W1[:,kc:]
        
        H1c = H1[:,:kc]
        H1d = H1[:,kc:]
        
        W2c = W2[:,:kc]
        W2d = W2[:,kc:]
        
        H2c = H2[:,:kc]
        H2d = H2[:,kc:]
        
        #Con_sim II        
        if (typ == 'con'):
            W = np.concatenate((W1, W2), axis = 1)
        
        
        if (typ == 'con'):
            grad_w1c = (2 * gama * np.dot((np.dot(W1, H1.T) - X1), H1c)
            + 2 * alpha * (W1c - W2c)
            + 2 * reg * W1c
            - 4 * delta * np.dot((S - np.dot(W, W.T)), W1c)
            ) 
        else:
            if (typ == 'sep'):
                grad_w1c = (2 * gama * np.dot((np.dot(W1, H1.T) - X1), H1c)
                + 2 * alpha * (W1c - W2c)
                + 2 * reg * W1c
                - 4 * delta * np.dot((S - np.dot(W1, W1.T)), W1c)
                )
                
            
        W1cn = W1c - learning_rate * grad_w1c
        
        if (typ == 'con'):
            grad_w2c = (2 * gama * np.dot((np.dot(W2, H2.T) - X2), H2c)
            - 2 * alpha * (W1c - W2c)
            + 2 * reg * W2c
            - 4 * delta * np.dot((S - np.dot(W, W.T)), W2c)
            )
        else:
            if (typ == 'sep'):
                grad_w2c = (2 * gama * np.dot((np.dot(W2, H2.T) - X2), H2c)
                - 2 * alpha * (W1c - W2c)
                + 2 * reg * W2c
                - 4 * delta * np.dot((S - np.dot(W2, W2.T)), W2c)
                )
            
        W2cn = W2c - learning_rate * grad_w2c
        
        if (typ == 'con'):    
            grad_w1d = (2 * np.dot((np.dot(W1, H1.T) - X1), H1d) 
            + 2 * beta * np.dot(W2d, np.dot(W1d.T,W2d))
            + 2 * reg * W1d
            - 4 * delta * np.dot((S - np.dot(W, W.T)), W1d)
            )
        else:
            if (typ == 'sep'):
                grad_w1d = (2 * np.dot((np.dot(W1, H1.T) - X1), H1d) 
                + 2 * beta * np.dot(W2d, np.dot(W1d.T,W2d))
                + 2 * reg * W1d
                - 4 * delta * np.dot((S - np.dot(W1, W1.T)), W1d)
                )
        
        W1dn = W1d - learning_rate * grad_w1d
        
        
        if (typ == 'con'):
            grad_w2d = (2 * np.dot((np.dot(W2, H2.T) - X2), H2d) 
            + 2 * beta * np.dot(W1d, np.dot(W1d.T,W2d))
            + 2 * reg * W2d
            - 4 * delta * np.dot((S - np.dot(W, W.T)), W2d)
            )
        else:
            if (typ == 'sep'):
                grad_w2d = (2 * np.dot((np.dot(W2, H2.T) - X2), H2d) 
                + 2 * beta * np.dot(W1d, np.dot(W1d.T,W2d))
                + 2 * reg * W2d
                - 4 * delta * np.dot((S - np.dot(W2, W2.T)), W2d)
                )
                
        
        W2dn = W2d - learning_rate * grad_w2d
        
        grad_h1 = -2 * gama * np.dot(np.transpose(X1 - np.dot(W1, H1.T)), W1) + 2 * reg * H1
        H1n = H1 - learning_rate * grad_h1
        
        grad_h2 = -2 * gama * np.dot(np.transpose(X2 - np.dot(W2, H2.T)), W2) + 2 * reg * H2
        H2n = H2 - learning_rate * grad_h2
        
        
        W1n = np.concatenate((W1cn,W1dn), axis = 1)
        W2n = np.concatenate((W2cn,W2dn), axis = 1)
        
        W1n[W1n<0] = 0
        H1n[H1n<0] = 0

        W2n[W2n<0] = 0
        H2n[H2n<0] = 0

        #print ('---------------------------------------------------------')
        
        #errorSqrX1 = lossfuncSqr(X1, np.dot(W1n, H1n.T) )
        #errorAbsX1 = lossfuncAbs(X1, np.dot(W1n, H1n.T) )
        errorX1 = error(X1, np.dot(W1, H1.T))
        
        #errorSqrX2 = lossfuncSqr(X2, np.dot(W2n, np.transpose(H2n)))
        #errorAbsX2 = lossfuncAbs(X2, np.dot(W2n, np.transpose(H2n)))
        errorX2 = error(X2, np.dot(W2, H2.T))
        
        #errorAbsC = lossfuncAbs(W1cn, W2cn)
        errorSqrC = lossfuncSqr(W1cn, W2cn)
        
        errorD = lossfuncD(np.transpose(W1dn), W2dn)
        
        #errorS1 = lossfuncS(S,W1)
        #errorS2 = lossfuncS(S,W2)
        
        if (typ == 'con'):
            errorS = error(S, np.dot(W, W.T))
        else:
            if (typ == 'sep'):
                errorS1 = error(S, np.dot(W1,W1.T))
                errorS2 = error(S, np.dot(W2,W2.T))
        
        #print "Iteration %s" %(e)
        #print "X1: %s, X2: %s, S1: %s, S2: %s" %(errorSqrX1, errorSqrX2, errorS1, errorS2)
        #print "-----"*10
        
        
        index.append(e)
        
        #errSqrX1.append(errorSqrX1)
        #errAbsX1.append(errorAbsX1)
        errX1.append(errorX1)

        #errSqrX2.append(errorSqrX2)
        #errAbsX2.append(errorAbsX2)
        errX2.append(errorX2)
        
        errX1X2.append((errorX1 + errorX2)/2)
        
        #errAbsC.append(errorAbsC)
        errSqrC.append(errorSqrC)
        
        errD.append(errorD)
        
        if (typ == 'con'):
            errS.append(errorS)
        else:
            if (typ == 'sep'):
                errS1.append(errorS1)
                errS2.append(errorS2)

        W1 = W1n
        H1 = H1n
        
        W2 = W2n
        H2 = H2n
        
        #print e
        
        if (e % 10 == 0):
            print (e)
            
        
    
    mode = 'test'
    if (mode == 'write'):        
        if (os.path.isdir(path) == False):
            os.mkdir(path)
        pathk = path + "k" + str(k)
        if (os.path.isdir(pathk) == False):
            os.mkdir(pathk)
        
        pathkc = pathk + "/c" + str(kc) + "d" + str(k-kc)    
        if (os.path.isdir(pathkc) == False):
            os.mkdir(pathkc)
        
        
        np.savetxt(pathkc + "/W1.csv", W1, delimiter=",")
        np.savetxt(pathkc + "/W2.csv", W2, delimiter=",")
        
        np.savetxt(pathkc + "/W1c.csv", W1c, delimiter=",")
        np.savetxt(pathkc + "/W2c.csv", W2c, delimiter=",")
        np.savetxt(pathkc + "/W1d.csv", W1d, delimiter=",")
        np.savetxt(pathkc + "/W2d.csv", W2d, delimiter=",")
        
        np.savetxt(pathkc + "/H1.csv", H1, delimiter=",")
        np.savetxt(pathkc + "/H2.csv", H2, delimiter=",")
        
        np.savetxt(pathkc + "/H1c.csv", H1c, delimiter=",")
        np.savetxt(pathkc + "/H2c.csv", H2c, delimiter=",")
        np.savetxt(pathkc + "/H1d.csv", H1d, delimiter=",")
        np.savetxt(pathkc + "/H2d.csv", H2d, delimiter=",")
        
        
        fw = open(pathkc + '/err.txt', "w")
        
        fw.write("Error X1: " + str(errX1[-1]) + "\n")
        fw.write("Error X2: " + str(errX2[-1]) + "\n")
        fw.write("Error X1+X2: " + str(errX1X2[-1]) + "\n")
        fw.write("Error C: " + str(errSqrC[-1]) + "\n")
        fw.write("Error D: " + str(errD[-1]) + "\n")
        if (typ == 'con'):
            fw.write("Error S: " + str(errS[-1]))
        else:
            if (typ == 'sep'):
                fw.write("Error S1: " + str(errS1[-1]) + "\n")
                fw.write("Error S2: " + str(errS2[-1]))
        
        
        errorX1 = np.abs(X1 - np.dot(W1,np.transpose(H1)))
        errorX2 = np.abs(X2 - np.dot(W2,np.transpose(H2)))
        
        #np.savetxt(pathkc + "/ErrorX1.csv", errorX1, delimiter=",")
        #np.savetxt(pathkc + "/ErrorX2.csv", errorX2, delimiter=",")
            
        fw.close()
        
        pathk = path + str(k)
        
        err1 = error(X1, np.dot(W1, H1.T))
        fw1 = open(pathk + 'err1.txt', "w")
        fw1.write(str(err1))
        fw1.close()
        
        err2 = error(X2, np.dot(W2, H2.T))
        fw2 = open(pathk + 'err2.txt', "w")
        fw2.write(str(err2))
        fw2.close()
        
        if (typ == 'con'):
            errs = error(S, np.dot(W,W.T))
            fw3 = open(pathk + 'errs.txt', "w")
            fw3.write(str(errs))
            fw3.close()
        else:
            if (typ == 'sep'):
                errs1 = error(S, np.dot(W1,W1.T))
                fw3 = open(pathk + 'errs2.txt', "w")
                fw3.write(str(errs1))
                fw3.close()
                
                errs2 = error(S, np.dot(W2,W2.T))
                fw4 = open(pathk + 'errs1.txt', "w")
                fw4.write(str(errs2))
                fw4.close()
        

    '''
    W1c = W1[:,:kc]
    W1d = W1[:,kc:]
        
    H1c = H1[:,:kc]
    H1d = H1[:,kc:]
        
    W2c = W2[:,:kc]
    W2d = W2[:,kc:]
        
    H2c = H2[:,:kc]
    H2d = H2[:,kc:]
    
    np.savetxt(path + "/W1c.csv", W1c, delimiter=",")
    np.savetxt(path + "/W1d.csv", W1d, delimiter=",")
        
    np.savetxt(path + "/H1c.csv", H1c, delimiter=",")
    np.savetxt(path + "/H1d.csv", H1d, delimiter=",")
    
    np.savetxt(path + "/W2c.csv", W2c, delimiter=",")
    np.savetxt(path + "/W2d.csv", W2d, delimiter=",")
        
    np.savetxt(path + "/H2c.csv", H2c, delimiter=",")
    np.savetxt(path + "/H2d.csv", H2d, delimiter=",")
    
    np.savetxt(path + "/W1.csv", W1, delimiter=",")
    np.savetxt(path + "/W2.csv", W2, delimiter=",")
        
    np.savetxt(path + "/H1.csv", H1, delimiter=",")
    np.savetxt(path + "/H2.csv", H2, delimiter=",")
        
    '''
    pathk = path + "k" + str(k)
    pathkc = pathk + "/c" + str(kc) + "d" + str(k-kc)
    print (pathkc)
    
    
    mode = 'test'
    
    plt.figure()
    plt.plot(index,errX1)
    plt.title('Error X1')
    plt.xlabel('Iteration')
    if mode == 'write':
        plt.savefig(pathkc + "/ErrorX1.png")
        plt.clf()
        plt.close()
    
    plt.figure()
    plt.plot(index,errX2)
    plt.title('Error X2')
    plt.xlabel('Iteration')
    if mode == 'write':
        plt.savefig(pathkc + "/ErrorX2.png")
        plt.clf()
        plt.close()
    
    plt.figure()
    plt.plot(index,errSqrC)
    plt.title('Error C')
    plt.xlabel('Iteration')
    if mode == 'write':
        plt.savefig(pathkc + "/ErrorC.png")
        plt.clf()
        plt.close()
    
    plt.figure()
    plt.plot(index,errD)
    plt.title('Error D')
    plt.xlabel('Iteration')
    if mode == 'write':
        plt.savefig(pathkc + "/ErrorD.png")
        plt.clf()
        plt.close()
    
    
    if (typ == 'con'):
        plt.figure()
        plt.plot(index,errS)
        plt.title('Error S')
        plt.xlabel('Iteration')
        if mode == 'write':
            plt.savefig(pathkc + "/ErrorS.png")
            plt.clf()
            plt.close()
    else:
        if (typ == 'sep'):
            plt.figure()
            plt.plot(index,errS1)
            plt.title('Error S1')
            plt.xlabel('Iteration')
            if mode == 'write':
                plt.savefig(pathkc + "/ErrorS1.png")
                plt.clf()
                plt.close()
            
            plt.figure()
            plt.plot(index,errS2)
            plt.title('Error S2')
            plt.xlabel('Iteration')
            if mode == 'write':
                plt.savefig(pathkc + "/ErrorS2.png")
                plt.clf()
                plt.close()
                
    print ('plot')
    plt.show()
    plt.rcParams.update({'font.size': 10})
    


    '''
    plt.figure(4)
    plt.plot(index,errS2)
    plt.title('Error S2')
    plt.xlabel('Iteration')
    if mode == 'write':
        plt.savefig(pathkc + "/ErrorS2.png")
    '''

    '''
    #fig = plt.figure()
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    #matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')
    plt.rcParams.update({'font.size': 24})
    
    
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize = (5,8))
    fig.subplots_adjust(hspace = .3)
    
    #plt.figure(figsize = (10,15))
    
    ax[0].plot(index,errSqrX1,label = 'X1 Error')
    ax[0].set_xlabel('Iteration')
    ax[0].set_ylabel('Reconstruction Error')
    ax[0].legend()
    
    ax[1].plot(index,errSqrX2,label = 'X2 Error')
    ax[1].set_xlabel('Iteration')
    ax[1].set_ylabel('Reconstruction Error')
    ax[1].legend()
    
    plt.savefig(pathkc + "/X1X2V_large.pdf")
    '''
    

def gd_grid(k, kc, path, alpha, beta, delta):
    
    with open(path[0:-14] + '/l.txt') as file:
        array2dX1 = [[float(digit) for digit in line.split('\t')] for line in file]
    
    X1 = np.array(array2dX1)
    
    with open(path[0:-14] + '/h.txt') as file:
        array2dX2 = [[float(digit) for digit in line.split('\t')] for line in file]
    
    X2 = np.array(array2dX2)
    
    with open(path[0:-14] + '/normal_mp.csv') as file:
        arrayS = [[float(digit) for digit in line.split(',')] for line in file]
        
    S = np.array(arrayS)
    
    epoc = 200
    
    m1,n1 = X1.shape
    m2,n2 = X2.shape
    
    W1 = np.random.rand(m1,k)
    H1 = np.random.rand(n1,k)
    
    W2 = np.random.rand(m2,k)
    H2 = np.random.rand(n2,k)
    
    W1 = W1 / 10.0
    H1 = H1 / 10.0
    
    W2 = W2 / 10.0
    H2 = H2 / 10.0
    
    index = []
    errX1 = []
    errX2 = []
    errX1X2 = []
    errSqrC = []
    errD = []
    errS1 = []
    errS2 = []
    
    #C
    #alpha = 0.6
    #D
    #beta = 0.3
    
    #X1X2
    gama = 2.0 - (alpha + beta)
    #gama = 0.8
    
    #S
    #delta = 0.1
    
    reg = 1

    for e in range(epoc):
        learning_rate = 0.001/np.sqrt(e+1)
    
        W1c = W1[:,:kc]
        W1d = W1[:,kc:]
        
        H1c = H1[:,:kc]
        H1d = H1[:,kc:]
        
        W2c = W2[:,:kc]
        W2d = W2[:,kc:]
        
        H2c = H2[:,:kc]
        H2d = H2[:,kc:]
        
        #Con_sim II        
        #W = np.concatenate((W1, W2), axis = 1)
        
        grad_w1c = (2 * gama * np.dot((np.dot(W1, H1.T) - X1), H1c)
        + 2 * alpha * (W1c - W2c)
        + 2 * reg * W1c
        #Seperate W1 and W2
        - 4 * delta * np.dot((S - np.dot(W1, W1.T)), W1c)
        #Con_sim II        
        #- 4 * delta * np.dot((S - np.dot(W, W.T)), W1c)
        )
        
        W1cn = W1c - learning_rate * grad_w1c
        
        grad_w2c = (2 * gama * np.dot((np.dot(W2, H2.T) - X2), H2c)
        - 2 * alpha * (W1c - W2c)
        + 2 * reg * W2c
        #Seperate W1 and W2
        - 4 * delta * np.dot((S - np.dot(W2, W2.T)), W2c)
        #Con_sim II        
        #- 4 * delta * np.dot((S - np.dot(W, W.T)), W2c)
        )
        
        W2cn = W2c - learning_rate * grad_w2c
        
        grad_w1d = (2 * np.dot((np.dot(W1, H1.T) - X1), H1d) 
        + 2 * beta * np.dot(W2d, np.dot(W1d.T,W2d))
        + 2 * reg * W1d
        #Seperate W1 and W2
        - 4 * delta * np.dot((S - np.dot(W1, W1.T)), W1d)
        #Con_sim II        
        #- 4 * delta * np.dot((S - np.dot(W, W.T)), W1d)
        )
        
        W1dn = W1d - learning_rate * grad_w1d
        
        grad_w2d = (2 * np.dot((np.dot(W2, H2.T) - X2), H2d) 
        + 2 * beta * np.dot(W1d, np.dot(W1d.T,W2d))
        + 2 * reg * W2d
        #Seperate W1 and W2
        - 4 * delta * np.dot((S - np.dot(W2, W2.T)), W2d)
        #Con_sim II        
        #- 4 * delta * np.dot((S - np.dot(W, W.T)), W2d)
        )
        
        W2dn = W2d - learning_rate * grad_w2d
        
        grad_h1 = -2 * gama * np.dot(np.transpose(X1 - np.dot(W1, H1.T)), W1) + 2 * reg * H1
        H1n = H1 - learning_rate * grad_h1
        
        grad_h2 = -2 * gama * np.dot(np.transpose(X2 - np.dot(W2, H2.T)), W2) + 2 * reg * H2
        H2n = H2 - learning_rate * grad_h2

        
        W1n = np.concatenate((W1cn,W1dn), axis = 1)
        W2n = np.concatenate((W2cn,W2dn), axis = 1)
        
        W1n[W1n<0] = 0
        H1n[H1n<0] = 0

        W2n[W2n<0] = 0
        H2n[H2n<0] = 0

        #print ('---------------------------------------------------------')
        
        #errorSqrX1 = lossfuncSqr(X1, np.dot(W1n, H1n.T) )
        #errorAbsX1 = lossfuncAbs(X1, np.dot(W1n, H1n.T) )
        errorX1 = error(X1, np.dot(W1, H1.T))
        
        #errorSqrX2 = lossfuncSqr(X2, np.dot(W2n, np.transpose(H2n)))
        #errorAbsX2 = lossfuncAbs(X2, np.dot(W2n, np.transpose(H2n)))
        errorX2 = error(X2, np.dot(W2, H2.T))
        
        #errorAbsC = lossfuncAbs(W1cn, W2cn)
        errorSqrC = lossfuncSqr(W1cn, W2cn)
        
        errorD = lossfuncD(np.transpose(W1dn), W2dn)
        
        
        errorS1 = error(S, np.dot(W1,W1.T))
        errorS2 = error(S, np.dot(W2,W2.T))
        
        #print "Iteration %s" %(e)
        #print "X1: %s, X2: %s, S1: %s, S2: %s" %(errorSqrX1, errorSqrX2, errorS1, errorS2)
        #print "-----"*10
        
        
        index.append(e)
        
        #errSqrX1.append(errorSqrX1)
        #errAbsX1.append(errorAbsX1)
        errX1.append(errorX1)

        #errSqrX2.append(errorSqrX2)
        #errAbsX2.append(errorAbsX2)
        errX2.append(errorX2)
        
        errX1X2.append((errorX1 + errorX2)/2)
        
        #errAbsC.append(errorAbsC)
        errSqrC.append(errorSqrC)
        
        errD.append(errorD)
        
        errS1.append(errorS1)
        errS2.append(errorS2)

        W1 = W1n
        H1 = H1n
        
        W2 = W2n
        H2 = H2n
        
        #print e
        
        if (e % 10 == 0):
            print (e)
            
        
    
    mode = 'write'
    if (mode == 'write'):        
        if (os.path.isdir(path) == False):
            os.mkdir(path)
        pathk = path + "k" + str(k)
        if (os.path.isdir(pathk) == False):
            os.mkdir(pathk)
        
        pathkc = pathk + "/c" + str(kc) + "d" + str(k-kc)    
        if (os.path.isdir(pathkc) == False):
            os.mkdir(pathkc)
        '''
        np.savetxt(pathkc + "/W1.csv", W1, delimiter=",")
        np.savetxt(pathkc + "/W2.csv", W2, delimiter=",")
        
        np.savetxt(pathkc + "/W1c.csv", W1c, delimiter=",")
        np.savetxt(pathkc + "/W2c.csv", W2c, delimiter=",")
        np.savetxt(pathkc + "/W1d.csv", W1d, delimiter=",")
        np.savetxt(pathkc + "/W2d.csv", W2d, delimiter=",")
        
        np.savetxt(pathkc + "/H1.csv", H1, delimiter=",")
        np.savetxt(pathkc + "/H2.csv", H2, delimiter=",")
        
        np.savetxt(pathkc + "/H1c.csv", H1c, delimiter=",")
        np.savetxt(pathkc + "/H2c.csv", H2c, delimiter=",")
        np.savetxt(pathkc + "/H1d.csv", H1d, delimiter=",")
        np.savetxt(pathkc + "/H2d.csv", H2d, delimiter=",")
        '''
        fw = open(pathkc + '/err.txt', "w")
        
        fw.write("Error X1: " + str(errX1[-1]) + "\n")
        fw.write("Error X2: " + str(errX2[-1]) + "\n")
        fw.write("Error X1+X2: " + str(errX1X2[-1]) + "\n")
        fw.write("Error C: " + str(errSqrC[-1]) + "\n")
        fw.write("Error D: " + str(errD[-1]) + "\n")
        fw.write("Error S1: " + str(errS1[-1]) + "\n")
        fw.write("Error S2: " + str(errS2[-1]))
        
        fw.close()
        
        '''
        errorX1 = np.abs(X1 - np.dot(W1,np.transpose(H1)))
        errorX2 = np.abs(X2 - np.dot(W2,np.transpose(H2)))
        
        np.savetxt(pathkc + "/ErrorX1.csv", errorX1, delimiter=",")
        np.savetxt(pathkc + "/ErrorX2.csv", errorX2, delimiter=",")
        '''    
        
        pathk = path + str(k)
        
        err1 = error(X1, np.dot(W1, H1.T))
        fw1 = open(pathk + 'err1.txt', "w")
        fw1.write(str(err1))
        fw1.close()
        
        err2 = error(X2, np.dot(W2, H2.T))
        fw2 = open(pathk + 'err2.txt', "w")
        fw2.write(str(err2))
        fw2.close()
        
        errc = error(X2, np.dot(W2, H2.T))
        fwc = open(pathk + 'errc.txt', "w")
        fwc.write(str(errc))
        fwc.close()
        
        errd = lossfuncD(W1dn.T, W2dn)
        fwd = open(pathk + 'errd.txt', "w")
        fwd.write(str(errd))
        fwd.close()
        
        errs1 = error(S, np.dot(W1,W1.T))
        fw3 = open(pathk + 'errs2.txt', "w")
        fw3.write(str(errs1))
        fw3.close()
        
        errs2 = error(S, np.dot(W2,W2.T))
        fw4 = open(pathk + 'errs1.txt', "w")
        fw4.write(str(errs2))
        fw4.close()
        
    pathk = path + "k" + str(k)
    pathkc = pathk + "/c" + str(kc) + "d" + str(k-kc)
    print (pathkc)
    

    plt.figure(1)
    plt.plot(index,errS1)
    plt.title('Error S1')
    plt.xlabel('Iteration')
    if mode == 'write':
        plt.savefig(pathkc + "/ErrorS1.png")
    
    plt.figure(2)
    plt.plot(index,errS2)
    plt.title('Error S2')
    plt.xlabel('Iteration')
    if mode == 'write':
        plt.savefig(pathkc + "/ErrorS2.png")
    
    plt.figure(3)
    plt.plot(index,errX1)
    plt.title('Error X1')
    plt.xlabel('Iteration')
    if mode == 'write':
        plt.savefig(pathkc + "/ErrorX1.png")
    
    plt.figure(4)
    plt.plot(index,errX2)
    plt.title('Error X2')
    plt.xlabel('Iteration')
    if mode == 'write':
        plt.savefig(pathkc + "/ErrorX2.png")

    #plt.show()


def error(A, B):
    return np.sqrt(np.mean((A - B) ** 2))

def errorMAE(A, B):
    return np.mean(np.abs(A - B))
    

def lossfuncSqr(X,Xn):
    m,n = X.shape
    sum = 0
    e = 0.0
    for i in range(m):
        for j in range(n):
            sum += math.pow(X[i,j] - Xn[i,j], 2)
            e += 1
    return math.sqrt(sum / e)
    
def lossfuncAbs(X,Xn):
    m,n = X.shape
    sum = 0
    e = 0.0
    for i in range(m):
        for j in range(n):
            sum += abs(X[i,j] - Xn[i,j])
            e += 1
    return sum / e

def lossfuncD(X,Xn):
    sum = 0
    e = 0.0
    Y = np.dot(X,Xn)
    m,n = Y.shape
    for i in range(m):
        for j in range(n):
            sum += abs(Y[i,j])
            e += 1
    return sum/e

def lossfuncS(S, W):
    sum = 0
    e = 0.0
    Y = np.dot(W, W.T)
    m, n = Y.shape
    for i in range(m):
        for j in range(n):
            sum += abs(S[i,j] - Y[i,j])
            e += 1
    return sum/e

def function(X1, W1, H1, X2, W2, H2, kc, alpha, beta, gama, reg):
    W1c = W1[:,:kc]
    W1d = W1[:,kc:]
    
    W2c = W2[:,:kc]
    W2d = W2[:,kc:]
    
    H1t = np.transpose(H1)
    H2t = np.transpose(H2)
    
    v =  (gama * np.linalg.norm(X1 - np.dot(W1, H1t), ord=None, axis=None, keepdims=False) ** 2 
    + gama * np.linalg.norm(X2 - np.dot(W2, H2t), ord=None, axis=None, keepdims=False) ** 2 
    + alpha * np.linalg.norm(W1c - W2c, ord=None, axis=None, keepdims=False) ** 2 
    + beta * np.linalg.norm(np.dot(np.transpose(W1d), W2d) , ord=None, axis=None, keepdims=False) ** 2 
    + reg * np.linalg.norm(W1, ord=None, axis=None, keepdims=False) ** 2 
    + reg * np.linalg.norm(H1, ord=None, axis=None, keepdims=False) ** 2 
    + reg * np.linalg.norm(W2, ord=None, axis=None, keepdims=False) ** 2 
    + reg * np.linalg.norm(H2, ord=None, axis=None, keepdims=False) ** 2)
    '''
    print ('ALL: ')
    print (v)
    '''
    return v

def function_sep(X1, W1, H1, X2, W2, H2, kc):
    W1c = W1[:,:kc]
    W1d = W1[:,kc:]
    
    W2c = W2[:,:kc]
    W2d = W2[:,kc:]
        
    H1t = np.transpose(H1)
    H2t = np.transpose(H2)
    
    X1 = np.linalg.norm(X1 - np.dot(W1, H1t), ord=None, axis=None, keepdims=False) ** 2
    X2 = np.linalg.norm(X2 - np.dot(W2, H2t), ord=None, axis=None, keepdims=False) ** 2
    C = np.linalg.norm(W1c - W2c, ord=None, axis=None, keepdims=False) ** 2
    D = np.linalg.norm(np.dot(np.transpose(W1d), W2d) , ord=None, axis=None, keepdims=False) ** 2
    
    R = np.linalg.norm(W1, ord=None, axis=None, keepdims=False) ** 2 
    + np.linalg.norm(H1, ord=None, axis=None, keepdims=False) ** 2
    + np.linalg.norm(W2, ord=None, axis=None, keepdims=False) ** 2 
    + np.linalg.norm(H2, ord=None, axis=None, keepdims=False) ** 2
    
    
    print (X1)
    print (X2)
    print (C)
    print (D)
    print (R)
    
    return X1, X2, C , D

def function_X1X2(X1, W1, H1, X2, W2, H2, kc, alpha, beta, gama, reg):
    H1t = np.transpose(H1)
    H2t = np.transpose(H2)
    
    X1X2 = (gama *np.linalg.norm(X1 - np.dot(W1, H1t), ord=None, axis=None, keepdims=False) ** 2
    + gama * np.linalg.norm(X2 - np.dot(W2, H2t), ord=None, axis=None, keepdims=False) ** 2)
    '''
    print ('X1X2')
    print (X1X2)
    '''
    return X1X2
    
def function_C(X1, W1, H1, X2, W2, H2, kc, alpha, beta, gama, reg):
    W1c = W1[:,:kc]
    W2c = W2[:,:kc]
    
    C = alpha * np.linalg.norm(W1c - W2c, ord=None, axis=None, keepdims=False) ** 2
    '''
    print ('C')
    print (C)
    '''
    return C

def function_D(X1, W1, H1, X2, W2, H2, kc, alpha, beta, gama, reg):
    W1d = W1[:,kc:]
    W2d = W2[:,kc:]
        
    D = beta * np.linalg.norm(np.dot(np.transpose(W1d), W2d) , ord=None, axis=None, keepdims=False) ** 2
    '''
    print ('D')
    print (D)
    '''
    return D

def function_R(X1, W1, H1, X2, W2, H2, kc, alpha, beta, gama, reg):
    R = (reg * np.linalg.norm(W1, ord=None, axis=None, keepdims=False) ** 2 
    + reg * np.linalg.norm(H1, ord=None, axis=None, keepdims=False) ** 2
    + reg * np.linalg.norm(W2, ord=None, axis=None, keepdims=False) ** 2 
    + reg * np.linalg.norm(H2, ord=None, axis=None, keepdims=False) ** 2)
    '''
    print ('R')
    print (R)
    '''
    return R


def grid_search(path, typ):
    
    #a = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    a = [0.9]
    b = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    d = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    #a = [0.9]
    #b = [0.9]
    #d = [0.9]
    #r = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    
    kc = 0
    for alpha in a:
        for beta in b:
            for delta in d:
                #path = 'C:/Project/EDU/files/2013/example/Topic/similarity/grid_mp_sep/'
                pathn = path + 'a' + str(alpha) + 'b' + str(beta) + 'd' + str(delta) + '/'
                if (os.path.isdir(pathn) == False):
                    os.mkdir(pathn)
                fwerr = open(path + '/errors' + '_a' + str(alpha) + '_b' + str(beta) + '_d' + str(delta) +  '.txt', "w")
    
                for k in range(20, 21):
                    for kc in range (1, k):
                        try:
                            gd_eps(typ, k, kc, pathn, alpha, beta, delta)
                        except:
                            fwerr.write('alpha: ' + str(alpha) + '\tbeta: ' + str(beta) + '\tdelta: ' + str(delta)+ '\tk: ' + str(k) + '\tkc: ' + str(kc)  + '\n')
                    print ('K %s and Kc %s compeleted' %(k, kc))
                    
                fwerr.close()

def grid_search_constant_k(path, typ, k, kc):
    
    a = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    #a = [0.1]
    b = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    #b = [0.1]
    d = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    #d = [0.1]
    s = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    #s = [0.1]
    
    
    #a = [0.9]
    #b = [0.9]
    #d = [0.9]
    #r = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    #kc = 0
    for alpha in a:
        for beta in b:
            for delta in d:
                for sigma in s:
                    pathn = path + 'a' + str(alpha) + 'b' + str(beta) + 'd' + str(delta) + 's' + str(sigma) + '/'
                    if (os.path.isdir(pathn) == False):
                        os.mkdir(pathn)
                    #gd_eps_dual(typ, k, kc, pathn, alpha, beta, delta, sigma)
                    
                    
                    try:
                        gd_eps_dual(typ, k, kc, pathn, alpha, beta, delta, sigma)
                    except:
                        e = sys.exc_info()[0]
                        print e
                        fwerr = open(path + '/errors' + '_a' + str(alpha) + '_b' + str(beta) + '_d' + str(delta) + '_s' + str(sigma) +  '.txt', "w")
                        fwerr.write('alpha: ' + str(alpha) + '\tbeta: ' + str(beta) + '\tdelta: ' + str(delta)+ '\ts: ' + str(sigma)+ '\tk: ' + str(k) + '\tkc: ' + str(kc)  + '\n' + e)
                        fwerr.close()
                    

'''
def grid_search_oli():
    path = ''
    
    #a = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    a = [0.5]
    b = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    d = [0,1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    #a = [0.1, 0.5, 0.9]
    #a = [0.9]
    #b = [0.9]
    #d = [0.9]
    #r = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    typ = 'con'
    #typ = 'sep'
    kc = 0
    for alpha in a:
        for beta in b:
            for delta in d:
                #path = 'C:/Project/EDU/files/2013/example/Topic/similarity/grid_mp_con/'
                path = 'C:/Project/EDU/OLI_175318/lg/p_' + typ + '/'
                pathn = path + 'a' + str(alpha) + 'b' + str(beta) + 'd' + str(delta) + '/'
                os.mkdir(pathn)
                for k in range(1, 21, 10):
                    for kc in range (10, k, 10):
                        #print pathn[0:-14] + '/l.txt'
                        #gd_grid(k, kc, pathn, alpha, beta, delta)
                        gd(typ, k, kc, pathn, alpha, beta, delta)
                    print ('K %s and Kc %s compeleted' %(k, kc))
'''

'''
def iterating(path, typ, k, kc, alpha, beta, delta):
    
    for itr in range (1,10):
        #path = 'C:/Project/EDU/files/2013/example/Topic/similarity/it_mod/'
        pathn = path + str(itr) + '/'
        gd(typ, k, kc, pathn, alpha, beta, delta)
        print ('K %s and Kc %s compeleted' %(k, kc))

def iterating_oli(path,typ):
    for itr in range (1):
        pathn = path + str(itr) + '/'
        for k in range(10, 300, 10):
            for kc in range (5, k, 5):
                gd(typ, k, kc, pathn, alpha, beta, delta)
            print ('K %s and Kc %s compeleted' %(k, kc))
'''


if __name__ == "__main__":
    
    #sns.set()
    alpha = 0.6
    beta = 0.1
    delta = 0.9
    k = 20
    kc = 10
    
    #path = 'C:/Project/EDU/files/2013/example/Topic/similarity/grid_mp_sep/a0.1b0.1d0.9-k17-c16d1/'
    #path = 'C:/Project/EDU/files/2013/example/Topic/similarity/grid_mp_con/a0.3b0.6d0.9-k19-c2d17-10/'
    #path = 'C:/Project/EDU/files/2013/example/Topic/similarity/grid_mp_sep/a0.1b0.9d0.9-k19-c13d6-10/'
    #path = 'C:/Project/EDU/files/2013/example/Topic/similarity/grid_mp_sep/'
    #path = 'C:/Project/EDU/OLI_175318/lg/mp_con/'
    
    
    
    typ = 'con'
    #typ = 'sep'
    
    #path = 'C:/Project/EDU/OLI_175318/hint/lg/grid_mp_con/a0.7b0.8d0.9-k30-c4d26/'
    #path = 'C:/Project/EDU/files/2013/example/Topic/similarity/grid_nmp_con/a0.2b0.9d0.9-k19-c13d6/'
    #path = 'C:/Project/EDU/files/2013/example/Topic/similarity/grid_nmp_con/a0.1b0.1d0.9-k20-c4d16/'
    #path = 'C:/Project/EDU/files/2013/example/Topic/similarity/grid_nmp_con/a0.6b0.1d0.9-k20-c12d8-1000/'
    path = 'C:/Project/EDU/files/2013/example/Topic/similarity/dual/'
    path = 'C:/Project/EDU/files/2013/example/Topic/similarity/baseline-error/'
    path = 'C:/Project/EDU/files/2013/example/Topic/similarity/baseline-mae/'
    #path = 'C:/Project/EDU/files/2013/example/Topic/similarity/dual/a0.1b0.4d0.9s0.3-k20-c10d10-2/'
    
    
    #grid_search(path, typ)
    
    #gd(typ,k,kc,path,alpha,beta,delta)
    #gd_eps(typ,k,kc,path,alpha,beta,delta)
    
    #iterating(path, typ, k, kc, alpha, beta, delta)
    
    #grid_search_constant_k(path, typ, k, kc)
    
    
    
    #path = 'C:/Project/EDU/files/2013/example/Topic/similarity/grid_mp_con/a0.1b0.1d0.9-k17-c5d12/'
    #path = 'C:/Project/EDU/files/2013/example/Topic/similarity/grid_mp_con/a0.3b0.1d0.9-k19-c6d-13/'
    #path = 'C:/Project/EDU/files/2013/example/Topic/similarity/grid_mp_con/a0.2b0.5d0.9-k18-c11d7/'
    #path = 'C:/Project/EDU/files/2013/example/Topic/similarity/k20-c12-f-CD/'
    path = 'C:/Project/EDU/files/2013/example/Topic/similarity/k20-c12-mae/'
    path = 'C:/Project/EDU/files/2013/example/Topic/similarity/No-structure/'
    path = 'C:/Project/EDU/files/2013/example/Topic/similarity/simple/'
    
    alpha = 0.6
    beta = 0.1
    delta = 0.9
    sigma = 0.3
    k = 20
    kc = 12
    '''
    alpha = 0.5
    beta = 0.5
    delta = 0.5
    k = 20
    kc = 12
    '''
    gd_simple(path, alpha, beta)
    
    #gd_eps(typ, k, kc, path, alpha, beta, delta)
    #gd_eps_no_structure(k, kc, path, alpha, beta)
    
    
    #gd_eps_dual(typ, k, kc, path, alpha, beta, delta, sigma)
    
    
    #grid_search(path, typ)
        
    
    
    '''
    k = 19
    kc = 2
    for i in range(1,11):
        pathn = path + str(i) + '/'
        gd_eps(typ,k,kc,pathn,alpha,beta,delta)
    '''
    
    '''
    for itr in range (1,11):
        #path = 'C:/Project/EDU/files/2013/example/Topic/60/LG/6040i10/'
        pathn = path + str(itr) + '/'
        for k in range(1, 31):
            for kc in range (1, k):
                gd(typ,k,kc,pathn,alpha,beta,delta)
    '''
    '''

    for i in range(1,2):
        path = 'C:/Project/EDU/files/2013/example/Topic/60/LG/6040_199/'
        path = 'C:/Project/EDU/files/2013/example/Topic/60/LG/6040i10-2/'
        path = 'C:/Project/EDU/files/2013/example/Topic/similarity/'
        path = path + str(i) + '/'
        gd(15,10,path)
        
    '''
    
    
    '''
    for itr in range (1,11):
        path = 'C:/Project/EDU/files/2013/example/Topic/similarity/it_mod/'
        path = path + str(itr) + '/'
        for k in range(5, 21):
            for kc in range (1, k):
                gd(k, kc, path)
            print 'K %s and Kc %s compeleted' %(k, kc)
    '''
    
    
