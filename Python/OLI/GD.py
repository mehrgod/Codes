# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 18:39:30 2020

@author: mirza
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from Distance import concateWcWd

def gd_eps(k, kc, pathin, pathout, alpha, beta, delta, grid, epoc):
    
    typ = 'con'
    
    #with open('C:/Project/EDU/OLI_175318/update/step/sep/tfidf/avg/l.txt') as file:
    with open(pathin + 'lg/l.txt') as file:
        array2dX1 = [[float(digit) for digit in line.split(',')] for line in file]
    
    X1 = np.array(array2dX1)
    
    #with open('C:/Project/EDU/OLI_175318/update/step/sep/tfidf/avg/h.txt') as file:
    with open(pathin + 'lg/h.txt') as file:
        array2dX2 = [[float(digit) for digit in line.split(',')] for line in file]
    
    X2 = np.array(array2dX2)
    
    #with open('C:/Project/EDU/OLI_175318/update/step/sep/tfidf/nmp.csv') as file:
    with open(pathin + 'nmp.csv') as file:
        arrayS = [[float(digit) for digit in line.split(',')] for line in file]
        
    S = np.array(arrayS)
    
    #epoc = 1000
    
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
    
    eps_list = []
    
    #X1X2
    gama = 3.0 - (alpha + beta)
    #gama = 0.8
    eps = 1
    #S
    #delta = 0.5
    
    reg = 1

    for e in range(epoc):
        learning_rate = 0.01/np.sqrt(e+1)
        #learning_rate_c = 0.003/np.sqrt(e+1)
        learning_rate_c = 0.01/np.sqrt(e+1)
    
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

        W1cn = W1c - learning_rate_c * grad_w1c
        
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
            
        W2cn = W2c - learning_rate_c * grad_w2c
        
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
        
        if (typ == 'con'):
            grad_eps = delta * np.sum(np.multiply( (-2 * np.dot(W, W.T)) , (S - eps * np.dot(W, W.T)) ))
        else:
            if (typ == 'sep'):
                grad_eps = delta * (np.sum(np.multiply( (-2 * np.dot(W1, W1.T)) , (S - eps * np.dot(W1, W1.T)) ))
                + np.sum(np.multiply( (-2 * np.dot(W2, W2.T)) , (S - eps * np.dot(W2, W2.T)) ))
                )
        
        eps = eps - learning_rate * grad_eps
        
        eps_list.append(eps)
        
        
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
            
        
    mode = 'write'
    #mode = 'test'
    
    if (mode == 'write'):        
        if (os.path.isdir(pathout) == False):
            os.mkdir(pathout)
        pathk = pathout + "k" + str(k)
        if (os.path.isdir(pathk) == False):
            os.mkdir(pathk)
        
        pathkc = pathk + "/c" + str(kc) + "d" + str(k-kc)    
        if (os.path.isdir(pathkc) == False):
            os.mkdir(pathkc)
        
        if (grid == 'test'):
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

            
        fw.close()
        fwx1.close()
        fwx2.close()
        fwe.close()
        
        pathk = pathout + str(k)
        
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
        

    pathk = pathout + "k" + str(k)
    pathkc = pathk + "/c" + str(kc) + "d" + str(k-kc)
    print (pathkc)
    
    
    #mode = 'test'
    mode = 'write'
    
    #sns.set()
    
    plt.figure()
    plt.plot(index,eps_list)
    plt.title('Error eps')
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    if mode == 'write':
        plt.savefig(pathkc + "/Eps.png")
        plt.savefig(pathkc + "/Eps.pdf")
        if (grid == 'grid'):
            plt.clf()
            plt.close()
    
    plt.figure()
    plt.plot(index,errX1)
    plt.title('Error X1')
    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction Error')
    if mode == 'write':
        plt.savefig(pathkc + "/ErrorX1.png")
        plt.savefig(pathkc + "/ErrorX1.pdf")
        if (grid == 'grid'):
            plt.clf()
            plt.close()
    
    plt.figure()
    plt.plot(index,errX2)
    plt.title('Error X2')
    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction Error')
    if mode == 'write':
        plt.savefig(pathkc + "/ErrorX2.png")
        plt.savefig(pathkc + "/ErrorX2.pdf")
        if (grid == 'grid'):
            plt.clf()
            plt.close()
    
    plt.figure()
    plt.plot(index,errSqrC)
    plt.title('Error C')
    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction Error')
    if mode == 'write':
        plt.savefig(pathkc + "/ErrorC.png")
        plt.savefig(pathkc + "/ErrorC.pdf")
        if (grid == 'grid'):
            plt.clf()
            plt.close()
    
    plt.figure()
    plt.plot(index,errD)
    plt.title('Error D')
    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction Error')
    if mode == 'write':
        plt.savefig(pathkc + "/ErrorD.png")
        plt.savefig(pathkc + "/ErrorD.pdf")
        if (grid == 'grid'):
            plt.clf()
            plt.close()
    
    
    plt.figure()
    plt.plot(index,errS)
    plt.title('Error S')
    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction Error')
    if mode == 'write':
        plt.savefig(pathkc + "/ErrorS.png")
        plt.savefig(pathkc + "/ErrorS.pdf")
        if (grid == 'grid'):
            plt.clf()
            plt.close()
            
    print ('plot')
    #print eps
    plt.show()
    #plt.rcParams.update({'font.size': 10})
    
def gd_mu(k, kc, pathin, pathout, alpha, beta, delta, lamda, ep, grid, epoc):
    
    #typ = 'con'
    
    with open(pathin + 'lg/l.txt') as file:
        array2dX1 = [[float(digit) for digit in line.split(',')] for line in file]
    
    X1 = np.array(array2dX1)
    
    with open(pathin + 'lg/h.txt') as file:
        array2dX2 = [[float(digit) for digit in line.split(',')] for line in file]
    
    X2 = np.array(array2dX2)
    
    #with open(pathin + 'nmp.csv') as file:
    #    arrayS = [[float(digit) for digit in line.split(',')] for line in file]
        
    #S = np.array(arrayS)
    
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
    errmu1 = []
    errmu2 = []
    errmu12 = []
        
    #errS1 = []
    #errS2 = []
    #errS = []
    
    #eps_list = []
    
    #X1X2
    gama = 3.0 - (alpha + beta)
    #gama = 0.8
    #S
    #delta = 0.5
    
    reg = 1

    for e in range(epoc):
        learning_rate = 0.05/np.sqrt(e+1)
        #learning_rate_c = 0.003/np.sqrt(e+1)
        learning_rate_c = 0.01/np.sqrt(e+1)
        learning_rate_h = 0.01/np.sqrt(e+1)
    
        W1c = W1[:,:kc]
        W1d = W1[:,kc:]
        
        H1c = H1[:,:kc]
        H1d = H1[:,kc:]
        
        W2c = W2[:,:kc]
        W2d = W2[:,kc:]
        
        H2c = H2[:,:kc]
        H2d = H2[:,kc:]
        
        #W = np.concatenate((W1, W2), axis = 1)
        
        grad_w1c = (-2 * gama * np.dot((np.dot(W1, H1.T) - X1), H1c)
        + 2 * alpha * (W1c - W2c)
        + 2 * reg * W1c
        #- 4 * delta * np.dot((S - eps * np.dot(W, W.T)), W1c)
        ) 

        W1cn = W1c - learning_rate_c * grad_w1c
        
        grad_w2c = (-2 * gama * np.dot((np.dot(W2, H2.T) - X2), H2c)
        - 2 * alpha * (W1c - W2c)
        #+ 2 * reg * W2c
        #- 4 * delta * np.dot((S - eps * np.dot(W, W.T)), W2c)
        )
            
        W2cn = W2c - learning_rate_c * grad_w2c
        
        grad_w1d = (-2 * gama * np.dot((np.dot(W1, H1.T) - X1), H1d) 
        + 2 * beta * np.dot(W2d, np.dot(W1d.T,W2d))
        #+ 2 * reg * W1d
        #- 4 * delta * np.dot((S - eps * np.dot(W, W.T)), W1d)
        )
        
        W1dn = W1d - learning_rate * grad_w1d
        
        grad_w2d = (-2 * gama * np.dot((np.dot(W2, H2.T) - X2), H2d) 
        + 2 * beta * np.dot(W1d, np.dot(W1d.T,W2d))
        #+ 2 * reg * W2d
        #- 4 * delta * np.dot((S - eps * np.dot(W, W.T)), W2d)
        )
        
        W2dn = W2d - learning_rate * grad_w2d
        
        vector_e1 = np.ones(n1)
        matrix_e1 = np.dot(vector_e1, vector_e1.T)
        mu1 = np.dot(H1.T, matrix_e1)/float(n1)
        
        vector_e2 = np.ones(n2)
        matrix_e2 = np.dot(vector_e2, vector_e2.T)
        mu2 = np.dot(H2.T, matrix_e2)/float(n2)
        
        #print 'H1.t', H1.T.shape
        #print 'mu1', mu1.shape
        #print 'mu1', mu1
        #print 'mu2', mu2.shape
        #print 'mu2', mu2
        #print 'mu1.mu2', np.dot(mu2,np.dot(mu1.T,mu2).T).shape
        #print H1.trace()
        
        #print np.dot(mu2.T, mu1).shape
        #print np.dot(np.transpose(X1 - np.dot(W1, H1.T)), W1).shape
        #print (H1.T - mu1).T.shape
        #print np.dot(mu2, np.dot(mu2.T, mu1)).T.shape
        
        
        grad_h1 = (-2 * gama * np.dot(np.transpose(X1 - np.dot(W1, H1.T)), W1) 
        + 2 * lamda * (H1.T - mu1).T
        + 2 * ep * np.dot(mu2, np.dot(mu1.T,mu2).T).T
        #- 2 * np.dot(mu2, np.dot(mu2.T, mu1)).T
        #+ 2 * reg * H1.T
        )
        
        H1n = H1 - learning_rate_h * grad_h1
        
        #print np.dot(np.transpose(X2 - np.dot(W2, H2.T)), W2).shape
        #print (H2.T - mu2).T.shape
        #print np.dot(mu1, np.dot(mu2.T, mu1).T).shape
        
        grad_h2 = (-2 * gama * np.dot(np.transpose(X2 - np.dot(W2, H2.T)), W2) 
        + 2 * lamda * (H2.T - mu2).T
        + 2 * ep * np.dot(mu1, np.dot(mu1.T, mu2)).T
        #- 2 * np.dot(mu1, np.dot(mu2.T, mu1).T).T
        #+ 2 * reg * H2
        )
        
        H2n = H2 - learning_rate_h * grad_h2
        
        #grad_eps = delta * np.sum(np.multiply( (-2 * np.dot(W, W.T)) , (S - eps * np.dot(W, W.T)) ))
        
        #eps = eps - learning_rate * grad_eps
        
        #eps_list.append(eps)
        
        
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
        
        #errormu1 = error(H1.T, np.dot(H1.T, matrix_e1)/float(n1))
        errormu1 = error(H1.T, mu1)
        
        #errormu2 = error(H2.T, np.dot(H2.T, matrix_e2)/float(n2))
        errormu2 = error(H2.T, mu2)
        
        errormu12 = lossfuncD(mu1.T, mu2)

        #errorS = error(S, eps * np.dot(W, W.T))
        
        
        index.append(e)
        
        errX1.append(errorX1)

        errX2.append(errorX2)
        
        errX1X2.append((errorX1 + errorX2)/2)
        
        errSqrC.append(errorSqrC)
        
        errD.append(errorD)
        
        errmu1.append(errormu1)
        
        errmu2.append(errormu2)
        
        errmu12.append(errormu12)
        
        #errS.append(errorS)
        
        W1 = W1n
        H1 = H1n
        
        W2 = W2n
        H2 = H2n
                
        if (e % 10 == 0):
            print (e)
            
        
    mode = 'write'
    #mode = 'test'
    
    if (mode == 'write'):        
        if (os.path.isdir(pathout) == False):
            os.mkdir(pathout)
        pathk = pathout + "k" + str(k)
        if (os.path.isdir(pathk) == False):
            os.mkdir(pathk)
        
        pathkc = pathk + "/c" + str(kc) + "d" + str(k-kc)    
        if (os.path.isdir(pathkc) == False):
            os.mkdir(pathkc)
        
        if (grid == 'test'):
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
            
        #for i in eps_list:
        #    fwe.write(str(i) + '\n')
            
        errX1mae = errorMAE(X1, np.dot(W1, H1.T))
        errX2mae = errorMAE(X2, np.dot(W2, H2.T))
  
        
        fw.write("Error X1 RMSE: " + str(errX1[-1]) + " MAE: " + str(errX1mae) + "\n")
        fw.write("Error X2 RMSE: " + str(errX2[-1]) + " MAE: " + str(errX2mae) + "\n")
        fw.write("Error X1+X2: " + str(errX1X2[-1]) + "\n")
        fw.write("Error C: " + str(errSqrC[-1]) + "\n")
        fw.write("Error D: " + str(errD[-1]) + "\n")
        fw.write("Error mu1: " + str(errmu1[-1]) + "\n")
        fw.write("Error mu2: " + str(errmu2[-1]))
        #fw.write("Error S: " + str(errS[-1]) + "\n")
        #fw.write("Eps: " + str(eps))
        
        #errorX1 = np.abs(X1 - np.dot(W1,np.transpose(H1)))
        #errorX2 = np.abs(X2 - np.dot(W2,np.transpose(H2)))

            
        fw.close()
        fwx1.close()
        fwx2.close()
        fwe.close()
        
        pathk = pathout + str(k)
        
        err1 = error(X1, np.dot(W1, H1.T))
        fw1 = open(pathk + 'err1.txt', "w")
        fw1.write(str(err1))
        fw1.close()
        
        err2 = error(X2, np.dot(W2, H2.T))
        fw2 = open(pathk + 'err2.txt', "w")
        fw2.write(str(err2))
        fw2.close()
        
        #errs = error(S, np.dot(W,W.T))
        #fw3 = open(pathk + 'errs.txt', "w")
        #fw3.write(str(errs))
        #fw3.close()
        

    pathk = pathout + "k" + str(k)
    pathkc = pathk + "/c" + str(kc) + "d" + str(k-kc)
    print (pathkc)
    
    
    #mode = 'test'
    mode = 'write'
    
    #sns.set()
    '''
    plt.figure()
    plt.plot(index,eps_list)
    plt.title('Error eps')
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    if mode == 'write':
        plt.savefig(pathkc + "/Eps.png")
        plt.savefig(pathkc + "/Eps.pdf")
        if (grid == 'grid'):
            plt.clf()
            plt.close()
    '''
    
    plt.figure()
    plt.plot(index,errX1)
    plt.title('Error X1')
    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction Error')
    if mode == 'write':
        plt.savefig(pathkc + "/ErrorX1.png")
        plt.savefig(pathkc + "/ErrorX1.pdf")
        if (grid == 'grid'):
            plt.clf()
            plt.close()
    
    plt.figure()
    plt.plot(index,errX2)
    plt.title('Error X2')
    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction Error')
    if mode == 'write':
        plt.savefig(pathkc + "/ErrorX2.png")
        plt.savefig(pathkc + "/ErrorX2.pdf")
        if (grid == 'grid'):
            plt.clf()
            plt.close()
    
    plt.figure()
    plt.plot(index,errSqrC)
    plt.title('Error C')
    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction Error')
    if mode == 'write':
        plt.savefig(pathkc + "/ErrorC.png")
        plt.savefig(pathkc + "/ErrorC.pdf")
        if (grid == 'grid'):
            plt.clf()
            plt.close()
    
    plt.figure()
    plt.plot(index,errD)
    plt.title('Error D')
    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction Error')
    if mode == 'write':
        plt.savefig(pathkc + "/ErrorD.png")
        plt.savefig(pathkc + "/ErrorD.pdf")
        if (grid == 'grid'):
            plt.clf()
            plt.close()
    
    plt.figure()
    plt.plot(index,errmu1)
    plt.title('Error mu1')
    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction Error')
    if mode == 'write':
        plt.savefig(pathkc + "/Errormu1.png")
        plt.savefig(pathkc + "/Errormu1.pdf")
        if (grid == 'grid'):
            plt.clf()
            plt.close()
    
    plt.figure()
    plt.plot(index,errmu2)
    plt.title('Error mu2')
    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction Error')
    if mode == 'write':
        plt.savefig(pathkc + "/Errormu2.png")
        plt.savefig(pathkc + "/Errormu2.pdf")
        if (grid == 'grid'):
            plt.clf()
            plt.close()
    
    plt.figure()
    plt.plot(index,errmu12)
    plt.title('Error mu12')
    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction Error')
    if mode == 'write':
        plt.savefig(pathkc + "/Errormu12.png")
        plt.savefig(pathkc + "/Errormu12.pdf")
        if (grid == 'grid'):
            plt.clf()
            plt.close()
    
    
    '''
    plt.figure()
    plt.plot(index,errS)
    plt.title('Error S')
    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction Error')
    if mode == 'write':
        plt.savefig(pathkc + "/ErrorS.png")
        plt.savefig(pathkc + "/ErrorS.pdf")
        if (grid == 'grid'):
            plt.clf()
            plt.close()
    '''        
    print ('plot')
    #print eps
    plt.show()
    #plt.rcParams.update({'font.size': 10})


def gd_mu_structure(k, kc, pathin, pathout, alpha, beta, delta, lamda, ep, grid, epoc):
    
    #typ = 'con'
    
    with open(pathin + 'lg/l.txt') as file:
        array2dX1 = [[float(digit) for digit in line.split(',')] for line in file]
    
    X1 = np.array(array2dX1)
    
    with open(pathin + 'lg/h.txt') as file:
        array2dX2 = [[float(digit) for digit in line.split(',')] for line in file]
    
    X2 = np.array(array2dX2)
    
    with open(pathin + 'nmp.csv') as file:
        arrayS = [[float(digit) for digit in line.split(',')] for line in file]
        
    S = np.array(arrayS)
    
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
    errmu1 = []
    errmu2 = []
    errmu12 = []
        
    #errS1 = []
    #errS2 = []
    errS = []
    
    eps_list = []
    
    #X1X2
    gama = 3.0 - (alpha + beta)
    #gama = 0.8
    #S
    #delta = 0.5
    eps = 1
    
    reg = 0.1

    for e in range(epoc):
        learning_rate = 0.001/np.sqrt(e+1)
        #learning_rate_c = 0.003/np.sqrt(e+1)
        learning_rate_c = 0.01/np.sqrt(e+1)
        learning_rate_d = 0.01/np.sqrt(e+1)
        learning_rate_h = 0.01/np.sqrt(e+1)
    
        W1c = W1[:,:kc]
        W1d = W1[:,kc:]
        
        H1c = H1[:,:kc]
        H1d = H1[:,kc:]
        
        W2c = W2[:,:kc]
        W2d = W2[:,kc:]
        
        H2c = H2[:,:kc]
        H2d = H2[:,kc:]
        
        W = np.concatenate((W1, W2), axis = 1)
        
        grad_w1c = (-2 * gama * np.dot((np.dot(W1, H1.T) - X1), H1c)
        + 2 * alpha * (W1c - W2c)
        + 2 * reg * W1c
        - 4 * delta * np.dot((S - eps * np.dot(W, W.T)), W1c)
        ) 

        W1cn = W1c - learning_rate_c * grad_w1c
        
        grad_w2c = (-2 * gama * np.dot((np.dot(W2, H2.T) - X2), H2c)
        - 2 * alpha * (W1c - W2c)
        + 2 * reg * W2c
        - 4 * delta * np.dot((S - eps * np.dot(W, W.T)), W2c)
        )
            
        W2cn = W2c - learning_rate_c * grad_w2c
        
        grad_w1d = (-2 * gama * np.dot((np.dot(W1, H1.T) - X1), H1d) 
        + 2 * beta * np.dot(W2d, np.dot(W1d.T,W2d))
        + 2 * reg * W1d
        - 4 * delta * np.dot((S - eps * np.dot(W, W.T)), W1d)
        )
        
        W1dn = W1d - learning_rate_d * grad_w1d
        
        grad_w2d = (-2 * gama * np.dot((np.dot(W2, H2.T) - X2), H2d) 
        + 2 * beta * np.dot(W1d, np.dot(W1d.T,W2d))
        + 2 * reg * W2d
        - 4 * delta * np.dot((S - eps * np.dot(W, W.T)), W2d)
        )
        
        W2dn = W2d - learning_rate_d * grad_w2d
        
        vector_e1 = np.ones(n1)
        matrix_e1 = np.dot(vector_e1, vector_e1.T)
        mu1 = np.dot(H1.T, matrix_e1)/float(n1)
        
        vector_e2 = np.ones(n2)
        matrix_e2 = np.dot(vector_e2, vector_e2.T)
        mu2 = np.dot(H2.T, matrix_e2)/float(n2)
        
        #print 'H1.t', H1.T.shape
        #print 'mu1', mu1.shape
        #print 'mu1', mu1
        #print 'mu2', mu2.shape
        #print 'mu2', mu2
        #print 'mu1.mu2', np.dot(mu2,np.dot(mu1.T,mu2).T).shape
        #print H1.trace()
        
        #print np.dot(mu2.T, mu1).shape
        #print np.dot(np.transpose(X1 - np.dot(W1, H1.T)), W1).shape
        #print (H1.T - mu1).T.shape
        #print np.dot(mu2, np.dot(mu2.T, mu1)).T.shape
        
        
        grad_h1 = (-2 * gama * np.dot(np.transpose(X1 - np.dot(W1, H1.T)), W1) 
        + 2 * lamda * (H1.T - mu1).T
        + 2 * ep * np.dot(mu2, np.dot(mu1.T,mu2).T).T
        + 2 * reg * H1
        )
        
        H1n = H1 - learning_rate_h * grad_h1
        
        #print np.dot(np.transpose(X2 - np.dot(W2, H2.T)), W2).shape
        #print (H2.T - mu2).T.shape
        #print np.dot(mu1, np.dot(mu2.T, mu1).T).shape
        
        grad_h2 = (-2 * gama * np.dot(np.transpose(X2 - np.dot(W2, H2.T)), W2) 
        + 2 * lamda * (H2.T - mu2).T
        + 2 * ep * np.dot(mu1, np.dot(mu1.T, mu2)).T
        + 2 * reg * H2
        )
        
        H2n = H2 - learning_rate_h * grad_h2
        
        grad_eps = delta * np.sum(np.multiply( (-2 * np.dot(W, W.T)) , (S - eps * np.dot(W, W.T)) ))
        
        eps = eps - learning_rate * grad_eps
        
        eps_list.append(eps)
        
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
        
        #errormu1 = error(H1.T, np.dot(H1.T, matrix_e1)/float(n1))
        errormu1 = error(H1.T, mu1)
        
        #errormu2 = error(H2.T, np.dot(H2.T, matrix_e2)/float(n2))
        errormu2 = error(H2.T, mu2)
        
        errormu12 = lossfuncD(mu1.T, mu2)

        errorS = error(S, eps * np.dot(W, W.T))
        
        
        index.append(e)
        
        errX1.append(errorX1)

        errX2.append(errorX2)
        
        errX1X2.append((errorX1 + errorX2)/2)
        
        errSqrC.append(errorSqrC)
        
        errD.append(errorD)
        
        errmu1.append(errormu1)
        
        errmu2.append(errormu2)
        
        errmu12.append(errormu12)
        
        errS.append(errorS)
        
        W1 = W1n
        H1 = H1n
        
        W2 = W2n
        H2 = H2n
                
        if (e % 10 == 0):
            print (e)
            
        
    mode = 'write'
    #mode = 'test'
    
    if (mode == 'write'):        
        if (os.path.isdir(pathout) == False):
            os.mkdir(pathout)
        pathk = pathout + "k" + str(k)
        if (os.path.isdir(pathk) == False):
            os.mkdir(pathk)
        
        pathkc = pathk + "/c" + str(kc) + "d" + str(k-kc)    
        if (os.path.isdir(pathkc) == False):
            os.mkdir(pathkc)
        
        if (grid == 'test'):
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
  
        
        fw.write("Error X1 RMSE: " + str(errX1[-1]) + " MAE: " + str(errX1mae) + "\n")
        fw.write("Error X2 RMSE: " + str(errX2[-1]) + " MAE: " + str(errX2mae) + "\n")
        fw.write("Error X1+X2: " + str(errX1X2[-1]) + "\n")
        fw.write("Error C: " + str(errSqrC[-1]) + "\n")
        fw.write("Error D: " + str(errD[-1]) + "\n")
        fw.write("Error mu1: " + str(errmu1[-1]) + "\n")
        fw.write("Error mu2: " + str(errmu2[-1]))
        fw.write("Error S: " + str(errS[-1]) + "\n")
        fw.write("Eps: " + str(eps))
        
        #errorX1 = np.abs(X1 - np.dot(W1,np.transpose(H1)))
        #errorX2 = np.abs(X2 - np.dot(W2,np.transpose(H2)))

            
        fw.close()
        fwx1.close()
        fwx2.close()
        fwe.close()
        
        pathk = pathout + str(k)
        
        err1 = error(X1, np.dot(W1, H1.T))
        fw1 = open(pathk + 'err1.txt', "w")
        fw1.write(str(err1))
        fw1.close()
        
        err2 = error(X2, np.dot(W2, H2.T))
        fw2 = open(pathk + 'err2.txt', "w")
        fw2.write(str(err2))
        fw2.close()
        
        errs = error(S, np.dot(W,W.T))
        fw3 = open(pathk + 'errs.txt', "w")
        fw3.write(str(errs))
        fw3.close()
        

    pathk = pathout + "k" + str(k)
    pathkc = pathk + "/c" + str(kc) + "d" + str(k-kc)
    print (pathkc)
    
    
    #mode = 'test'
    mode = 'write'
    
    #sns.set()
    
    plt.figure()
    plt.plot(index,eps_list)
    plt.title('Error eps')
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    if mode == 'write':
        plt.savefig(pathkc + "/Eps.png")
        plt.savefig(pathkc + "/Eps.pdf")
        if (grid == 'grid'):
            plt.clf()
            plt.close()
    
    
    plt.figure()
    plt.plot(index,errX1)
    plt.title('Error X1')
    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction Error')
    if mode == 'write':
        plt.savefig(pathkc + "/ErrorX1.png")
        plt.savefig(pathkc + "/ErrorX1.pdf")
        if (grid == 'grid'):
            plt.clf()
            plt.close()
    
    plt.figure()
    plt.plot(index,errX2)
    plt.title('Error X2')
    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction Error')
    if mode == 'write':
        plt.savefig(pathkc + "/ErrorX2.png")
        plt.savefig(pathkc + "/ErrorX2.pdf")
        if (grid == 'grid'):
            plt.clf()
            plt.close()
    
    plt.figure()
    plt.plot(index,errSqrC)
    plt.title('Error C')
    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction Error')
    if mode == 'write':
        plt.savefig(pathkc + "/ErrorC.png")
        plt.savefig(pathkc + "/ErrorC.pdf")
        if (grid == 'grid'):
            plt.clf()
            plt.close()
    
    plt.figure()
    plt.plot(index,errD)
    plt.title('Error D')
    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction Error')
    if mode == 'write':
        plt.savefig(pathkc + "/ErrorD.png")
        plt.savefig(pathkc + "/ErrorD.pdf")
        if (grid == 'grid'):
            plt.clf()
            plt.close()
    
    plt.figure()
    plt.plot(index,errmu1)
    plt.title('Error mu1')
    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction Error')
    if mode == 'write':
        plt.savefig(pathkc + "/Errormu1.png")
        plt.savefig(pathkc + "/Errormu1.pdf")
        if (grid == 'grid'):
            plt.clf()
            plt.close()
    
    plt.figure()
    plt.plot(index,errmu2)
    plt.title('Error mu2')
    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction Error')
    if mode == 'write':
        plt.savefig(pathkc + "/Errormu2.png")
        plt.savefig(pathkc + "/Errormu2.pdf")
        if (grid == 'grid'):
            plt.clf()
            plt.close()
    
    plt.figure()
    plt.plot(index,errmu12)
    plt.title('Error mu12')
    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction Error')
    if mode == 'write':
        plt.savefig(pathkc + "/Errormu12.png")
        plt.savefig(pathkc + "/Errormu12.pdf")
        if (grid == 'grid'):
            plt.clf()
            plt.close()
    
    
    
    plt.figure()
    plt.plot(index,errS)
    plt.title('Error S')
    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction Error')
    if mode == 'write':
        plt.savefig(pathkc + "/ErrorS.png")
        plt.savefig(pathkc + "/ErrorS.pdf")
        if (grid == 'grid'):
            plt.clf()
            plt.close()
            
    print ('plot')
    #print eps
    plt.show()
    #plt.rcParams.update({'font.size': 10})

def gd_mu_parameter(k, kc, pathin, pathout, alpha, beta, delta, lamda, ep, grid, epoc):
    
    #typ = 'con'
    
    with open(pathin + 'lg/l.txt') as file:
        array2dX1 = [[float(digit) for digit in line.split(',')] for line in file]
    
    X1 = np.array(array2dX1)
    
    with open(pathin + 'lg/h.txt') as file:
        array2dX2 = [[float(digit) for digit in line.split(',')] for line in file]
    
    X2 = np.array(array2dX2)
    
    #with open(pathin + 'nmp.csv') as file:
    #    arrayS = [[float(digit) for digit in line.split(',')] for line in file]
        
    #S = np.array(arrayS)
    
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
    errmu1 = []
    errmu2 = []
    errmu12 = []
        
    #errS1 = []
    #errS2 = []
    #errS = []
    
    #eps_list = []
    
    #X1X2
    gama = 3.0 - (alpha + beta)
    #gama = 0.8
    #S
    #delta = 0.5
    #eps = 1
    
    reg = 0.01

    for e in range(epoc):
        #learning_rate = 0.001/np.sqrt(e+1)
        learning_rate_c = 0.01/np.sqrt(e+1)
        learning_rate_d = 0.01/np.sqrt(e+1)
        learning_rate_h = 0.01/np.sqrt(e+1)
    
        W1c = W1[:,:kc]
        W1d = W1[:,kc:]
        
        H1c = H1[:,:kc]
        H1d = H1[:,kc:]
        
        W2c = W2[:,:kc]
        W2d = W2[:,kc:]
        
        H2c = H2[:,:kc]
        H2d = H2[:,kc:]
        
        #W = np.concatenate((W1, W2), axis = 1)
        
        grad_w1c = (-2 * gama * np.dot((np.dot(W1, H1.T) - X1), H1c)
        + 2 * alpha * (W1c - W2c)
        + 2 * reg * W1c
        #- 4 * delta * np.dot((S - eps * np.dot(W, W.T)), W1c)
        ) 

        W1cn = W1c - learning_rate_c * grad_w1c
        
        grad_w2c = (-2 * gama * np.dot((np.dot(W2, H2.T) - X2), H2c)
        - 2 * alpha * (W1c - W2c)
        + 2 * reg * W2c
        #- 4 * delta * np.dot((S - eps * np.dot(W, W.T)), W2c)
        )
            
        W2cn = W2c - learning_rate_c * grad_w2c
        
        grad_w1d = (-2 * gama * np.dot((np.dot(W1, H1.T) - X1), H1d) 
        + 2 * beta * np.dot(W2d, np.dot(W1d.T,W2d))
        + 2 * reg * W1d
        #- 4 * delta * np.dot((S - eps * np.dot(W, W.T)), W1d)
        )
        
        W1dn = W1d - learning_rate_d * grad_w1d
        
        grad_w2d = (-2 * gama * np.dot((np.dot(W2, H2.T) - X2), H2d) 
        + 2 * beta * np.dot(W1d, np.dot(W1d.T,W2d))
        + 2 * reg * W2d
        #- 4 * delta * np.dot((S - eps * np.dot(W, W.T)), W2d)
        )
        
        W2dn = W2d - learning_rate_d * grad_w2d
        
        vector_e1 = np.ones(n1)
        matrix_e1 = np.dot(vector_e1, vector_e1.T)
        mu1 = np.dot(H1.T, matrix_e1)/float(n1)
        
        vector_e2 = np.ones(n2)
        matrix_e2 = np.dot(vector_e2, vector_e2.T)
        mu2 = np.dot(H2.T, matrix_e2)/float(n2)
        
        #print 'H1.t', H1.T.shape
        #print 'mu1', mu1.shape
        #print 'mu1', mu1
        #print 'mu2', mu2.shape
        #print 'mu2', mu2
        #print 'mu1.mu2', np.dot(mu2,np.dot(mu1.T,mu2).T).shape
        #print H1.trace()
        
        #print np.dot(mu2.T, mu1).shape
        #print np.dot(np.transpose(X1 - np.dot(W1, H1.T)), W1).shape
        #print (H1.T - mu1).T.shape
        #print np.dot(mu2, np.dot(mu2.T, mu1)).T.shape
        
        
        grad_h1 = (-2 * gama * np.dot(np.transpose(X1 - np.dot(W1, H1.T)), W1) 
        + 2 * lamda * (H1.T - mu1).T
        - 2 * ep * np.dot(mu2, np.dot(mu1.T,mu2).T).T
        + 2 * reg * H1
        )
        
        H1n = H1 - learning_rate_h * grad_h1
        
        #print np.dot(np.transpose(X2 - np.dot(W2, H2.T)), W2).shape
        #print (H2.T - mu2).T.shape
        #print np.dot(mu1, np.dot(mu2.T, mu1).T).shape
        
        grad_h2 = (-2 * gama * np.dot(np.transpose(X2 - np.dot(W2, H2.T)), W2) 
        + 2 * lamda * (H2.T - mu2).T
        - 2 * ep * np.dot(mu1, np.dot(mu1.T, mu2)).T
        + 2 * reg * H2
        )
        
        H2n = H2 - learning_rate_h * grad_h2
        
        #grad_eps = delta * np.sum(np.multiply( (-2 * np.dot(W, W.T)) , (S - eps * np.dot(W, W.T)) ))
        
        #eps = eps - learning_rate * grad_eps
        
        #eps_list.append(eps)
        
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
        
        #errormu1 = error(H1.T, np.dot(H1.T, matrix_e1)/float(n1))
        errormu1 = error(H1.T, mu1)
        
        #errormu2 = error(H2.T, np.dot(H2.T, matrix_e2)/float(n2))
        errormu2 = error(H2.T, mu2)
        
        errormu12 = lossfuncD(mu1.T, mu2)

        #errorS = error(S, eps * np.dot(W, W.T))
        
        
        index.append(e)
        
        errX1.append(errorX1)

        errX2.append(errorX2)
        
        errX1X2.append((errorX1 + errorX2)/2)
        
        errSqrC.append(errorSqrC)
        
        errD.append(errorD)
        
        errmu1.append(errormu1)
        
        errmu2.append(errormu2)
        
        errmu12.append(errormu12)
        
        #errS.append(errorS)
        
        W1 = W1n
        H1 = H1n
        
        W2 = W2n
        H2 = H2n
                
        if (e % 10 == 0):
            print (e)
            
        
    mode = 'write'
    #mode = 'test'
    
    if (mode == 'write'):        
        if (os.path.isdir(pathout) == False):
            os.mkdir(pathout)
        pathk = pathout + "k" + str(k)
        if (os.path.isdir(pathk) == False):
            os.mkdir(pathk)
        
        pathkc = pathk + "/c" + str(kc) + "d" + str(k-kc)    
        if (os.path.isdir(pathkc) == False):
            os.mkdir(pathkc)
        
        if (grid == 'test'):
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
            
        #for i in eps_list:
        #    fwe.write(str(i) + '\n')
            
        errX1mae = errorMAE(X1, np.dot(W1, H1.T))
        errX2mae = errorMAE(X2, np.dot(W2, H2.T))
  
        
        fw.write("Error X1 RMSE: " + str(errX1[-1]) + " MAE: " + str(errX1mae) + "\n")
        fw.write("Error X2 RMSE: " + str(errX2[-1]) + " MAE: " + str(errX2mae) + "\n")
        fw.write("Error X1+X2: " + str(errX1X2[-1]) + "\n")
        fw.write("Error C: " + str(errSqrC[-1]) + "\n")
        fw.write("Error D: " + str(errD[-1]) + "\n")
        fw.write("Error mu1: " + str(errmu1[-1]) + "\n")
        fw.write("Error mu2: " + str(errmu2[-1]))
        #fw.write("Error S: " + str(errS[-1]) + "\n")
        #fw.write("Eps: " + str(eps))
        
        #errorX1 = np.abs(X1 - np.dot(W1,np.transpose(H1)))
        #errorX2 = np.abs(X2 - np.dot(W2,np.transpose(H2)))

            
        fw.close()
        fwx1.close()
        fwx2.close()
        fwe.close()
        
        pathk = pathout + str(k)
        
        err1 = error(X1, np.dot(W1, H1.T))
        fw1 = open(pathk + 'err1.txt', "w")
        fw1.write(str(err1))
        fw1.close()
        
        err2 = error(X2, np.dot(W2, H2.T))
        fw2 = open(pathk + 'err2.txt', "w")
        fw2.write(str(err2))
        fw2.close()
        
        #errs = error(S, np.dot(W,W.T))
        #fw3 = open(pathk + 'errs.txt', "w")
        #fw3.write(str(errs))
        #fw3.close()
        

    pathk = pathout + "k" + str(k)
    pathkc = pathk + "/c" + str(kc) + "d" + str(k-kc)
    print (pathkc)
    
    
    #mode = 'test'
    mode = 'write'
    
    #sns.set()
    
    #plt.figure()
    #plt.plot(index,eps_list)
    #plt.title('Error eps')
    #plt.xlabel('Iteration')
    #plt.ylabel('Value')
    #if mode == 'write':
    #    plt.savefig(pathkc + "/Eps.png")
    #    plt.savefig(pathkc + "/Eps.pdf")
    #    if (grid == 'grid'):
    #        plt.clf()
    #        plt.close()
    
    
    plt.figure()
    plt.plot(index,errX1)
    plt.title('Error X1')
    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction Error')
    if mode == 'write':
        plt.savefig(pathkc + "/ErrorX1.png")
        plt.savefig(pathkc + "/ErrorX1.pdf")
        if (grid == 'grid'):
            plt.clf()
            plt.close()
    
    plt.figure()
    plt.plot(index,errX2)
    plt.title('Error X2')
    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction Error')
    if mode == 'write':
        plt.savefig(pathkc + "/ErrorX2.png")
        plt.savefig(pathkc + "/ErrorX2.pdf")
        if (grid == 'grid'):
            plt.clf()
            plt.close()
    
    plt.figure()
    plt.plot(index,errSqrC)
    plt.title('Error C')
    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction Error')
    if mode == 'write':
        plt.savefig(pathkc + "/ErrorC.png")
        plt.savefig(pathkc + "/ErrorC.pdf")
        if (grid == 'grid'):
            plt.clf()
            plt.close()
    
    plt.figure()
    plt.plot(index,errD)
    plt.title('Error D')
    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction Error')
    if mode == 'write':
        plt.savefig(pathkc + "/ErrorD.png")
        plt.savefig(pathkc + "/ErrorD.pdf")
        if (grid == 'grid'):
            plt.clf()
            plt.close()
    
    plt.figure()
    plt.plot(index,errmu1)
    plt.title('Error mu1')
    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction Error')
    if mode == 'write':
        plt.savefig(pathkc + "/Errormu1.png")
        plt.savefig(pathkc + "/Errormu1.pdf")
        if (grid == 'grid'):
            plt.clf()
            plt.close()
    
    plt.figure()
    plt.plot(index,errmu2)
    plt.title('Error mu2')
    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction Error')
    if mode == 'write':
        plt.savefig(pathkc + "/Errormu2.png")
        plt.savefig(pathkc + "/Errormu2.pdf")
        if (grid == 'grid'):
            plt.clf()
            plt.close()
    
    plt.figure()
    plt.plot(index,errmu12)
    plt.title('Error mu12')
    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction Error')
    if mode == 'write':
        plt.savefig(pathkc + "/Errormu12.png")
        plt.savefig(pathkc + "/Errormu12.pdf")
        if (grid == 'grid'):
            plt.clf()
            plt.close()
    
    
    
    #plt.figure()
    #plt.plot(index,errS)
    #plt.title('Error S')
    #plt.xlabel('Iteration')
    #plt.ylabel('Reconstruction Error')
    #if mode == 'write':
    #    plt.savefig(pathkc + "/ErrorS.png")
    #    plt.savefig(pathkc + "/ErrorS.pdf")
    #    if (grid == 'grid'):
    #        plt.clf()
    #        plt.close()
            
    print ('plot')
    #print eps
    plt.show()
    #plt.rcParams.update({'font.size': 10})


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
        
        #print 'Mode write'
        
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
    
    #sns.set()
    
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

def grid_search(pathin, pathout):
    
    #a = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    #b = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    #d = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    #a = [0.9]
    #b = [0.9]
    #d = [0.9]
    #r = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    #a = [0.1]
    #a = [0.5]
    #a = [1.0]
    a = [1.4]
    b = [0.1, 0.5, 1.0, 1.4]
    d = [0.1, 0.5, 1.0, 1.4]
    
    
    kc = 0
    for alpha in a:
        for beta in b:
            for delta in d:
                pathn = pathout + 'a' + str(alpha) + 'b' + str(beta) + 'd' + str(delta) + '/'
                
                if (os.path.isdir(pathout) == False):
                    os.mkdir(pathout)
                if (os.path.isdir(pathn) == False):
                    os.mkdir(pathn)                
                for k in range(4, 21, 2):
                    for kc in range (1, k):                        
                        try:
                            gd_eps(k, kc, pathin, pathn, alpha, beta, delta, grid = 'grid', epoc = 100)
                        except Exception as e:
                            fwerr = open(pathout + '/errors' + '_a' + str(alpha) + '_b' + str(beta) + '_d' + str(delta) +  '.txt', "w")
                            fwerr.write('alpha: ' + str(alpha) + '\tbeta: ' + str(beta) + '\tdelta: ' + str(delta)+ '\tk: ' + str(k) + '\tkc: ' + str(kc)  + '\n' + str(e))
                            fwerr.close()                            
                    print ('K %s and Kc %s compeleted' %(k, kc))
                    
                
                
def find_best_param_4(path):
    
    x1_error = []
    x2_error = []
    x12_error = []
    c_error = []
    d_error = []
    s_error = []
    sum_all_error = []
    xxs_error = []
    index = []
    real_index = []
    ix = 0
    
    x1_dict = {}
    x2_dict = {}
    x12_dict = {}
    c_dict = {}
    d_dict = {}
    s_dict = {}
    sum_all_dict = {}
    xxs_dict = {}
    
    x1_param = ''
    x2_param = ''
    x12_param = ''
    c_param = ''
    d_param = ''
    s_param = ''
    sum_all_param = ''
    xxs_param = ''
    
    x1_min = 1.0
    x2_min = 1.0
    x12_min = 1.0
    c_min = 1.0
    d_min = 1.0
    s_min = 1.0
    sum_all_min = 1.0
    xxs_min = 1.0
    
    dirs = os.listdir(path)
    
    for d in dirs:
        if os.path.isdir(path + d) and len(d) == 12:
            pathd = path + d + '/'
            print (d)
            dird = os.listdir(pathd)
            for k in dird:
                if os.path.isdir(pathd + k):
                    pathk = pathd + k + '/'
                    dirsk = os.listdir(pathk)
                    for c in dirsk:
                        pathc = pathk + c + '/'
                        if os.path.isdir(pathc):
                            pathf = pathc + 'err.txt'
                            print (pathf)
                            f = open(pathf, "r")
                            lines = f.readlines()
                                    
                            e1 = float(lines[0].split()[3])
                            e2 = float(lines[1].split()[3])
                            e12 = float(lines[2].split()[2])
                            ec = float(lines[3].split()[2])
                            ed = float(lines[4].split()[2])
                            es = float(lines[5].split()[2])
                            sum_all = e1 + e2 + e12 + ec + ed + es
                            exxs = e1 + e2 + es
                            
                            x1_error.append(e1)
                            x2_error.append(e2)
                            x12_error.append(e12)
                            c_error.append(ec)
                            d_error.append(ed)
                            s_error.append(es)
                            sum_all_error.append(sum_all)
                            xxs_error.append(exxs)
                            
                            
                            comb = d + ' ' + k + ' ' + c
                            
                            x1_dict[e1] = comb
                            x2_dict[e2] = comb
                            x12_dict[e12] = comb
                            c_dict[ec] = comb
                            d_dict[ed] = comb
                            s_dict[es] = comb
                            sum_all_dict[sum_all] = comb
                            xxs_dict[exxs] = comb
                            
                            
                            if e1 < x1_min:
                                x1_min = e1
                                x1_param = comb
                            
                            if (e2 < x2_min):
                                x2_min = e2
                                x2_param = comb
                            
                            if (e12 < x12_min):
                                x12_min = e12
                                x12_param = comb
                            
                            if (ec < c_min):
                                c_min = ec
                                c_param = comb
                            
                            if (ed < d_min):
                                d_min = ed
                                d_param = comb
                            
                            if (es < s_min):
                                s_min = es
                                s_param = comb
                                
                            
                            if (sum_all < sum_all_min):
                                sum_all_min = sum_all
                                sum_all_param = comb
                            
                            if (exxs < xxs_min):
                                xxs_min = exxs
                                xxs_param = comb
                                
                            
                            ix = ix + 1
                            index.append(ix)
                            real_index.append('K='+ k.replace('k','') + ', Kc=' + c.split('d')[0].replace('c','') + ', Kd=' + c.split('d')[1].replace('d',''))
                                
                            #print '-----'*5
                                    
            
    print (x1_min, x1_param)
    print (x2_min, x2_param)
    print (x12_min, x12_param)
    print (c_min, c_param)
    print (d_min, d_param)
    print (s_min, s_param)
    print (sum_all_min, sum_all_param)
    print (xxs_min, xxs_param)
    
    fw = open(path + '/min_error.txt', "w")
    fw.write(x1_param + ': ' + str(x1_min) + '\n')
    fw.write(x2_param + ': ' + str(x2_min) + '\n')
    fw.write(x12_param + ': ' + str(x12_min) + '\n')
    fw.write(c_param + ': ' + str(c_min) + '\n')
    fw.write(d_param + ': ' + str(d_min) + '\n')
    fw.write(s_param + ': ' + str(s_min) + '\n')
    fw.write(sum_all_param + ': ' + str(sum_all_min))
    fw.write(xxs_param + ': ' + str(xxs_min))
    
    fw.close()
    
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.rcParams.update({'font.size': 24})
    
    plt.figure()
    plt.plot(index,x1_error)
    plt.title('Error X1')
    plt.xlabel('Iteration')
    plt.legend()
    plt.savefig(path + "/ErrorX1.png")
    
    plt.figure()
    plt.plot(index,x2_error)
    plt.title('Error X2')
    plt.xlabel('Iteration')
    plt.legend()
    plt.savefig(path + "/ErrorX2.png")
    
    plt.figure()
    plt.plot(index,x12_error)
    plt.title('Error X12')
    plt.xlabel('Iteration')
    plt.legend()
    plt.savefig(path + "/ErrorX12.png")
    
    plt.figure()
    plt.plot(index,c_error)
    plt.title('Error C')
    plt.xlabel('Iteration')
    plt.legend()
    plt.savefig(path + "/ErrorC.png")
    
    plt.figure()
    plt.plot(index,d_error)
    plt.title('Error D')
    plt.xlabel('Iteration')
    plt.legend()
    plt.savefig(path + "/ErrorD.png")
    
    plt.figure()
    plt.plot(index,s_error)
    plt.title('Error S')
    plt.xlabel('Iteration')
    plt.legend()
    plt.savefig(path + "/ErrorS.png")
    
    plt.figure()
    plt.plot(index,sum_all_error)
    plt.title('Error Sum All')
    plt.xlabel('Iteration')
    plt.legend()
    plt.savefig(path + "/ErrorAll.png")
    
    plt.figure()
    plt.xticks(rotation = 90)
    plt.plot(index,xxs_error)
    plt.title('Error X1+X2+S')
    plt.xlabel('Iteration')
    plt.legend()
    plt.savefig(path + "ErrorXXS.png")
    
    plt.show()
    
    x1_error.sort()
    x2_error.sort()
    x12_error.sort()
    c_error.sort()
    d_error.sort()
    s_error.sort()
    sum_all_error.sort()
    xxs_error.sort()
    
    fwt = open (path + '/top10.txt', 'w')
    
    out1 = ''
    out2 = ''
    out12 = ''
    outc = ''
    outd = ''
    outs = ''
    outa = ''
    outx = ''
    
    for i in range(10):
        out1 = out1 + '\t' + x1_dict.get(x1_error[i]) + ': ' + str(x1_error[i])
        out2 = out2 + '\t' + x2_dict.get(x2_error[i]) + ': ' + str(x2_error[i])
        out12 = out12 + '\t' + x12_dict.get(x12_error[i]) + ': ' + str(x12_error[i])
        outc = outc + '\t' + c_dict.get(c_error[i]) + ': ' + str(c_error[i])
        outd = outd + '\t' + d_dict.get(d_error[i]) + ': ' + str(d_error[i])
        outs = outs + '\t' + s_dict.get(s_error[i]) + ': ' + str(s_error[i])
        outa = outa + '\t' + sum_all_dict.get(sum_all_error[i]) + ': ' + str(sum_all_error[i])
        outx = outx + '\t' + xxs_dict.get(xxs_error[i]) + ': ' + str(xxs_error[i])

    fwt.write('X1\t' + out1[1:] + '\nX2\t' + out2[1:] + '\nX12\t' + out12[1:] + '\nC\t' + outc[1:] + '\nD\t' + outd[1:] + '\nS\t' + outs[1:] + '\nSum\t' + outa[1:] + '\nX1X2S\t' + outx[1:])
    
    fwt.close()
                


if __name__ == "__main__":
    
    
    #path = "C:/Project/EDU/OLI_175318/update/step/sep/tfidf/lg/grid/a1.0b1.0d1.4-k10kc6-1000/"
    #path = "C:/Project/EDU/OLI_175318/update/step/sep/tfidf/post/grid/"
    #path = "C:/Project/EDU/OLI_175318/update/step/sep/tfidf/pre/grid/a1.4b0.5d1.4-k10kc6-1000/"
    #path = "C:/Project/EDU/OLI_175318/update/step/sep/tfidf/post/grid/a1.0b1.4d1.4-k10-c5d5-1000/"
    path = "C:/Project/EDU/OLI_175318/update/step/sep/tfidf/avg/grid/a1.0b1.0d1.4-k10-kc7kd3-1000/"
    path = "C:/Project/EDU/OLI_175318/update/step/sep/train-test/0.5/lg/grid-0.1-0.1-1.0/"
    path = "C:/Project/EDU/OLI_175318/update/step/sep/train-test/0.5/lg/grid/a0.1b0.1d1.0-k14-c3d11/"
    pathout = "C:/Project/EDU/OLI_175318/update/step/sep/train-test/0.4/lg/grid/"
    path = "C:/Project/EDU/OLI_175318/update/step/sep/train-test/method1/4/0.5/"
    
    path = "C:/Project/EDU/files/2013/example/Topic/60/prediction/4/0.5/"
    
    '''
    for OLI:
        alpha = 1.5
        beta = 0.5
    for MG:
        alpha = 1.0
        beta = 1.0
    '''
    
    
    alpha = 1.0
    beta = 1.0
    delta = 0.5
    lamda = 0.1
    ep = 0.5
    
    k = 20
    kc = 10
    
    #gd_eps(k, kc, path, path + "lg/grid/a0.1b1.4d1.4-k12-c3d9/", alpha, beta, delta, grid = 'test', epoc = 1000)
    
    '''
    index = "a" + str(alpha) + "b" + str(beta) + "d" + str(delta) + "-k" + str(k) + "-c" + str(kc) + "d" + str(k-kc)
    pathn = path + "lg/grid/" + index + "/"
    #pathn = path + "lg/" + index + "/"
    print pathn
    if (os.path.isdir(pathn) == False):
        os.mkdir(pathn)
    else:
        print "Directory Already Exist!"
        
    
    gd_eps(k, kc, path, pathn, alpha, beta, delta, grid = 'test', epoc = 1000)
    concateWcWd(pathn + "k" + str(k) + "/c" + str(kc) + "d" + str(k-kc) + "/")
    '''

    
    
    #grid_search(path, path + "lg/grid/")
    
    
    
    #find_best_param_4(path + "lg/grid/")
    '''
    path = 'C:/Project/EDU/OLI_175318/update/step/sep/train-test/method1/'
    path = "C:/Project/EDU/files/2013/example/Topic/60/predictionFilter2/"
    
    dir1 = ['1/','2/','3/','4/']
    dir2 = ['0.1/','0.2/','0.3/','0.4/','0.5/']
    for d1 in dir1:
        for d2 in dir2:
            pathn = path + d1 + d2
            print pathn
            grid_search(pathn, pathn + "lg/grid/")
            #find_best_param_4(pathn + "lg/grid/")
    '''
    path = 'C:/Project/EDU/OLI_175318/update/step/sep/train-test/method1/1/0.1/'
    #path = 'C:/Project/EDU/files/2013/example/Topic/60/predictionFilter2/1/0.1/'
    pathn = path + 'MuParamTest/'
    #gd_mu(k, kc, path, pathn, alpha, beta, delta, lamda, ep, grid = 'test', epoc = 1000)
    #gd_mu_structure(k, kc, path, pathn, alpha, beta, delta, lamda, ep, grid = 'test', epoc = 1000)
    gd_mu_parameter(k, kc, path, pathn, alpha, beta, delta, lamda, ep, grid = 'test', epoc = 1000)
  