# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 16:58:11 2019

@author: mirza
"""

import os
import matplotlib.pyplot as plt

def find_best_param(path):
    
    x1_error = []
    x2_error = []
    x12_error = []
    c_error = []
    d_error = []
    s_error = []
    sum_all_error = []
    xxs_error = []
    index = []
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
            print d
            dird = os.listdir(pathd)
            for k in dird:
                if os.path.isdir(pathd + k):
                    pathk = pathd + k + '/'
                    dirsk = os.listdir(pathk)
                    for c in dirsk:
                        pathc = pathk + c + '/'
                        if os.path.isdir(pathc):
                            pathf = pathc + 'err.txt'
                            
                            f = open(pathf, "r")
                            lines = f.readlines()
                                    
                            e1 = float(lines[0].split()[2])
                            e2 = float(lines[1].split()[2])
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
                                
                            #print '-----'*5
                                    
            
    print x1_min, x1_param
    print x2_min, x2_param
    print x12_min, x12_param
    print c_min, c_param
    print d_min, d_param
    print s_min, s_param
    print sum_all_min, sum_all_param
    print xxs_min, xxs_param
    
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
    
def find_best_param_sep(path):
    
    x1_error = []
    x2_error = []
    x12_error = []
    c_error = []
    d_error = []
    s1_error = []
    s2_error = []
    sum_all_error = []
    xxs_error = []
    index = []
    ix = 0
    
    x1_dict = {}
    x2_dict = {}
    x12_dict = {}
    c_dict = {}
    d_dict = {}
    s1_dict = {}
    s2_dict = {}
    sum_all_dict = {}
    xxs_dict = {}
    
    x1_param = ''
    x2_param = ''
    x12_param = ''
    c_param = ''
    d_param = ''
    s1_param = ''
    s2_param = ''
    sum_all_param = ''
    xxs_param = ''
    
    x1_min = 1.0
    x2_min = 1.0
    x12_min = 1.0
    c_min = 1.0
    d_min = 1.0
    s1_min = 1.0
    s2_min = 1.0
    sum_all_min = 1.0
    xxs_min = 1.0
    
    dirs = os.listdir(path)
    
    for d in dirs:
        if os.path.isdir(path + d) and len(d) == 12:
            pathd = path + d + '/'
            print 'd', d
            dird = os.listdir(pathd)
            for k in dird:
                if os.path.isdir(pathd + k):
                    pathk = pathd + k + '/'
                    dirsk = os.listdir(pathk)
                    for c in dirsk:
                        pathc = pathk + c + '/'
                        if os.path.isdir(pathc):
                            pathf = pathc + 'err.txt'
                            
                            f = open(pathf, "r")
                            lines = f.readlines()
                                    
                            e1 = float(lines[0].split()[2])
                            e2 = float(lines[1].split()[2])
                            e12 = float(lines[2].split()[2])
                            ec = float(lines[3].split()[2])
                            ed = float(lines[4].split()[2])
                            es1 = float(lines[5].split()[2])
                            es2 = float(lines[6].split()[2])
                            sum_all = e1 + e2 + e12 + ec + ed + es1 + es2
                            exxs = e1 + e2 + es1 + es2
                            
                            x1_error.append(e1)
                            x2_error.append(e2)
                            x12_error.append(e12)
                            c_error.append(ec)
                            d_error.append(ed)
                            s1_error.append(es1)
                            s2_error.append(es2)
                            sum_all_error.append(sum_all)
                            xxs_error.append(exxs)
                            
                            
                            comb = d + ' ' + k + ' ' + c
                            
                            x1_dict[e1] = comb
                            x2_dict[e2] = comb
                            x12_dict[e12] = comb
                            c_dict[ec] = comb
                            d_dict[ed] = comb
                            s1_dict[es1] = comb
                            s2_dict[es2] = comb
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
                            
                            if (es1 < s1_min):
                                s1_min = es1
                                s1_param = comb
                                
                            if (es2 < s2_min):
                                s2_min = es2
                                s2_param = comb
                                
                            if (sum_all < sum_all_min):
                                sum_all_min = sum_all
                                sum_all_param = comb
                            
                            if (exxs < xxs_min):
                                xxs_min = exxs
                                xxs_param = comb
                                
                            ix = ix + 1
                            index.append(ix)
                                
            
    print 'x1', x1_min, x1_param
    print 'x2',x2_min, x2_param
    print '12',x12_min, x12_param
    print 'c',c_min, c_param
    print 'd',d_min, d_param
    print 's1',s1_min, s1_param
    print 's2',s2_min, s2_param
    print 'sum',sum_all_min, sum_all_param
    print 'xxs',xxs_min, xxs_param
    
    fw = open(path + '/min_error.txt', "w")
    fw.write(x1_param + ': ' + str(x1_min) + '\n')
    fw.write(x2_param + ': ' + str(x2_min) + '\n')
    fw.write(x12_param + ': ' + str(x12_min) + '\n')
    fw.write(c_param + ': ' + str(c_min) + '\n')
    fw.write(d_param + ': ' + str(d_min) + '\n')
    fw.write(s1_param + ': ' + str(s1_min) + '\n')
    fw.write(s2_param + ': ' + str(s2_min) + '\n')
    fw.write(sum_all_param + ': ' + str(sum_all_min))
    fw.write(xxs_param + ': ' + str(xxs_min))
    
    fw.close()
    
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
    plt.plot(index,s1_error)
    plt.title('Error S')
    plt.xlabel('Iteration')
    plt.legend()
    plt.savefig(path + "/ErrorS1.png")
    
    plt.figure()
    plt.plot(index,s2_error)
    plt.title('Error S')
    plt.xlabel('Iteration')
    plt.legend()
    plt.savefig(path + "/ErrorS2.png")
    
    plt.figure()
    plt.plot(index,sum_all_error)
    plt.title('Error Sum All')
    plt.xlabel('Iteration')
    plt.legend()
    plt.savefig(path + "/ErrorAll.png")
    
    plt.figure()
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
    s1_error.sort()
    s2_error.sort()
    sum_all_error.sort()
    xxs_error.sort()
    
    fwt = open (path + '/top10.txt', 'w')
    
    out1 = ''
    out2 = ''
    out12 = ''
    outc = ''
    outd = ''
    outs1 = ''
    outs2 = ''
    outa = ''
    outx = ''
    
    for i in range(10):
        out1 = out1 + '\t' + x1_dict.get(x1_error[i]) + ': ' + str(x1_error[i])
        out2 = out2 + '\t' + x2_dict.get(x2_error[i]) + ': ' + str(x2_error[i])
        out12 = out12 + '\t' + x12_dict.get(x12_error[i]) + ': ' + str(x12_error[i])
        outc = outc + '\t' + c_dict.get(c_error[i]) + ': ' + str(c_error[i])
        outd = outd + '\t' + d_dict.get(d_error[i]) + ': ' + str(d_error[i])
        outs1 = outs1 + '\t' + s1_dict.get(s1_error[i]) + ': ' + str(s1_error[i])
        outs2 = outs2 + '\t' + s2_dict.get(s2_error[i]) + ': ' + str(s2_error[i])
        outa = outa + '\t' + sum_all_dict.get(sum_all_error[i]) + ': ' + str(sum_all_error[i])
        outx = outx + '\t' + xxs_dict.get(xxs_error[i]) + ': ' + str(xxs_error[i])
        
    fwt.write('X1: ' + out1[1:] + '\nX2: ' + out2[1:] + '\nX12: ' + out12[1:] + '\nC: ' + outc[1:] + '\nD: ' + outd[1:] + '\nS1: ' + outs1[1:] + '\nS2: ' + outs2[1:] + '\nSum: ' + outa[1:] + '\nX1X2S: ' + outx[1:])
    
    fwt.close()

if __name__ == "__main__":
    
    #path = 'C:/Project/EDU/files/2013/example/Topic/similarity/grid_mp_sep/'
    #path = 'C:/Project/EDU/files/2013/example/Topic/similarity/grid_mp_con/'
    path = 'C:/Project/EDU/files/2013/example/Topic/similarity/grid_nmp_con/'
    
    #path = 'C:/Project/EDU/OLI_175318/hint/lg/grid_mp_con/'
    
    find_best_param(path)
    #find_best_param_sep(path)
    