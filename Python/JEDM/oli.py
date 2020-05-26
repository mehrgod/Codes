# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 18:05:07 2020

@author: mirza
"""

import matplotlib.pyplot as plt

def histogram():
    path = "C:/Project/EDU/OLI_175318/update/"
    
    grades = []
    
    #f = open(path + "post.txt")
    #f = open(path + "pre.txt")
    f = open(path + "average-problem-score-100.txt")
    
    lines = f.readlines()
    
    for l in lines:
        g = int(l.split('.')[0])
        grades.append(g)
    
    
    print max(grades)
    print min(grades)
    
    #plt.hist(grades, bins = 36)
    plt.hist(grades, bins = 50)
    #plt.savefig(path + "post.png")
    plt.savefig(path + "average-problem-score-100.png")
    plt.show()
    
def histogram_clean():
    path = "C:/Project/EDU/OLI_175318/update/"
    
    f = open(path + "student-perf-sequence.txt")
    
    lines = f.readlines()
    
    pre = []
    post = []
    score = []
    
    for l in lines:
        ix, pr, po, sc, sq = l.split('\t')
        pre.append(int(pr))
        post.append(int(po))
        score.append(int(sc))
    
    plt.figure(1)
    plt.hist(pre, bins = 20)
    plt.savefig(path + "pre.png")
    
    plt.figure(2)
    plt.hist(post, bins = 36)
    plt.savefig(path + "post.png")
    
    plt.figure(3)
    plt.hist(score, bins = 53)
    plt.savefig(path + "score.png")
    
    
    
    #plt.savefig(path + "post.png")
    #plt.savefig(path + "average-problem-score-100.png")
    plt.show()
    
def histogram_simple():
    path = "C:/Project/EDU/OLI_175318/update/"
    
    f = open(path + "lg2.txt")
    
    lines = f.readlines()
    
    lg = []
    
    for l in lines:
        lg.append(float(l))
    plt.figure(1)
    plt.hist(lg, bins = 35)
    plt.savefig(path + "lg2.png")
    
    
    
    
    plt.show()

if __name__ == "__main__":
    
    #histogram_clean()
    histogram_simple()