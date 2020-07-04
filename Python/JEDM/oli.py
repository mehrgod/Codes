# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 18:05:07 2020

@author: mirza
"""

import matplotlib
import seaborn as sns
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
    
def histogram_clean(path):
    #path = "C:/Project/EDU/OLI_175318/update/"
    
    #f = open(path + "student-perf-sequence.txt")
    #f = open(path + "perf.txt")
    f = open(path + "grades.txt")
    
    lines = f.readlines()[1:]
    
    #pre = []
    post = []
    #score = []
    
    for l in lines:
        #ix, pr, po, sc, sq = l.split('\t')
        #ix, pr, po, sc = l.split('\t')
        ix, po = l.split('\t')
        #pre.append(int(pr))
        post.append(int(po))
        #score.append(int(sc))
    
    '''
    plt.figure(1)
    #OLI-psy
    #plt.hist(pre, bins = 20)
    #MG
    plt.hist(pre, bins = 14)
    plt.savefig(path + "pre.png")
    plt.savefig(path + "pre.pdf")
    '''
    
    plt.figure(2)
    #OLI-psy
    #plt.hist(post, bins = 36)
    #MG
    #plt.hist(post, bins = 24)
    
    plt.hist(post, bins = 'auto')
    plt.savefig(path + "post.png")
    plt.savefig(path + "post.pdf")
    
    '''
    plt.figure(3)
    #OLI-psy
    #plt.hist(score, bins = 53)
    #MG
    plt.hist(score, bins = 23)
    plt.savefig(path + "score.png")
    plt.savefig(path + "score.pdf")
    '''
    
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
    #plt.savefig(path + "lg2.png")
    
    plt.show()

def stat_attempts_histogram(path):
    #path = "C:/Project/EDU/OLI_175318/update/step/sep/"
    
    attempt_length = []
    
    with open(path + 'Sequence.txt') as lines:
        for line in lines:
            
            id, vec = line.split('\t')
            l = str(vec).replace("_", "").strip()
            attempt_length.append(len(l))
            print len(l)
            
    #plt.hist(attempt_length, bins = 'auto')
    plt.hist(attempt_length, bins = 100)
    #plt.title('Frequency of attempt numbers')
    plt.xlabel('Number of Attempts')
    plt.ylabel('Frequency')
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    #matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')
    #plt.rcParams.update({'font.size': 24})
    plt.savefig(path + 'histogram-attempts.pdf')
    plt.show()
            
def stat_hint_histogram(path):
    #path = "C:/Project/EDU/OLI_175318/update/step/sep/"
    
    hints = []
    
    with open(path + 'Sequence.txt') as lines:
        for line in lines:
            id, vec = line.split('\t')
            counter1 = str(vec).count('h')
            counter2 = str(vec).count('H')
            counter = counter1 + counter2
            hints.append(counter)
            
    plt.hist(hints, bins = 'auto')
    #plt.hist(hints, bins = 40)
    #plt.xlabel('Number of Examples')
    plt.xlabel('Number of Hints')
    plt.ylabel('Frequency')
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    plt.savefig(path + 'histogram-hints.pdf')
    plt.show()
    
def stat_s_histogram(path):
    #path = "C:/Project/EDU/OLI_175318/update/step/sep/"
    
    hints = []
    
    with open(path + 'Sequence.txt') as lines:
        for line in lines:
            id, vec = line.split('\t')
            counter1 = str(vec).count('s')
            counter2 = str(vec).count('S')
            counter = counter1 + counter2
            hints.append(counter)
            
    #plt.hist(hints, bins = 'auto')
    plt.hist(hints, bins = 50)
    plt.xlabel('Number of Successes')
    plt.ylabel('Frequency')
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    plt.savefig(path + 'histogram-success.pdf')
    plt.show()
    
def stat_f_histogram(path):
    #path = "C:/Project/EDU/OLI_175318/update/step/sep/"
    
    hints = []
    
    with open(path + 'Sequence.txt') as lines:
        for line in lines:
            id, vec = line.split('\t')
            counter1 = str(vec).count('f')
            counter2 = str(vec).count('F')
            counter = counter1 + counter2
            hints.append(counter)
            
    #plt.hist(hints, bins = 'auto')
    plt.hist(hints, bins = 50)
    plt.xlabel('Number of Failures')
    plt.ylabel('Frequency')
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    plt.savefig(path + 'histogram-failure.pdf')
    plt.show()


if __name__ == "__main__":
    
    #histogram_clean()
    #histogram_simple()
    
    sns.set()
    #matplotlib.rc_file_defaults()
    
    path = "C:/Project/EDU/OLI_175318/update/step/sep/"
    path = "C:/Project/EDU/files/2013/example/Topic/60/"
    path = "C:/Project/EDU/Statistics-ds1139/"
    
    #histogram_clean(path)
    
    #stat_attempts_histogram(path)
    #stat_hint_histogram(path)
    stat_f_histogram(path)
    stat_s_histogram(path)