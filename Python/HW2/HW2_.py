#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from math import log

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import random
from pprint import pprint


# ## load test and train data

# In[198]:


df_test = pd.read_csv("C:\\Users\\n2ofu\\OneDrive\\Desktop\\test.txt", header = None)
df_test


# In[199]:


df_train = pd.read_csv("C:\\Users\\n2ofu\\OneDrive\\Desktop\\train.txt", header = None)
df_train


# In[144]:


test_df = pd.read_csv("C:\\Users\\n2ofu\\OneDrive\\Desktop\\test.txt", header = None)
test_df = np.array(test_df)
test_df


# ## Potential Splits

# In[139]:


def get_potential_splits(data):
    potential_splits = {}
    _n,n_columns = data.shape
    for column_index in range(n_columns - 1):
        potential_splits[column_index] = []
        values = data[:, column_index]
        unique_values = np.unique(values)
    
        for index in range(len(unique_values)):
            if index != 0:
                current_value = unique_values[index]
                previous_value = unique_values[index - 1]
                potential_split = (current_value + previous_value)/2

                potential_splits[column_index].append(potential_split)

    return potential_splits


# In[194]:


potential_splits_test= get_potential_splits(test_df)
print(potential_splits_test)


# In[204]:


sns.lmplot(x ="4",y= "1",hue = "10",data=df_test.rename(columns=lambda x: str(x)),
          fit_reg = False,size = 6, aspect = 1.5)
plt.vlines(x = potential_splits[4], ymin = 0, ymax = 40)
#plt.hlines(y = potential_splits[1], xmin = 0, xmax = 7)


# ## Split data

# In[295]:


def split_data(data, split_column, split_value):
    
    split_column_values = data[:, split_column]

    data_below = data[split_column_values <= split_value]
    data_above = data[split_column_values > split_value]
    
    return data_below, data_above


# In[305]:


split_column = 4 #[y for y in potential_splits.values()]
split_value = 2.5 #[x for x in potential_splits]


# In[306]:


data_below,data_above = split_data(test_df,split_column, split_value)
data_below


# In[307]:


data_above


# In[308]:


plotting_df = pd.DataFrame(test_df, columns =df.columns)
sns.lmplot(x = "4", y = "1",hue = "10", data= plotting_df.rename(columns=lambda x: str(x)),
           fit_reg = False, size = 6, aspect = 1.5)
plt.vlines(x = split_value, ymin = 20, ymax = 40)


# In[343]:


def entropy(data):
    target_column = data[:,-1] ## last column
    _, counts = np.unique(target_column, return_counts = True)
    
    probs = counts / counts.sum()
    entropy = sum(probs * - np.log2(probs))
    
    return entropy


# In[346]:


entropy(data_below)


# In[ ]:


def overall_entropy(data_below, data_above):
    n_data_points = len(data_below) + len(data_above)

    p_data_below = len(data_below)/n_data_points
    p_data_above = len(data_above)/n_data_points

    overall_entropy = (p_data_below * entropy(data_below)
                      + p_data_above* entropy(data_above))  
    return overall_entropy


# In[347]:


overall_entropy(data_below, data_above)


# In[348]:


def best_split(data, potential_splits):
    
    all_entropy = 999
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            data_below,data_above = split_data(data,split_column = column_index, split_value =value)
            current_overall_entropy = overall_entropy(data_below, data_above)

            if current_overall_entropy <= all_entropy:
                all_entropy = current_overall_entropy
                best_split_column = column_index 
                best_split_value = value
                
    return best_split_column, best_split_value


# In[350]:


best_split(test_df, potential_splits)


# In[356]:


def data(D,index, value):
    
    counts = np.unique(target_column, return_counts = True)
    probs = counts / counts.sum()
    classes = [x for x in potential_splits]
    
    return counts, probs, classes


# In[357]:


def IG(D, index, value):
    
    counts, probs, classes = data(D,index,value)
    
    HD = overall_entropy(data)
    HDy = np.sum([p*np.log2(p) for p in probs[0]])
    HDn = np.sum([p*np.log2(p) for p in probs[1]])

    IG = HD - counts[0]/np.sum(counts)*HDy - counts[1]/np.sum(counts)*HDn

    return IG


# In[359]:


def G(D, index, value):
    
    counts, probs, classes = data(D,index,value)
    
    GDy = 1-np.sum([p**2 for p in probs[0]])
    GDn = 1-np.sum([p**2 for p in probs[1]])

    G = counts[0]/np.sum(counts)*GDy + counts[1]/np.sum(counts)*GDn

    return G


# In[360]:


def CART(D, index, value):
    
    counts, probs, classes = data(D,index,value)
    
    CART = 2*(counts[0]/np.sum(counts)) * (counts[1]/np.sum(counts)) * np.sum([abs((probs[0][i] - probs[1][i]) for i in range(len(probs[0])))])
    
    return CART


# In[309]:


def best_split(D, criterion):
    
    if criterion == "IG":
        print "IG"
    if criterion == "G":
        print "G"
    if criterion == "CART":
        print "CART"
    
    for column_index in D:
        for value in potential_splits[column_index]:
            data_below,data_above = split_data(data,split_column = column_index, split_value =value)
            current_overall_entropy = overall_entropy(data_below, data_above)

            if current_overall_entropy <= all_entropy:
                all_entropy = current_overall_entropy
                best_split_column = column_index 
                best_split_value = value
                
    return best_split_column, best_split_value


# In[361]:


def load(filename):
    data = pd.read_csv(filename, header = None)
    
    attri = data[:,:-1]
    classes = data[:,-1]
    
    return (attribute,classes)


# In[ ]:


    

