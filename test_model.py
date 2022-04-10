# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 20:22:41 2022

@author: 81916
"""
import pickle
from train_model import Data 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
with open("network.model", "rb" ) as f:
    obj = pickle.load(f)


epochs = range(len(obj.train_loss_list))

plt.plot(epochs[10:],obj.train_loss_list[10:], 'b', label='training loss',linewidth = 1)
plt.plot(epochs[10:], obj.test_loss_list[10:], 'r', label='testing loss',linewidth = 1)
plt.title('Training and testing loss')
plt.legend(loc='upper right')
plt.figure()

plt.plot(epochs[10:],obj.test_accuracy_list[10:], 'b', label='testing accurancy')
plt.title('Testing accuracy')
plt.legend()
plt.show()
#%% visualiza parameters
with open("acc_list", "rb" ) as f:
    alist = pickle.load(f)
f.close()
data={}
stepsize_list = [1e-2, .5e-2, 1e-3]
X = [[],[],[]]
y_tick = []
for i,a in enumerate(alist):
    X[i // 9].append(a['acc'])
for a in alist[:9]:
    y_tick.append((a['hide'],a['reg']))
for i in range(3):
    data[stepsize_list[i]] = X[i]
pd_data=pd.DataFrame(data,index=y_tick,columns=stepsize_list)
    

heat = sns.heatmap(data = pd_data, square = False, annot= True,fmt=".4f",linewidths=.5,  cmap = "RdPu") 
heat.set_xlabel('Learning rate',fontsize=10)

heat.set_ylabel('(hidder layer,reg factor)',fontsize=10)
heat.tick_params(labelsize=8)

