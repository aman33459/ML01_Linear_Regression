import pandas as pd
import numpy as np
import matplotlib.pyplot as mpl
from collections import defaultdict

data = pd.read_csv("train.csv")
data1 = pd.read_csv('test.csv')
y = data['Severity']
del data['Severity']
del data['Accident_ID']
for column in data.columns:
    fmean=np.mean(data[column])
    frange = np.amax(data[column]) - np.amin(data[column])
    data[column]-=fmean
    data[column] /= frange
arr = np.random.uniform(0.1,1,(1,data.shape[1]+1))
dic= dict()
prop = open('values.properties' , 'r')
yexp = []
for i in range(0,len(y)):
    yexp.append(dic[y[i]])
for i in prop:
    lis = i.split(':')
    dic[lis[0]] = float(lis[1])
ones = np.ones(data.shape[0]);
datan = data.assign(value = ones)
arrt = np.transpose(arr)
yout=np.dot(datan,arrt)
diff =  yexp - yout.T
diff = diff.T
gradient = np.dot(-datan.T , diff)
gradient = gradient/ len(data)
lr = 0.5
gradient*=lr
gradient = gradient.T
arr = arr-gradient
