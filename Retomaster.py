# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
from sklearn.manifold import Isomap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import math


def graphPoints(X,y): #Grafica los puntos y los puntos proyectados
    for i in range(0,X.shape[0]):
        p = X[i,0:2]
        #p2 = Xp[i,0:2]            
        if(y[i]):
            plt.scatter(p[0],p[1],color="red", s=4)
        else:
            plt.scatter(p[0],p[1],color="black", s=4)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.show()
    
def distance (x1,y1,x2,y2):
    d = ((x2-x1)+(y2-y1))**2
    return math.sqrt(d)

def neighbors(punto,X,k):
    ds=list()
    num_fila=X.shape[0]
    for i in range (num_fila):
        d=distance(X[i,1],X[i,2],punto[0],punto[1])
        ds.append((i,d))

    ds.sort(key=lambda tup: tup[1]) 

    nb= list()
    for i in range (k):
        nb.append(ds[i][0])
    print(nb)
    return nb

def pc (nb,k):
    b=0
    c=0
    d=0
    for m in range (k):
        a=nb[m]
        if(X[a,2]==1):   #Evaluar para tipo 1
            b=b+1
        elif(X[a,2]==0): #Evaluar para tipo 0
            c=c+1
        else:            #Evaluar para tipo 2
            d=d+1
    if(c>b and c>d):
        return 0
    elif(b>c and b>d):
        return 1
    else:
        return 2

#Lectura iris1.csv
df = pd.read_csv("C:/Users/Ana/.spyder-py3/iris.csv")
df["Tipo_Flor"] = df["Tipo_Flor"].replace(["Iris-versicolor", "Iris-virginica", "Iris-setosa"], [0,1,2])
data = df.values
X = data[:,0:-1]
y = data[:,-1]

emb = LinearDiscriminantAnalysis(n_components = 2)
X1t = emb.fit_transform(X,y)

emb = MDS(n_components = 2)
X2t = emb.fit_transform(X,y)

emb = Isomap(n_components = 2)
X3t = emb.fit_transform(X,y)

plt.scatter(X1t[:,0], X1t[:,1], c=y)
plt.title('Iris dataset, LDA')
plt.show()
plt.scatter(X2t[:,0], X2t[:,1], c=y)
plt.title('Iris dataset, MDS')
plt.show()
plt.scatter(X3t[:,0], X3t[:,1], c=y)
plt.title('Iris dataset, Isomap')
plt.show()

plt.scatter(X[:,0], X[:,1], c=y)
plt.title('Iris dataset')
