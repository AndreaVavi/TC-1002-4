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