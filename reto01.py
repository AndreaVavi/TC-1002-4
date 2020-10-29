#Importar librerías
import numpy as np
import pandas as pd
from sklearn.manifold import Isomap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import math

#Declaración Funcionaes


# 6. Programación de KNN

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
    return nb

def pc (nb,k):
    t0=0
    t1=0
    t2=0
    for m in range (k):
        a=nb[m]
        if(X[a,2]==1):   #Evaluar para tipo 1
            t1=t1+1
        elif(X[a,2]==0): #Evaluar para tipo 0
            t0=t0+1
        else:            #Evaluar para tipo 2
            t2=t2+1
    if(t0>t1 and t0>t2):
        return 0
    elif(t1>t0 and t1>t2):
        return 1
    else:
        return 2

# 1. Data Set seleccionado: Iris

# 2. Lectura iris.csv 
df = pd.read_csv("//Users/AndreaVavi/Downloads/iris.csv")
df["Tipo_Flor"] = df["Tipo_Flor"].replace(["Iris-versicolor", "Iris-virginica", "Iris-setosa"], [0,1,2])
data = df.values

# 3. División en matriz de características (X) y vector de clases (y)
X = data[:,0:-1]
y = data[:,-1]


# 4. A Prueba de distintos métodos
emb = LinearDiscriminantAnalysis(n_components = 2)
X1t = emb.fit_transform(X,y)

emb = MDS(n_components = 2)
X2t = emb.fit_transform(X,y)

emb = Isomap(n_components = 2)
X3t = emb.fit_transform(X,y)

# 4.B Graficación distintos métodos 
plt.scatter(X1t[:,0], X1t[:,1], c=y)
plt.title('Iris dataset, LDA')
plt.show()

plt.scatter(X2t[:,0], X2t[:,1], c=y)
plt.title('Iris dataset, MDS')
plt.show()

plt.scatter(X3t[:,0], X3t[:,1], c=y)
plt.title('Iris dataset, Isomap')
plt.show()

#Grafica datos (sin método)
plt.scatter(X[:,0], X[:,1], c=y)
plt.title('Iris dataset')

# 5. Seleccionar métodos de reducción *

#allData = X.copy()
#for i in range(150):
#    np.insert(allData, 4, y[i], axis=1)

#Llamar a las funciones

l=len(X)
r0=[1,1]
k=4


for i in range(l):
    X[i,2] = y[i]
    

nb= neighbors(r0,X,k)
print ("\nLos valores vecinos son: ")
for m in range (k):
    a=nb[m]
    print(X[a])

# 7. Graficación de puntos con el evaluado resaltado (en rojo) y mostrar clasificación
p=pc(nb,k)
print("\nLa clasificación del punto es: ",p)

plt.scatter(r0[0], r0[1], c="red")
plt.show()