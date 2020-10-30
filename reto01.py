# Importar librerias
import numpy as np
import pandas as pd
from sklearn.manifold import Isomap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import random
import math

# Declaracion Funciones

# 6. Programacion de KNN

def distance (x1,y1,x2,y2):
    d = ((x2-x1)**2 + (y2-y1)**2)
    return math.sqrt(d)

def neighbors(punto,X,k):
    ds = list()
    num_fila = X.shape[0]
    for i in range (num_fila):
        d = distance(X[i,0],X[i,1],punto[0],punto[1])
        ds.append((i,d))
    ds.sort(key=lambda tup: tup[1]) 
    nb = list()
    for i in range (k):
        nb.append(ds[i][0])
    return nb

def pc (nb,k,X):
    t0 = 0
    t1 = 0
    t2 = 0
    for m in range (k):
        a = nb[m]
        if(X[a,2] == 1):   #Evaluar para tipo 1
            t1 = t1+1
        elif(X[a,2] == 0): #Evaluar para tipo 0
            t0 = t0+1
        else:              #Evaluar para tipo 2
            t2 = t2+1
    if(t0 > t1 and t0 > t2):
        return 0
    elif(t1 > t0 and t1 > t2):
        return 1
    else:
        return 2

# 1. Dataset seleccionado: Iris

# 2. Lectura iris.csv
df = pd.read_csv("C:/Users/Ana/.spyder-py3/iris.csv")
df["Tipo_Flor"] = df["Tipo_Flor"].replace(["Iris-versicolor", "Iris-virginica", "Iris-setosa"], [0,1,2])
data = df.values

# 3. Division en matriz de caracteri­sticas (X) y vector de clases (y)
X = data[:,0:-1]
y = data[:,-1]

# 4.A Prueba de distintos metodos
emb = LinearDiscriminantAnalysis(n_components = 2)
X1t = emb.fit_transform(X,y)

emb = MDS(n_components = 2)
X2t = emb.fit_transform(X,y)

emb = Isomap(n_components = 2)
X3t = emb.fit_transform(X,y)

# Grafica datos (sin metodo)
plt.scatter(X[:,0],X[:,1],c=y)
plt.title('Iris dataset')
plt.grid()
plt.show()

# 4.B Graficacion distintos metodos 
plt.scatter(X1t[:,0],X1t[:,1],c=y)
plt.title('Iris dataset, LDA')
plt.grid()
plt.show()

plt.scatter(X2t[:,0],X2t[:,1],c=y)
plt.title('Iris dataset, MDS')
plt.grid()
plt.show()

plt.scatter(X3t[:,0],X3t[:,1],c=y)
plt.title('Iris dataset, Isomap')
plt.grid()
plt.show()

# Llamar a las funciones (prediccion para r0)
l = len(X)
r0 = np.array([random.randint(-3,3),random.randint(-3,3)])
k = 5

# 5. Seleccionar metodo: MDS
plt.scatter(X2t[:,0],X2t[:,1],c=y)
plt.title('Iris dataset, MDS')
plt.scatter(r0[0],r0[1],c="red")  # Graficacion de r0 en rojo
plt.grid()
plt.show()

X2t = np.insert(X2t, 2, y[:], axis=1)
    
nb = neighbors(r0,X2t,k)
print("\nLos vecinos son: ")
for m in range (k):
    a = nb[m]
    print(X2t[a])

# 7. Graficacion de puntos con el evaluado clasificado
p = pc(nb,k,X2t)
print("\nLa clasificación del punto es: ",p)

if(p == 0):
    plt.scatter(X2t[:,0],X2t[:,1],c=y)
    plt.title('Iris dataset, MDS')
    plt.scatter(r0[0],r0[1],c="purple")
    plt.grid()
elif(p == 1):
    plt.scatter(X2t[:,0],X2t[:,1],c=y)
    plt.title('Iris dataset, MDS')
    plt.scatter(r0[0],r0[1],c="teal")
    plt.grid()
else:
    plt.scatter(X2t[:,0],X2t[:,1],c=y)
    plt.title('Iris dataset, MDS')
    plt.scatter(r0[0],r0[1],c="yellow")
    plt.grid()

plt.show()