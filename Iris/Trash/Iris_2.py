import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
np.set_printoptions(suppress=True)


#Loading data


Iris        = load_iris()
Iris_Data   = Iris.data
Iris_Target = Iris.target
Iris_Names  = Iris.target_names

print(Iris_Data.shape)


# Lenght for each of the flowers
training_length = 30
testing_length  = 20
total_length    = training_length + testing_length



#Splitting data into testing and training data
Iris_Training_Data      = []
Iris_Testing_Data       = []
Iris_Training_Target    = []
Iris_Testing_Target     = []

for n in range(0, len(Iris_Data), total_length):
    Iris_Training_Data.extend(  Iris_Data[n                     : n + training_length])
    Iris_Testing_Data.extend(   Iris_Data[n+training_length     : (n+total_length)])
    Iris_Training_Target.extend(Iris_Target[n                   : n + training_length])
    Iris_Testing_Target.extend( Iris_Target[n+training_length   : (n+total_length)])

Iris_Training_Data      = np.array(Iris_Training_Data)
Iris_Testing_Data       = np.array(Iris_Testing_Data) 
Iris_Training_Target    = np.array(Iris_Training_Target)
Iris_Testing_Target     = np.array(Iris_Testing_Target) 



#Normalizing the training data to have mu = 0, std = 1
mu  = Iris_Training_Data.mean(axis=0)
std = Iris_Training_Data.std(axis=0)

Iris_Training_Data  = (Iris_Training_Data-mu)/std
Iris_Testing_Data   = (Iris_Testing_Data-mu )/std



#Add bias component at the end of each datapoint: [x1,x2,...,1]
Iris_Training_Data  = np.hstack((Iris_Training_Data,    np.ones((len(Iris_Training_Data),   1))))
Iris_Testing_Data   = np.hstack((Iris_Testing_Data,     np.ones((len(Iris_Testing_Data),    1))))



#Converting labels to one-hot target vectors
T = np.zeros((len(Iris_Names),Iris_Training_Data.shape[0]))
T = T.T

for n in range(len(Iris_Training_Target)):
    T[n][Iris_Training_Target[n]] = 1



#Initializing W : (flower_name, weight)
rng = np.random.default_rng(0)
W   = rng.random((3,5))



#Training loop

def sigmoid(z):
    return 1/(1+np.e**(-z))

alpha = 0.01
epsilon = 10**(-6)
MSE_old = np.inf
upper_epoch_limit = 10**5

for n in range(upper_epoch_limit):
    Z = (W @ Iris_Training_Data.T).T
    G = sigmoid(Z)
    MSE = 1/2*np.sum((G-T)**2)
    D = (G-T) * G * (1-G)
    grad_MSE = D.T@Iris_Training_Data
    W = W - alpha*grad_MSE
    if abs((MSE - MSE_old) / MSE_old) < epsilon:
        break
    MSE_old = MSE




Z = ( W @ Iris_Testing_Data.T).T
G = sigmoid(Z)
Test_Guessed_Flower = np.argmax(G, axis=1)


# print(G.shape)

# print(Test_Guessed_Flower)


# print(Test_Guessed_Flower.shape)
# print(Iris_Testing_Target.shape)


Confusion_Matrix = np.zeros((3,3), dtype=int)
for n in range(len(Iris_Testing_Target)):
    Confusion_Matrix[Test_Guessed_Flower[n]][Iris_Testing_Target[n]] +=1

print(Confusion_Matrix)