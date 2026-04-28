import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix

iris = load_iris()


# print(iris)

X = iris.data #training Data [150][4] : flower.nr, feature
Y = iris.target

training_idx = []
testing_idx = []

for c in range(3):
    start = 50*c
    training_idx.extend(range(start, start+30))
    testing_idx.extend(range(start+30, start+50))

X_training = X[training_idx]
Y_training = Y[training_idx]
X_testing  = X[testing_idx] 
Y_testing  = Y[testing_idx]

mu_training = X_training.mean(axis=0)
sigma_training = X_training.std(axis=0)
X_training = (X_training-mu_training)/sigma_training
X_testing = (X_training-mu_training)/sigma_training

# Make x = [x1,x2,x3,x4,1]
X_train_aug = np.c_[X_training, np.ones(X_training.shape[0])]
X_test_aug = np.c_[X_testing, np.ones(X_testing.shape[0])]

#Target vectors
tk_setosa =     [1,0,0]
tk_versicolor = [0,1,0]
tk_virginica =  [0,0,1]

T_training = np.zeros((len(Y_training), 3))
T_training[np.arange(len(Y_training)), Y_training] = 1

#Initializing W to random values: standard normal *0.01
rng = np.random.default_rng(0)
W = 0.01 * rng.standard_normal((3, 5))

def sigmoid(z):
    return 1/(1+np.e**(-z))

MSE_old = np.inf
MSE_history = []

for n in range(10000):
    Z = X_train_aug @ W.T
    G = sigmoid(Z)

    #MSE
    E = G - T_training
    MSE_new = 0.5*np.sum(E**2)
    MSE_history.append(MSE_new)

    if abs(MSE_old - MSE_new) < 1e-6:
        break

    MSE_old = MSE_new

    D = (G - T_training) * G * (1 - G)
    grad_W = D.T @ X_train_aug


    alpha = 0.1
    W = W - alpha * grad_W


G_train = sigmoid(X_train_aug @ W.T)
y_train_pred = np.argmax(G_train, axis=1)

G_test = sigmoid(X_test_aug @ W.T)
y_test_pred = np.argmax(G_test, axis=1)



C_train = confusion_matrix(Y_training, y_train_pred)
C_test = confusion_matrix(Y_testing, y_test_pred)


train_error = np.mean(y_train_pred != Y_training)
test_error = np.mean(y_test_pred != Y_testing)

print(test_error)