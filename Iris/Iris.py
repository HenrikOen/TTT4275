import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import seaborn as sns
np.set_printoptions(suppress=True)


#--------------------------Importing and splitting data-----------------------------
def load_and_split_data(split = 30, num_flowers = 3 , inverted=False): 

    #Loading data
    Iris        = load_iris()
    Data   = Iris.data
    Target = Iris.target
    Names  = Iris.target_names

    class_length = len(Data) // num_flowers
    training_length = split
    testing_length = class_length - split

    Training_Data = []
    Testing_Data = []
    Training_Target = []
    Testing_Target = []

    #split the data into trainig and target
    for start in range(0, len(Data), class_length):
        end = start + class_length

        if not inverted:
            train_slice = slice(start, start + training_length)
            test_slice  = slice(start + training_length, end)
        else:
            test_slice  = slice(start, start + testing_length)
            train_slice = slice(start + testing_length, end)

        Training_Data.extend(Data[train_slice])
        Testing_Data.extend(Data[test_slice])
        Training_Target.extend(Target[train_slice])
        Testing_Target.extend(Target[test_slice])

    return np.array(Training_Data), np.array(Testing_Data), np.array(Training_Target), np.array(Testing_Target), Names




#--------------------------Helper function for prepering data-----------------------------

# Normalizing the training data to have mu = 0, std = 1
def Normalizing_Data(Training_Data : np.array, Testing_Data : np.array): 

    mu  = Training_Data.mean(axis=0)
    std = Training_Data.std(axis=0)
    Training_Data = (Training_Data-mu)/std
    Testing_Data  = (Testing_Data-mu )/std

    return Training_Data, Testing_Data


#Add bias component at the end of each datapoint: [x1,x2,...,1]
def Adding_Bias_Component(Training_Data : np.array , Testing_Data : np.array): 
    Training_Data  = np.hstack((Training_Data,    np.ones((len(Training_Data),   1))))
    Testing_Data   = np.hstack((Testing_Data,     np.ones((len(Testing_Data),    1))))
    return Training_Data, Testing_Data


#Converting labels to one-hot target vectors
def one_hot_target(Training_Data : np.array, Training_Target : np.array, Training_Names : np.array): 
    T = np.zeros((len(Training_Names),Training_Data.shape[0]))
    T = T.T
    
    for n in range(len(Training_Target)):
        T[n][Training_Target[n]] = 1
    return T



#Removing spesific features from the data
def Cutting_Features(Training_Data, Testing_Data, *feature_nums, ):
    feature_nums = sorted(feature_nums, reverse=True)
    for feature in feature_nums:
        Training_Data   = np.delete(Training_Data, feature, axis=1)
        Testing_Data    = np.delete(Testing_Data,  feature, axis=1)
    return Training_Data, Testing_Data



 #Initializing W : (flower_name, weight)
def Initialize_W(num_flowers, num_terms):
    rng = np.random.default_rng(0)
    W   = rng.random((num_flowers,num_terms))
    return W


def sigmoid(z):
    return 1/(1+np.e**(-z))



#--------------------------Calling helper function for prepering data before training loop-----------------------------

#Performing Normalization, adding bias component and one-hot-target on the data
def processing_Data(Training_Data, Testing_Data,
                    Training_Target,
                    Names):
    
    Training_Data, Testing_Data = Normalizing_Data(Training_Data, Testing_Data)
    Training_Data, Testing_Data = Adding_Bias_Component(Training_Data, Testing_Data)
    T = one_hot_target(Training_Data, Training_Target, Names)
    W = Initialize_W(len(Names), Training_Data.shape[1])
    return Training_Data, Testing_Data, T, W




#--------------------------Training loop-----------------------------

def training_Loop(Training_Data, W, T,
                    alpha       =0.1,
                    epsilon     =10**(-6),
                    upper_epoch_limit = 2*10**4):
    
    MSE_old = np.inf
    MSE_log = []
    for n in range(upper_epoch_limit):
        Z = (W @ Training_Data.T).T
        G = sigmoid(Z)
        MSE = 1/2*np.sum((G-T)**2)/(len(Training_Data))
        MSE_log.append(MSE)

        D = (G-T) * G * (1-G)
        grad_MSE = D.T@Training_Data
        W = W - alpha*grad_MSE

        if abs((MSE - MSE_old) / MSE_old) < epsilon:
            break
        MSE_old = MSE
    return MSE_log, W





#--------------------------Calculating Confusion Matrix+ Error Rate and plotting data-----------------------------

# Calculating confusion matrix and error rate
def Confusion_matrix_error_rate(Testing_Data : np.array , Testing_Target : np.array, Names, W : np.array):

    Z = ( W @ Testing_Data.T).T
    G = sigmoid(Z)
    Testing_result = np.argmax(G, axis=1)
    Confusion_Matrix = np.zeros((len(Names), len(Names)), dtype=int)
    for n in range(len(Testing_Target)):
        Confusion_Matrix[Testing_Target[n]][Testing_result[n]] +=1
    
    Total_tests = np.sum(Confusion_Matrix)
    Num_error   = Total_tests - np.trace(Confusion_Matrix)
    Erorr_rate = Num_error/Total_tests

    return Confusion_Matrix, Erorr_rate



#Plotting histogram of the feature for each of the flowers
def plot_feature_histogram(Training_Data, Training_Target, Testing_Data, Testing_Target, Names):
    Data = np.vstack((Training_Data, Testing_Data))
    Target = np.hstack((Training_Target, Testing_Target))
    feature_names = ["sepal length", "sepal width", "petal length", "petal width"]
    _, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        bins = np.linspace(0, 8, 80)
        for cls in range(len(Names)):
            ax.hist(Data[Target == cls, i], bins=bins, alpha=0.6, label=Names[cls], edgecolor='black')
        ax.set_title(feature_names[i])
        ax.set_xlabel("cm")
        ax.set_ylabel("count")
        ax.set_xlim(0, 8)
    axes[0].legend()
    plt.tight_layout()
    plt.show()


#Plotting the MSE over each epoch
def plot_MSE(MSE_logs, labels=None):
    for i, MSE_log in enumerate(MSE_logs):
        label = labels[i] if labels and i < len(labels) else None
        plt.plot(MSE_log, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    # plt.yscale("log")
    if labels:
        plt.legend()
    plt.show()



#--------------------------All in one function for running the experiment-----------------------------

#all in one function for performing the the loading, preparing and training of the model
def run_experiment(Num_training=30, alpha = 0.01, num_flowers=3, Inverted=False, removed_features = 0):
    feature_priority = [1, 0, 3, 2]
    features_to_remove = tuple(feature_priority[:removed_features])

    Training_Data, Testing_Data, Training_Target, Testing_Target, Names = load_and_split_data(Num_training, num_flowers, Inverted)
    Training_Data, Testing_Data = Cutting_Features(Training_Data, Testing_Data, *features_to_remove)
    Training_Data, Testing_Data, T, W = processing_Data(Training_Data, Testing_Data, Training_Target, Names)
    MSE_log, W= training_Loop(Training_Data, W, T, alpha)
    confusion_matrix_train, error_rate_train = Confusion_matrix_error_rate(Training_Data, Training_Target, Names, W)
    confusion_matrix, error_rate = Confusion_matrix_error_rate(Testing_Data, Testing_Target, Names, W)

    return MSE_log, error_rate, confusion_matrix, error_rate_train, confusion_matrix_train, Names



#--------------------------Main-----------------------------

def main():

    #--------------------------Changeable parameters-----------------------------
    
    #Change in features:
    alpha            = (0.1, )          #Step factors
    Inverted         = (False, True)         #decieds if the training should be the first or last 30 samples from the dataset
    removed_features = (0, 1, 2, 3)     #Number of removed features


    #Plots that should display
    plotting_features = True
    plotting_MSE = True


    #--------------------------#Looping through experiment of the parameters used-----------------------------
    MSE_log_log           = []
    confusion_matrix_log  = []
    error_rate_log        = []
    labels            = []

    for a in alpha:
        for I in Inverted:
            for r in removed_features:
                MSE_log, error_rate, confusion_matrix, error_rate_train, confusion_matrix_train, Names = run_experiment(alpha=a, Inverted=I, removed_features=r)
                MSE_log_log.append(MSE_log)
                confusion_matrix_log.append(confusion_matrix)
                error_rate_log.append(error_rate)
                label = f"alpha={a}, Inverted={I}, removed_features={r}"
                labels.append(label)
                
                print("---------------------------------------")
                print("Properties:", label)
                print(f"Error rate (test): {error_rate:.3f}")
                print("Confusion Matrix (test):\n", confusion_matrix)
                print(f"Train Error rate (training): {error_rate_train:.3f}")
                print("Train Confusion Matrix (training):\n", confusion_matrix_train)


    #Plotting
    if plotting_features:
        Training_Data, Testing_Data, Training_Target, Testing_Target, Names = load_and_split_data()
        plot_feature_histogram(Training_Data, Training_Target, Testing_Data, Testing_Target, Names)

    if plotting_MSE:
        plot_MSE(MSE_log_log, labels=labels)
    
    

    
main()
