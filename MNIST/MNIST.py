import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import time
import torch

# Load the .mat file
def load_files():
    data = scipy.io.loadmat("data_all.mat")

    templatev      = (data["trainv"]/255).astype(np.float32)
    template_lab    = data["trainlab"].ravel()
    num_train   = data["num_train"]

    testv       = (data["testv"]/255).astype(np.float32)
    test_lab     = data["testlab"].ravel()
    num_test    = data["num_test"]

    return templatev, template_lab, num_train, testv, test_lab, num_test


#Performing NN on data
def NN(templatev, templatelab, testv, test_lab, GPU=True, K=1):

    start = time.time()

    if GPU:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        templatev = torch.tensor(templatev, dtype=torch.float32, device=device)
        testv = torch.tensor(testv, dtype=torch.float32, device=device)

        distances = torch.cdist(testv, templatev)
        distances = distances.cpu().numpy().T 
    else:
        distances = scipy.spatial.distance.cdist(templatev, testv, metric="euclidean")
    

    KNN_idx = np.argsort(distances, axis=0)[:K, :]
    # print(KNN_idx)
    KNN_labels = templatelab[KNN_idx]

    predicted_lab = np.zeros(KNN_labels.shape[1], dtype=int)

    for n in range(KNN_labels.shape[1]):
        votes = np.bincount(KNN_labels[:, n], minlength=10)
        
        predicted_lab[n] = np.argmax(votes)

    # NN_idx = np.argmin(distances, axis=0)
 
    # predicted_lab = templatelab[NN_idx]
    Confusion_Matrix = np.zeros((10, 10), dtype=int)
    
    for n in range(len(predicted_lab)):
        # print(test_lab)
        # print(predicted_lab)
        Confusion_Matrix[test_lab[n]][predicted_lab[n]] +=1

    Total_tests = np.sum(Confusion_Matrix)
    Num_error   = Total_tests - np.trace(Confusion_Matrix)
    Error_rate = Num_error/Total_tests

    end = time.time()

    time_taken = end-start

    

    return Confusion_Matrix, Error_rate, predicted_lab, time_taken


#Plotting random batch of wrong and correct predictions
def Plot_number(testv, test_lab, predicted_lab, num_plots = (3,3)):

    rng = np.random.default_rng(seed=42)
    wrong_idx = np.where(test_lab != predicted_lab)[0]
    random_wrong_idx = rng.choice(wrong_idx, size=min(num_plots[1], len(wrong_idx)), replace=False)


    for n in random_wrong_idx:
        x = testv[n, :].reshape((28, 28))
        plt.title(f"Correct Number: {test_lab[n]} \nPredicted Number: {predicted_lab[n]}")
        plt.imshow(x, cmap="gray")
        plt.show()
    
    correct_idx = np.where(test_lab == predicted_lab)[0]
    random_correct_idx = rng.choice(correct_idx, size=min(num_plots[1], len(correct_idx)), replace=False)


    for n in random_correct_idx:
        x = testv[n, :].reshape((28, 28))
        plt.title(f"Correct Number: {test_lab[n]} \nPredicted Number: {predicted_lab[n]}")
        plt.imshow(x, cmap="gray")
        plt.show()

#Combining the load, NN and plotting into a single runnable function
def run_experiment(data_start=0, data_end=10000, GPU=True, num_plots =(0,0), num_runs=1, clustering=False, M=64, K=1):

    Total_time = 0
    templatev, template_lab, num_train, testv, test_lab, num_test    = load_files()

    if clustering:
        start = time.time()
        templatev, template_lab, num_train =Clustering(M, templatev, template_lab, num_train)
        end = time.time()
        print("Clustering time: ",(end-start))
        Total_time += end-start

    steps = (data_end-data_start)//num_runs
    Total_Confusion_Matrix = np.zeros((10, 10), dtype=int)
    Total_Error_rate = 0
    

    for run in range(num_runs):
        
        start, end = data_start+steps*run, data_start+steps*(run+1)
        Confusion_Matrix, Error_rate, predicted_lab, time_taken = NN(templatev, template_lab, testv[start:end], test_lab[start : end], GPU=GPU, K=K)
        Plot_number(testv[start:end], test_lab[start : end], predicted_lab, num_plots = num_plots)

        Total_Confusion_Matrix += Confusion_Matrix
        Total_Error_rate += Error_rate/num_runs
        Total_time += time_taken
    return Total_Confusion_Matrix, Total_Error_rate, Total_time



def Clustering(M, templatev, template_lab, num_train):

    templatev, template_lab, num_train, testv, test_lab, num_test    = load_files()
    num_train = M*10

    all_Ci = []
    all_labels = []

    for i in range(10):
        templatev_i = templatev[template_lab==i]

        kmeans = KMeans(n_clusters=M, random_state=42, n_init="auto")
        idxi = kmeans.fit_predict(templatev_i)
        Ci = kmeans.cluster_centers_
        all_Ci.append(Ci)
        all_labels.append(np.full(M, i))
    clustered_templatev = np.vstack(all_Ci)
    clustered_templatelab = np.hstack(all_labels)


    return clustered_templatev, clustered_templatelab, num_train







def main():
    
    ##Default parameters
    data_start = 0
    data_end = 1000
    num_correct_plots, num_wrong_plots = (0,0)
    GPU = True #Note that this is only affects NN
    num_runs=1

    #Clustering parameters
    clustering = True
    M = 64

    #KNN parameter. If K = 1 => NN
    K=7

    Confusion_Matrix, Error_rate, time_taken = run_experiment(data_start, data_end, GPU, (num_correct_plots, num_wrong_plots), num_runs=num_runs, clustering=clustering, M=M, K=K)
    print("Time:        ", time_taken, "seconds")
    print("Error_rate: ", Error_rate)
    print(Confusion_Matrix)


main()
