import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import time
import torch




#--------------------------Loading Data-----------------------------
def load_files():
    data = scipy.io.loadmat("data_all.mat")

    templatev      = (data["trainv"]/255).astype(np.float32)
    template_lab    = data["trainlab"].ravel()
    num_train   = data["num_train"]

    testv       = (data["testv"]/255).astype(np.float32)
    test_lab     = data["testlab"].ravel()
    num_test    = data["num_test"]

    return templatev, template_lab, num_train, testv, test_lab, num_test




#--------------------------Performing KNN/NN (NB: K=1 is equivalent to NN)-----------------------------
def KNN(templatev, templatelab, testv, test_lab, GPU=True, K=1):

    start = time.time()

    #Calculate euclidian dinstances (either with CPU or GPU):
    if GPU:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        templatev = torch.tensor(templatev, dtype=torch.float32, device=device)
        testv = torch.tensor(testv, dtype=torch.float32, device=device)

        distances = torch.cdist(testv, templatev)
        distances = distances.cpu().numpy().T 
    else:
        distances = scipy.spatial.distance.cdist(templatev, testv, metric="euclidean")
    
    #Finding K closest templates
    KNN_idx = np.argsort(distances, axis=0)[:K, :]
    KNN_labels = templatelab[KNN_idx]


    #Predics label (based on K- closest templates)
    predicted_lab = np.zeros(KNN_labels.shape[1], dtype=int)

    for n in range(KNN_labels.shape[1]):
        votes = np.bincount(KNN_labels[:, n], minlength=10)
        
        predicted_lab[n] = np.argmax(votes)


 
    #Calculate the Confusion Matrix and Error Rate
    Confusion_Matrix = np.zeros((10, 10), dtype=int)
    
    for n in range(len(predicted_lab)):

        Confusion_Matrix[test_lab[n]][predicted_lab[n]] +=1

    Total_tests = np.sum(Confusion_Matrix)
    Num_error   = Total_tests - np.trace(Confusion_Matrix)
    Error_rate = Num_error/Total_tests

    end = time.time()
    time_taken = end-start

    return Confusion_Matrix, Error_rate, predicted_lab, time_taken, KNN_idx





#--------------------------Clustering templates-----------------------------

def Clustering(M, templatev, template_lab, num_train):

    templatev, template_lab, num_train, _, _, _    = load_files()
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





#--------------------------Plotting numbers-----------------------------
def Plot_number(testv, test_lab, predicted_lab, num_plots = (9,9), templatev=None, templatelab=None, KNN_idx=None, show_difference=0, show_K_closest_samples=0, K=1, M=None):

    rng = np.random.default_rng(seed=42)
    info = f"K={K}" + (f", M={M}" if M is not None else "")

    # Plot wrong predictions (regular grid)
    wrong_idx = np.where(test_lab != predicted_lab)[0]
    random_wrong_idx = rng.choice(wrong_idx, size=min(num_plots[1], len(wrong_idx)), replace=False)

    for batch_start in range(0, len(random_wrong_idx), 9):
        batch_indices = random_wrong_idx[batch_start:batch_start + 9]
        _, axes = plt.subplots(3, 3, figsize=(10, 10))
        axes = axes.flatten()
        for i, n in enumerate(batch_indices):
            axes[i].imshow(testv[n].reshape(28, 28), cmap="gray")
            axes[i].set_title(f"Correct: {test_lab[n]} | Predicted: {predicted_lab[n]}")
            axes[i].axis('off')
        for i in range(len(batch_indices), 9):
            axes[i].axis('off')
        plt.suptitle(f"Wrong Predictions ({info})", fontsize=14)
        plt.tight_layout()
        plt.show()

    # Plot correct predictions (regular grid)
    correct_idx = np.where(test_lab == predicted_lab)[0]
    random_correct_idx = rng.choice(correct_idx, size=min(num_plots[0], len(correct_idx)), replace=False)

    for batch_start in range(0, len(random_correct_idx), 9):
        batch_indices = random_correct_idx[batch_start:batch_start + 9]
        _, axes = plt.subplots(3, 3, figsize=(10, 10))
        axes = axes.flatten()
        for i, n in enumerate(batch_indices):
            axes[i].imshow(testv[n].reshape(28, 28), cmap="gray")
            axes[i].set_title(f"Correct: {test_lab[n]} | Predicted: {predicted_lab[n]}")
            axes[i].axis('off')
        for i in range(len(batch_indices), 9):
            axes[i].axis('off')
        plt.suptitle(f"Correct Predictions ({info})", fontsize=14)
        plt.tight_layout()
        plt.show()

    # Plot test|NN|diff for show_difference samples
    if show_difference > 0 and templatev is not None and KNN_idx is not None:
        diff_wrong = rng.choice(wrong_idx, size=min(show_difference, len(wrong_idx)), replace=False)
        diff_correct = rng.choice(correct_idx, size=min(show_difference, len(correct_idx)), replace=False)
        for label, indices in [("Wrong", diff_wrong), ("Correct", diff_correct)]:
            for batch_start in range(0, len(indices), 3):
                batch_indices = indices[batch_start:batch_start + 3]
                _, axes = plt.subplots(3, 3, figsize=(10, 10))
                for i, n in enumerate(batch_indices):
                    nn = KNN_idx[0, n]
                    axes[i, 0].imshow(testv[n].reshape(28, 28), cmap="gray")
                    axes[i, 0].set_title(f"Correct: {test_lab[n]} | Predicted: {predicted_lab[n]}")
                    axes[i, 0].axis('off')
                    axes[i, 1].imshow(templatev[nn].reshape(28, 28), cmap="gray")
                    axes[i, 1].set_title(f"NN: {templatelab[nn]}")
                    axes[i, 1].axis('off')
                    axes[i, 2].imshow((testv[n] - templatev[nn]).reshape(28, 28), cmap="bwr")
                    axes[i, 2].set_title("Difference")
                    axes[i, 2].axis('off')
                for i in range(len(batch_indices), 3):
                    for j in range(3): axes[i, j].axis('off')
                plt.suptitle(f"{label} Predictions - Difference ({info})", fontsize=14)
                plt.tight_layout()
                plt.show()

    if templatev is not None and KNN_idx is not None and show_K_closest_samples > 0:
        all_idx = np.arange(len(test_lab))
        all_selected = rng.choice(all_idx, size=min(show_K_closest_samples, len(all_idx)), replace=False)
        num_neighbors = KNN_idx.shape[0]
        for batch_start in range(0, len(all_selected), 4):
            batch = all_selected[batch_start:batch_start + 4]
            _, axes = plt.subplots(len(batch), 1 + num_neighbors, figsize=(3 * (1 + num_neighbors), 3 * len(batch)), squeeze=False)
            for i, n in enumerate(batch):
                axes[i, 0].imshow(testv[n].reshape(28, 28), cmap="gray")
                axes[i, 0].set_title(f"Test\nTrue:{test_lab[n]} Pred:{predicted_lab[n]}")
                axes[i, 0].axis('off')
                for k in range(num_neighbors):
                    axes[i, k + 1].imshow(templatev[KNN_idx[k, n]].reshape(28, 28), cmap="gray")
                    axes[i, k + 1].set_title(f"Neighbor {k+1}\n{templatelab[KNN_idx[k, n]]}")
                    axes[i, k + 1].axis('off')
            plt.suptitle(f"K-Closest Samples ({info})", fontsize=14)
            plt.tight_layout()
            plt.show()






#--------------------------Combining all functions into a single runnable code-----------------------------

def run_experiment(data_start=0, data_end=10000, GPU=True, num_plots =(0,0), num_runs=1, M=1, K=1, show_K_closest_samples=False, show_difference=False):

    Total_time = 0
    templatev, template_lab, num_train, testv, test_lab, num_test    = load_files()

    if M > 1:
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
        Confusion_Matrix, Error_rate, predicted_lab, time_taken, KNN_idx = KNN(templatev, template_lab, testv[start:end], test_lab[start : end], GPU=GPU, K=K)
        neighbor_args = dict(templatev=templatev, templatelab=template_lab, KNN_idx=KNN_idx) if (show_K_closest_samples or show_difference) else {}
        Plot_number(testv[start:end], test_lab[start : end], predicted_lab, num_plots=num_plots, show_difference=show_difference, show_K_closest_samples=show_K_closest_samples, K=K, M=M, **neighbor_args)

        Total_Confusion_Matrix += Confusion_Matrix
        Total_Error_rate += Error_rate/num_runs
        Total_time += time_taken
    return Total_Confusion_Matrix, Total_Error_rate, Total_time









#--------------------------Main-----------------------------

def main():
    
    #--------------------------Changeable parameters-----------------------------

    #Default parameters

    data_length = 1000     #Number of test samples used
    GPU = True              #Note that this is only affects KNN. Can lead to faster processing time, especially for big data_length and clusterin=False
    num_runs=1              #Edit this to make the program split the test samples run into num_runs batches. 


    Run_Typical_Examples = True #runs the typical values of M and K (that were used in the report)

    #Clustering: 
    M = 1           #M=1 means no clustering, M>1 means clustering with M clusters per class

    #NN/KNN
    K=1    #KNN parameter. If K = 1, it will run as a NN classifier

    #Plotting
    num_correct_plots, num_wrong_plots = (9,9) #Plots image of num_correct_plots number of correctly classified digits and num_wrong_plots number of incorrectly classified digits 
    show_K_closest_samples = 1  #Plots the K closest templates for this many samples (0 = disabled)
    show_difference = 1  #Plots test|NN|diff layout for this many samples (0 = disabled, only works for NN)


    #--------------------------Running the experiment-----------------------------
    if Run_Typical_Examples:

        print("---------------------------------------")
        Confusion_Matrix, Error_rate, time_taken = run_experiment(0, data_length, GPU, (9, 9), num_runs=1, M=1, K=1, show_K_closest_samples=0, show_difference=3)
        print("K=",K,"M=",M)
        print("Total Time:        ", time_taken, "seconds")
        print("Error_rate: ", Error_rate)
        print(Confusion_Matrix)

        print("---------------------------------------")
        Confusion_Matrix, Error_rate, time_taken = run_experiment(0, data_length, GPU, (0, 0), num_runs=1 , M=64,K=1, show_K_closest_samples=0, show_difference=0)
        print("K=",K,"M=",M)
        print("Total Time:        ", time_taken, "seconds")
        print("Error_rate: ", Error_rate)
        print(Confusion_Matrix) 

        print("---------------------------------------")
        Confusion_Matrix, Error_rate, time_taken = run_experiment(0, data_length, GPU, (0, 0), num_runs=1 , M=64,K=7, show_K_closest_samples=3,  show_difference=0)
        print("K=",K,"M=",M)
        print("Total Time:        ", time_taken, "seconds")
        print("Error_rate: ", Error_rate)
        print(Confusion_Matrix)


        
    else:
        Confusion_Matrix, Error_rate, time_taken = run_experiment(0, data_length, GPU, (num_correct_plots, num_wrong_plots), num_runs=num_runs, M=M, K=K, show_K_closest_samples=show_K_closest_samples, show_difference=show_difference)
        print("Time:        ", time_taken, "seconds")
        print("Error_rate: ", Error_rate)
        print(Confusion_Matrix)




main()
