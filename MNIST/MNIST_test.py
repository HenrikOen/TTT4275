import scipy.io
import numpy as np
import matplotlib.pyplot as plt


import time

import torch

# Load the .mat file
data = scipy.io.loadmat("data_all.mat")

templatev      = (data["trainv"]/225).astype(np.float32)
template_lab    = data["trainlab"].ravel()
num_train   = data["num_train"]

testv       = (data["testv"]/255).astype(np.float32)
testlab     = data["testlab"].ravel()
num_test    = data["num_test"]

print(templatev.shape)







def NN_CPU(templatev, templatelab, testv, testlab):

    start = time.time()
    distances = scipy.spatial.distance.cdist(templatev, testv, metric="euclidean")
    NN_idx = np.argmin(distances, axis=0)
    predicted_lab = templatelab[NN_idx]
    Confusion_Matrix = np.zeros((10, 10), dtype=int)
    

    


    for n in range(len(predicted_lab)):
        # print(testlab)
        # print(predicted_lab)
        Confusion_Matrix[testlab[n]][predicted_lab[n]] +=1

    Total_tests = np.sum(Confusion_Matrix)
    Num_error   = Total_tests - np.trace(Confusion_Matrix)
    Error_rate = Num_error/Total_tests

    end = time.time()

    print("Time:", end - start, "seconds")

    print(Error_rate)
    print(Confusion_Matrix)

    return Confusion_Matrix, Error_rate, predicted_lab, NN_idx


    



def NN_GPU(templatev, templatelab, testv, testlab):

    device = "cuda" if torch.cuda.is_available() else "cpu"


    templatev = torch.tensor(templatev, dtype=torch.float32, device=device)
    testv = torch.tensor(testv, dtype=torch.float32, device=device)
    templatelab = torch.tensor(templatelab.ravel(), dtype=torch.long, device=device)

    
    start = time.time()
    distances = torch.cdist(testv, templatev)
    NN_idx = torch.argmin(distances, dim=1)
    predicted_lab = templatelab[NN_idx]
    Confusion_Matrix = np.zeros((10, 10), dtype=int)
    
    for n in range(len(predicted_lab)):
        # print(testlab)
        # print(predicted_lab)
        Confusion_Matrix[testlab[n]][predicted_lab[n]] +=1

    Total_tests = np.sum(Confusion_Matrix)
    Num_error   = Total_tests - np.trace(Confusion_Matrix)
    Error_rate = Num_error/Total_tests

    end = time.time()

    print("Time:", end - start, "seconds")

    print(Error_rate)
    print(Confusion_Matrix)
    return Confusion_Matrix, Error_rate, predicted_lab.cpu().numpy(), predicted_lab.cpu().numpy()



def Plot_number(testv, testlab, predicted_lab):

    rng = np.random.default_rng(seed=42)
    wrong_idx = np.where(testlab != predicted_lab)[0]
    random_wrong_idx = rng.choice(wrong_idx, size=1, replace=False)


    for n in random_wrong_idx:
        x = testv[n, :].reshape((28, 28))
        plt.title(f"Correct Number: {testlab[n]} \nPredicted Number: {predicted_lab[n]}")
        plt.imshow(x, cmap="gray")
        plt.show()
    
    correct_idx = np.where(testlab == predicted_lab)[0]
    random_correct_idx = rng.choice(correct_idx, size=0, replace=False)


    for n in random_correct_idx:
        x = testv[n, :].reshape((28, 28))
        plt.title(f"Correct Number: {testlab[n]} \nPredicted Number: {predicted_lab[n]}")
        plt.imshow(x, cmap="gray")
        plt.show()

Confusion_Matrix, Error_rate, predicted_lab, predicted_lab = NN_GPU(templatev, template_lab, testv[0:10000], testlab[0:10000])

Plot_number(testv[0:10000], testlab[0:10000], predicted_lab)