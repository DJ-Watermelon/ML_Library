import math
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# Parse the files
def parse_data(file_name, num_samples):
    data = np.zeros((num_samples, 5)).astype(int)
    
    with open(file_name, "r") as file:
        i = 0
        for line in file:
            line = line.strip().split()
            for j in range(5):
                data[i][j] = int(line[j])
            i += 1
    return data

# compute P(DS | S, F, D, T) for every comb.
def compute_prob(T,DS,S,F,D):
    DS_cp = {}
    for s in range(2):
        for f in range(2):
            for d in range(2):
                for t in range(2):
                    DS_sum = 0.0
                    for c in range(3):
                        # compute joint probability P(T = t,S = s,F = f,D = d,C = c)
                        DS_cp[(s,f,d,t,c)] = T[t] * DS[c] * (S[t, c] if s else (1 - S[t, c])) \
                                            * (F[c] if f else (1 - F[c])) \
                                            * (D[c] if d else (1 - D[c]))
                        DS_sum += DS_cp[(s,f,d,t,c)]
                    # normalizing term      
                    DS_cp[(s,f,d,t)] = DS_sum
    return DS_cp

# Perform EM algorithm
def EM(data, T, DS, S, F, D, tol):
    prev_jp = 0
    jp = float('inf')
    
    while True:
    ## Expectation - hidden variables fixed     
        DS_cp = compute_prob(T,DS,S,F,D)
        
        if np.abs(jp - prev_jp) < tol:
            return DS_cp
        prev_jp = jp
        
    ## Maximization
        # Sum conditional and joint prob. over each data point
        Tc = np.zeros(2)
        DSc = np.zeros(3)
        Sc = np.zeros((2, 3, 2))
        Fc = np.zeros((3, 2))
        Dc = np.zeros((3, 2))
        jp = 0.0
        
        for sample in data:
            s,f,d,t,c = sample
            norm = DS_cp[tuple(sample[:-1])] 
                
            if c < 0: # only consider unlabeled data
                for c in range(3):
                    prob = DS_cp[tuple(sample[:-1]) + (c,)]
                    jp += prob
                    prob /= norm # normalize         
                    Tc[t] += prob
                    DSc[c] += prob
                    Sc[t, c, s] += prob
                    Fc[c, f] += prob
                    Dc[c, d] += prob  
            else:
                jp += 1
                Tc[t] += 1
                DSc[c] += 1
                Sc[t, c, s] += 1
                Fc[c, f] += 1
                Dc[c, d] += 1
        
        # normalize values
        T = Tc / np.sum(Tc)
        DS = DSc / np.sum(DSc)
        S = (Sc / np.sum(Sc, axis=2, keepdims=True))[:, :, 1]
        F = (Fc / np.sum(Fc, axis=1, keepdims=True))[:, 1]        
        D = (Dc / np.sum(Dc, axis=1, keepdims=True))[:, 1]
       
# determine classification of samples
def classify(data,DS_cp):
    label = np.zeros(data.shape[0])
    for i, sample in enumerate(data):
        vals = tuple(sample[:-1])
        prob_label = np.zeros(3)
        # pick c that maximizes the likelihood
        for c in range(3):
            prob_label[c] = DS_cp[vals + (c,)]
        label[i] = np.argmax(prob_label)
    return label

# return accuracy of predictions
def accuracy(pred_label, true_label, n):
    return np.sum(pred_label == true_label) / n

num_class = 3
num_train_samples = 2000
num_test_samples = 100
tol = 0.01
        
trainData = parse_data("traindata.txt", num_train_samples)
testData = parse_data("testdata.txt", num_test_samples)
testLabel = testData[:,-1]    

# initialize hidden variables
T_init = np.array([0.9, 0.1]) # T[0] = Pr(T=False), T[1] = Pr(T = True)
DS_init = np.array([0.5, 0.25, 0.25]) # DS[0] = Pr(DS = None), DS[1] = Pr(DS = Mild), DS[0] = Pr(DS = Severe)
S_init = np.array([[0.002, 0.6, 0.75], [0.002, 0.05, 0.07]]) # Rows: T = False, True; Cols: DS = None, Mild, Severe 
F_init = np.array([0.0005, 0.8, 0.3]) # DS = None, Mild, Severe
D_init = np.array([0.001, 0.15, 0.9]) # DS = None, Mild, Severe

delta_vars = np.linspace(0, 4, 21)

# mean and standard deviation of test accuracy over trials for a given delta
delta_mean = np.zeros(21)
delta_std = np.zeros(21)
delta_mean_EM = np.zeros(21)
delta_std_EM = np.zeros(21) 

for i, delta in enumerate(delta_vars):
    trial_acc = np.zeros(20)
    trial_acc_EM = np.zeros(20) 
    # run EM and compute test accuracy for 20 trials
    for j in range(20):
        # introduce noise to data
        T = T_init + np.random.rand(T_init.size) * delta
        T /= np.sum(T)
        
        DS = DS_init + np.random.rand(DS_init.size) * delta
        DS /= np.sum(DS)
        
        S_delta = np.random.rand(S_init.shape[0], S_init.shape[1]) * delta
        S = S_init + S_delta
        S_delta += np.random.rand(S_init.shape[0], S_init.shape[1]) * delta
        S /= (1 + S_delta)
        
        F_delta = np.random.rand(F_init.size) * delta
        F = F_init + F_delta
        F_delta += np.random.rand(F_init.size) * delta
        F /= (1 + F_delta)
        
        D_delta = np.random.rand(D_init.size) * delta
        D = D_init + D_delta
        D_delta += np.random.rand(D_init.size) * delta
        D /= (1 + D_delta)
        
        # compute accuracy before EM
        DS_cp = compute_prob(T, DS, S, F, D)
        trial_acc[j] = accuracy(classify(testData, DS_cp), testLabel, num_test_samples) * 100
        
        # apply EM and compute accuracy
        DS_cp_EM = EM(trainData, T, DS, S, F, D, tol)
        trial_acc_EM[j] = accuracy(classify(testData, DS_cp_EM), testLabel, num_test_samples) * 100
        
    # compute mean and standard deviation
    delta_mean[i] = np.mean(trial_acc)
    delta_std[i] = np.std(trial_acc)
    delta_mean_EM[i] = np.mean(trial_acc_EM)
    delta_std_EM[i] = np.std(trial_acc_EM)

# plot
plt.errorbar(delta_vars, delta_mean, yerr=delta_std, capsize=5, label="Before EM")
plt.errorbar(delta_vars, delta_mean_EM, yerr=delta_std_EM, capsize=5, label="After EM")
plt.xlabel("Noise Variable Delta")
plt.ylabel("Accuracy (%)")
plt.title("Mean Accuracy Over 20 Trials for Noise Variable Delta")
plt.grid(True)
plt.legend()
plt.show()
