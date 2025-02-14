import math
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

num_words = 3566
num_class = 2
num_train_samples = 1061
num_test_samples = 707

# Parse the files
def parse_data(data_file_name, label_file_name, num_samples):
    data = np.zeros((num_samples, num_words)).astype(int)
    label= np.zeros(num_samples).astype(int)
    
    with open(data_file_name, "r") as data_file:
        for line in data_file:
            indices = line.strip().split()
            i, j = int(indices[0]), int(indices[1])
            data[i-1][j-1] = 1
    
    k = 0        
    with open(label_file_name, "r") as label_file:
        for line in label_file:
            value = int(line.strip().split()[0])
            label[k] = value
            k += 1
    return data, label

# determine classification of samples
def classify(data, theta_class, theta_word):
    n = data.shape[0]
    pred_label = np.zeros(n).astype(int)
    for i in tqdm(range(n)):
        posterior = np.log(theta_class)
        for j in range(num_words):
            if data[i, j]: # true
                posterior += np.log(theta_word[j])
            else: # false
                posterior += np.log(1 - theta_word[j])
        pred_label[i] = np.argmax(posterior) + 1
    return pred_label

# return accuracy of predictions
def accuracy(pred_label, true_label, n):
    return np.sum(pred_label == true_label) / n


# Get training and test data and word dictionary
words = []
with open("words.txt", "r") as file:
    for line in file:
        value = line.strip().split()[0]
        words.append(value)
        
trainData, trainLabel = parse_data("trainData.txt", "trainLabel.txt", num_train_samples)
testData, testLabel = parse_data("testData.txt", "testLabel.txt", num_test_samples)   

cc = np.zeros(num_class) # num of samples per class
fc = np.zeros((num_class, num_words)) # num of samples for each feature per class

# get counts for training samples
for i in range(num_train_samples):
    label = trainLabel[i] - 1
    cc[label] += 1
    fc[label] += trainData[i]

# compute theta probabilities
theta_class = cc / num_train_samples  
theta_word = np.zeros((num_words, num_class))
for i in range(num_words):
    for c in range(num_class):
        theta_word[i, c] = (fc[c, i] + 1) / (cc[c] + 2)

print("Training Accuracy:", accuracy(classify(trainData, theta_class, theta_word) , trainLabel, num_train_samples))
print("Testing Accuracy:", accuracy(classify(testData, theta_class, theta_word), testLabel, num_test_samples))

# Compute measure of discrimination for each word
measure = np.zeros(num_words)
for i in range(num_words):
    measure[i] = np.abs(np.log(theta_word[i,0]) - np.log(theta_word[i,1]))


# Sort (in descending order) and print top ten most discriminative words
top_ten = measure.argsort()[::-1]
print("Top ten most discriminative words:")
for i in range(10):
    print(i + 1, words[top_ten[i]], measure[top_ten[i]])