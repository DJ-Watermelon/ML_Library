import math
import heapq
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

num_words = 6968

# Parse the files
def parse_data(data_file_name, label_file_name, num_samples):
    data = np.zeros((num_samples, num_words))
    label= np.zeros(num_samples)
    
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

# Decision tree struct
class DecisionNode:
    def __init__(self):
        self.conditions = [] # tuple (word, {0,1}) 1 means c=true, 0 means c=false
        self.isleaf = True
        self.label = None 
        self.split_on = None
        self.true_node = None
        self.false_node = None
        self.info_gain = None
        self.indices = None
        
    def __lt__(self, other):
        return self.info_gain > other.info_gain
    
    def display(self):  # tuple (word, {0,1}) 1 means c=true, 0 means c=false
        print(f"Label: {self.label}")
        print(f"IG: {self.info_gain}" )

# returns the majority point estimate of the labels   
def point_estimate(ys, indices):
    class_1 = 0
    class_2 = 0
    for i in indices:
        if int(ys[i]) == 1:
            class_1 += 1
        else:
            class_2 += 1
    return 1 if class_1 >= class_2 else 2    

# Computes the information content on labels
def info_content(ys, indices):
    n = len(indices)
    n1 = 0
    n2 = 0
    for i in indices:
        if int(ys[i]) == 1:
            n1 += 1
        else:
            n2 += 1
            
    if n1 and n2:
        p1 = n1 / n
        p2 = n2 / n
        return -(p1 * math.log2(p1) + p2 * math.log2(p2))
    else:
        return - math.log2(1) 

# compute the (non-constant portion of) average info_gain and weighted info_gain
def info_gain(ys, indices_1, indices_2):
    info_1 = info_content(ys, indices_1)
    info_2 = info_content(ys, indices_2)
    avg_info_gain = 0.5 * (info_1 + info_2)
    n1 = len(indices_1)
    n2 = len(indices_2)
    n = n1 + n2
    if n1 and n2:
        weighted_info_gain = (n1 / n * info_1) + (n2 / n * info_2)
    elif n1:
        weighted_info_gain = info_1
    else: 
        weighted_info_gain = info_2
    return avg_info_gain, weighted_info_gain

# classify data point by recursing down the decision tree
def classify(tree, xs, i):
    node = tree
    if node.isleaf:
        return node.label
    else:
        return classify(tree.true_node, xs, i) if xs[i][node.split_on] else classify(tree.false_node, xs, i)

# returns the percentage of data classified correctly
def accuracy(xs, ys, tree):
    n = len(ys)
    correct = 0
    for i in range(n):
        if classify(tree, xs, i) == int(ys[i]):
            correct += 1
    return correct / n

# return best feature to split on and the information gain
def select_best_split(xs, ys, indices, cond, info_gain_type):
    # get indices of points that satisfy all cond  
    max_info_gain = float('-inf')   
    
    for c in range(xs.shape[1]):
        # skip if already a condition
        if any(t[0] == c for t in cond):
            continue
        c_true = [i for i in indices if int(xs[i][c]) == 1]
        c_false = [i for i in indices if int(xs[i][c]) == 0]
        
        gain = -info_gain(ys, c_true, c_false)[info_gain_type]
        if gain > max_info_gain:
            max_info_gain = gain
            argmax = c
    return argmax, info_content(ys, indices) + max_info_gain

# Build the tree from the data
def train_decision_tree(xs, ys, test_xs, test_ys, words, info_gain_type):
    pq = []
    train_accuracy = []
    test_accuracy = []
    # Create root node

    c, info_gain = select_best_split(xs, ys, range(1500), [], info_gain_type)
    tree = DecisionNode()
    tree.label = point_estimate(ys, range(1500))
    tree.split_on = c
    tree.info_gain = info_gain
    tree.indices = range(1500)
    tree.display()
    heapq.heappush(pq, tree) # negative info gain for max-heap

    for i in range(100):
        train_accuracy.append(accuracy(xs, ys, tree))
        test_accuracy.append(accuracy(test_xs, test_ys, tree))
        
        # select best node to split on
        node = heapq.heappop(pq)
        print(f"prev conditions: {node.conditions}")
        print(f"Node #{i}, split on: {words[node.split_on]}, IG: {node.info_gain}")
        
        # Create left node
        print("Left Node:")
        true_cond = node.conditions + [(node.split_on, 1)]
        true_indices = [i for i in node.indices if int(xs[i][node.split_on]) == 1]
        c_true, info_gain = select_best_split(xs, ys, true_indices, true_cond, info_gain_type)
        
        true_node = DecisionNode()
        true_node.conditions = true_cond
        true_node.label = point_estimate(ys, true_indices)
        true_node.split_on = c_true
        true_node.info_gain = info_gain
        true_node.indices = true_indices
        
        true_node.display()
        heapq.heappush(pq, true_node)
        
        # Create right node
        print("Right Node:")
        false_cond = node.conditions + [(node.split_on, 0)]
        false_indices = [i for i in node.indices if int(xs[i][node.split_on]) == 0]
        c_false, info_gain = select_best_split(xs, ys, false_indices, false_cond, info_gain_type)
    
        false_node = DecisionNode()
        false_node.conditions = false_cond
        false_node.label = point_estimate(ys, false_indices)
        false_node.split_on = c_false
        false_node.info_gain = info_gain
        false_node.indices = false_indices
        
        false_node.display()
        heapq.heappush(pq, false_node)
        
        # Update original node as an internal node
        node.isleaf = False
        node.true_node = true_node
        node.false_node = false_node
    return train_accuracy, test_accuracy

# Get training and test data
words = []
with open("words.txt", "r") as file:
    for line in file:
        value = line.strip().split()[0]
        words.append(value)
trainData, trainLabel = parse_data("trainData.txt", "trainLabel.txt", 1500)
testData, testLabel = parse_data("testData.txt", "testLabel.txt", 1500)   

train_accuracy, test_accuracy = train_decision_tree(trainData, trainLabel, testData, testLabel, words, 1)

plt.plot(range(1, 101), train_accuracy, color='green', label='Training')
plt.plot(range(1, 101), test_accuracy, color='red', label='Testing')
plt.title('Weighted Information Gain Decision Tree')
plt.xlabel('Number of Nodes')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.show()