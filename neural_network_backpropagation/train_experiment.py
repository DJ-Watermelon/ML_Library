# version 1.1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from neural_net import NeuralNetwork
from operations import *

def load_dataset(csv_path, target_feature):
    dataset = pd.read_csv(csv_path)
    t = np.expand_dims(dataset[target_feature].to_numpy().astype(float), axis=1)
    X = dataset.drop([target_feature], axis=1).to_numpy()
    return X, t

X, y = load_dataset("data/wine_quality.csv", "quality")

n_features = X.shape[1]
net = NeuralNetwork(n_features, [64, 32, 32, 16, 1], [ReLU(), ReLU(), ReLU(), Sigmoid(), Identity()], MeanSquaredError(), learning_rate=0.001)
epochs = 500

# Perform cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=486)

train_loss = np.zeros((5, 500))
mae = np.zeros(5)
for i, (train_indices, test_indices) in enumerate(kf.split(X)):
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    trained_W, epoch_losses = net.train(X_train, y_train, epochs)
    train_loss[i] = epoch_losses
    mae[i] = net.evaluate(X_test, y_test, mean_absolute_error)

print("MAE mean: ", np.mean(mae))
print("MAE std: ", np.std(mae))
plt.plot(np.arange(1, epochs+1), np.mean(train_loss, axis=0))
plt.xlabel("Epoch #")
plt.ylabel("Average Training Loss")
plt.title("Average Training Loss Over 5-Folds Cross Validation")
plt.grid()
plt.show()