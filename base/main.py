import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
import re
import csv
from datetime import datetime
import sys
import numpy as np
import string
import argparse
import csv
import pandas as pd
from tqdm import trange
import _pickle as pickle
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import sys
import gensim
from random import randrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import torchvision.transforms as transforms
from torch.autograd import Variable
import re
from torch.utils.data import DataLoader, Sampler
from torchvision import datasets,transforms
from nltk.stem import SnowballStemmer
torch.manual_seed(1)


def representTest(vocabulary, file, start, end):
    """Train a FNN_2 neural network with given learning rate, epoch number, and batch size.

    Parameters
    ----------
    n : int
        The number of nodes in the graph.
    learning_rate : float
        The learning rate for neural network.
    epoch : int
        Number of iterations over the whole train set.
    batch_size : int
        Number of train examples to feed the model every time.

    Returns
    -------
    None
    Saves the model in "FNN_2.pt".
    """
    df = pd.read_csv(file)
    total = 0
    testTwitters = df.text
    testAuthors = df.author
    representations = list()
    for j in range(start, end):
        twitter = testTwitters[j]
        toRemove = "`~!@#$%^*\(\)_+\}\{<>?:\",./;'\[\]\\-=\'1234567890"
        for i in toRemove:
            if i in twitter:
                twitter = twitter.replace(i, " ")
        content = twitter.split(" ")
        content = [x for x in content if x]
        represent = [0 for i in range(len(vocabulary) + 2)]
        for word in content:
            total += 1
            if word in vocabulary:
                represent[vocabulary.index(word)] += 1
            if testAuthors[j] == "democrat":
                represent[len(vocabulary) + 1] = 1
            else:
                represent[len(vocabulary) + 1] = 0
            # For bias
            represent[len(vocabulary)] = 1
        representations.append(represent)
    return representations


def representation(file, start, end):
    """Train a FNN_2 neural network with given learning rate, epoch number, and batch size.

    Parameters
    ----------
    n : int
        The number of nodes in the graph.
    learning_rate : float
        The learning rate for neural network.
    epoch : int
        Number of iterations over the whole train set.
    batch_size : int
        Number of train examples to feed the model every time.

    Returns
    -------
    None
    Saves the model in "FNN_2.pt".
    """
    df = pd.read_csv(file)
    print(df)
    sys.exit()
    twitters = df.text[start:end]
    vocabulary = list()
    sb = SnowballStemmer("english")
    representations = list()
    for twitter in twitters:
        toRemove = "`~!@#$%^*\(\)_+\}\{<>?:\",./;'\[\]\\-=\'1234567890"
        for i in toRemove:
            if i in twitter:
                twitter = twitter.replace(i, " ")
        content = twitter.split(" ")
        content = [sb.stem(x) for x in content if x]
        for word in content:
            if not (word in vocabulary):
                vocabulary.append(word)
        represent = list([0 for i in range(len(vocabulary))])
        for word in content:
            represent[vocabulary.index(word)] += 1
        representations.append(represent)

    print("Length of vocab ", len(vocabulary))
    # Fills in the gap and add addition tag that describes the political affinity of the author
    for i in range(len(representations)):
        represent = representations[i]
        # Fill the gap between represent length and the vocab length
        gap = len(vocabulary) - len(represent)
        represent.extend([0 for i in range(gap)])
        #For the bias
        represent.extend([1])
        author = df.author[i]
        if author == "democrat":
            represent.append(1)
        else:
            represent.append(0)
    return vocabulary, len(representations[0]), representations, df.label[start:end]


def findAccuracy(solution, start, end, file):
    """Train a FNN_2 neural network with given learning rate, epoch number, and batch size.

    Parameters
    ----------
    n : int
        The number of nodes in the graph.
    learning_rate : float
        The learning rate for neural network.
    epoch : int
        Number of iterations over the whole train set.
    batch_size : int
        Number of train examples to feed the model every time.

    Returns
    -------
    None
    Saves the model in "FNN_2.pt".
    """
    print("solution length", len(solution))
    df = pd.read_csv(file)
    final = 0
    for i in range(len(solution)):
        if solution[i] == df.label[i + start]:
            final += 1
    return final / len(solution)

def findMax(result):
    """Train a FNN_2 neural network with given learning rate, epoch number, and batch size.

    Parameters
    ----------
    n : int
        The number of nodes in the graph.
    learning_rate : float
        The learning rate for neural network.
    epoch : int
        Number of iterations over the whole train set.
    batch_size : int
        Number of train examples to feed the model every time.

    Returns
    -------
    None
    Saves the model in "FNN_2.pt".
    """
    index = 0
    for i in range(len(result)):
        if result[i] > result[index]:
            index = i
    return index + 1

class Net(nn.Module):
    """Train a FNN_2 neural network with given learning rate, epoch number, and batch size.

    Parameters
    ----------
    n : int
        The number of nodes in the graph.
    learning_rate : float
        The learning rate for neural network.
    epoch : int
        Number of iterations over the whole train set.
    batch_size : int
        Number of train examples to feed the model every time.

    Returns
    -------
    None
    Saves the model in "FNN_2.pt".
    """
    def __init__(self, vocabulary_length):
        super().__init__()
        self.fc1 = nn.Linear(vocabulary_length, 100)
        self.fc2 = nn.Linear(100, 17)
    def forward(self, x):
        x = self.fc2(F.relu(self.fc1(x)))
        return x


class Dataset(torch.utils.data.Dataset):
    """Train a FNN_2 neural network with given learning rate, epoch number, and batch size.

    Parameters
    ----------
    n : int
        The number of nodes in the graph.
    learning_rate : float
        The learning rate for neural network.
    epoch : int
        Number of iterations over the whole train set.
    batch_size : int
        Number of train examples to feed the model every time.

    Returns
    -------
    None
    Saves the model in "FNN_2.pt".
    """
    def __init__(self, filepath : str, dataLen : int, start: int):
        """Generate packet id for a packet

        Parameters
        ----------
        filepath : str
            Where dataset is stored.
        dataLen : int
            Length of training examples in the dataset.

        Returns
        -------
        Dataset
            An dataset object that stores training data and iterate through them more efficiently with 
        a DataLoader object. More on DataLoader, check: https://pytorch.org/docs/stable/data.html#map-style-datasets
        """
        self.start = start
        self.dataLen = dataLen

        self.file = torch.load(filepath)

    def __getitem__(self, index: int):
        # with open("training.txt", "rb") as file:
        #     train = pickle.loads(file.read())
        return self.file[index][0], self.file[index][1]

    def __len__(self):
        return self.dataLen

def validate(values):
    """Train a FNN_2 neural network with given learning rate, epoch number, and batch size.

    Parameters
    ----------
    n : int
        The number of nodes in the graph.
    learning_rate : float
        The learning rate for neural network.
    epoch : int
        Number of iterations over the whole train set.
    batch_size : int
        Number of train examples to feed the model every time.

    Returns
    -------
    None
    Saves the model in "FNN_2.pt".
    """
    # Unroll the dictionary
    learning_rate = values["learning_rate"]
    epochs = values["epochs"]
    batch_size = values["batch_size"]
    start = values["start"]
    end = values["end"]
    file = values["file"]
    nodes = values["nodes"]


#Trains the neural network
def trainNet(values):
    """Train a FNN_2 neural network with given learning rate, epoch number, and batch size.

    Parameters
    ----------
    n : int
        The number of nodes in the graph.
    learning_rate : float
        The learning rate for neural network.
    epoch : int
        Number of iterations over the whole train set.
    batch_size : int
        Number of train examples to feed the model every time.

    Returns
    -------
    None
    Saves the model in "FNN_2.pt".
    """
    # Unroll the dictionary
    learning_rate = values["learning_rate"]
    epochs = values["epochs"]
    batch_size = values["batch_size"]
    start = values["start"]
    end = values["end"]
    file = values["file"]
    nodes = values["nodes"]
    test_start = values["test_start"]
    test_end = values["test_end"]
    test_material = values["test_material"]

    net = Net(vocabulary_length = nodes)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate,  weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()

    dataset = Dataset(filepath = "representations.txt", dataLen = end - start, start = 10)
    loader = DataLoader(dataset,batch_size = 50)

    # Before training
    material = torch.load("representations.txt")
    solution = [findMax(net(j[0].float()).tolist()) for j in material]
    accuracy = findAccuracy(solution = solution, start = start, end = end, file = file)
    print("Accuracy is ", accuracy, " at epoch {}".format(0))
    test_solution = [findMax(net(j.float()).tolist()) for j in test_material]
    test_accuracy = findAccuracy(solution = test_solution, start = test_start, end = test_end, file = file)
    print("Test accuracy is ", test_accuracy, " at epoch {}".format(0))

    #List of accuracies for training and testing data:
    accuracies = list()
    accuracies.append(accuracy)
    test_accuracies = list()
    test_accuracies.append(test_accuracy)

    # Begin training
    for i in trange(epochs):
        for train, target in loader:
            # To be modified
            score = net(train.float())
            loss = criterion(score, target.view(batch_size))
            loss.backward()
            optimizer.step()
        #Let's see the accuracy
        ###########################################################################
        if i % 5 == 0 or i == epochs - 1:
            solution = [findMax(net(j[0].float()).tolist()) for j in material]
            accuracy = findAccuracy(solution = solution, start = start, end = end, file = file)
            print("Accuracy is ", accuracy, " at epoch {}".format(i+1))
            test_solution = [findMax(net(j.float()).tolist()) for j in test_material]
            test_accuracy = findAccuracy(solution = test_solution, start = test_start, end = test_end, file = file)
            print("Test accuracy is ", test_accuracy, " at epoch {}".format(i+1))
            accuracies.append(accuracy)
            test_accuracies.append(test_accuracy)


    #save model
    print("Successfully trained/saved model.")
    torch.save(net.state_dict(),"base.pt")
    return accuracies, test_accuracies


def main():
    """Train a FNN_2 neural network with given learning rate, epoch number, and batch size.

    Parameters
    ----------
    n : int
        The number of nodes in the graph.
    learning_rate : float
        The learning rate for neural network.
    epoch : int
        Number of iterations over the whole train set.
    batch_size : int
        Number of train examples to feed the model every time.

    Returns
    -------
    None
    Saves the model in "FNN_2.pt".
    """
    start = 0
    end = 900
    test_start = 900
    test_end = 1200
    vocabulary, vocabulary_length, representations, labels = representation("train.csv", start = start, end = end)
    # print(len(representations), len(representations[0]))
    # sys.exit()
    representations = torch.tensor(np.array(representations))
    representations = [[representations[i], torch.tensor(np.array([labels[i]-1]))] for i in range(end - start)]

    # with open("training_data.txt", "w") as f:
    #     # f.write(pickle.dumps(representations))
    #     f.write(pickle.dumps(representations))
    torch.save(representations, "representations.txt")

    # For testing
    test_representations = torch.tensor(np.array(representTest(file = "train.csv", vocabulary = vocabulary, start = test_start, end = test_end)))

    # build a dict
    values = {"learning_rate": 0.01, "epochs": 25, "batch_size": 50, "start": 0, "end": 900, "file": "train.csv", "nodes": vocabulary_length, \
        "test_material": test_representations, "test_start":test_start, "test_end":test_end}

    # Train the model
    accuracies, test_accuracies = trainNet(values)
    # line 1 points
    x1 = [0,1,6,11,16,21,25]
    # plotting the line 1 points 
    plt.plot(x1, accuracies, label = "Train Accuracy")
    # line 2 points
    # plotting the line 2 points 
    plt.plot(x1, test_accuracies, label = "Test Accuracy")
    plt.xlabel('No.epochs')
    # Set the y axis label of the current axis.
    plt.ylabel('Accuracy')
    # Set a title of the current axes.
    plt.title('Train/Test Accuracy vs epochs')
    # show a legend on the plot
    plt.legend()
    # Display a figure.
    plt.savefig("base.png")
    plt.show()


if __name__ == "__main__":
    main()