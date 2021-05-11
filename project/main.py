import json
import matplotlib.pyplot as plt
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
torch.manual_seed(1)
import re
from torch.utils.data import DataLoader, Sampler
from torchvision import datasets,transforms
from nltk.stem import SnowballStemmer

#####################################################################
# with open('data.json', 'r') as fi:
#     [tweet_ids, tweet_id2topic]=json.load(fi)
#print(len(tweet_ids))
#####################################################################

def softmax(x):
    #x = [i - max(x) for i in x]
    return (np.exp(x) / np.exp(x).sum()).tolist()
def percentage(x):
    return (x / x.sum()).tolist()

# Here I employ bag of words representations.
def representTest(vocabulary, file, start, end):
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

# Representing...
# def representation(file, start, end):
#     df = pd.read_csv(file)
#     twitters = df.text[start:end]
#     vocabulary = list()
#     representations = list()
#     for twitter in twitters:
#         toRemove = "`~!@#$%^*\(\)_+\}\{<>?:\",./;'\[\]\\-=\'1234567890"
#         for i in toRemove:
#             if i in twitter:
#                 twitter = twitter.replace(i, " ")
#         content = twitter.split(" ")
#         content = [x for x in content if x]
#         for word in content:
#             if not (word in vocabulary):
#                 vocabulary.append(word)
#         represent = list([0 for i in range(len(vocabulary))])
#         for word in content:
#             represent[vocabulary.index(word)] += 1
#         representations.append(represent)

#     print("Length of vocab ", len(vocabulary))
#     # Fills in the gap and add addition tag that describes the political affinity of the author
#     for i in range(len(representations)):
#         represent = representations[i]
#         # Fill the gap between represent length and the vocab length
#         gap = len(vocabulary) - len(represent)
#         represent.extend([0 for i in range(gap)])
#         #For the bias
#         represent.extend([1])
#         author = df.author[i]
#         if author == "democrat":
#             represent.append(1)
#         else:
#             represent.append(0)
#     return vocabulary, len(representations[0]), representations, df.label[start:end]

# Checks accuracy
def findAccuracy(solution, start, end, file):
    print("solution length", len(solution))
    df = pd.read_csv(file)
    final = 0
    for i in range(len(solution)):
        if solution[i] == df.label[i + start]:
            final += 1
    return final / len(solution)

# Find max value in a list. It returns an index+1 because the labels ranges from 1-17 while
# indexes range from 0-16.
def findMax(result):
    index = 0
    for i in range(len(result)):
        if result[i] > result[index]:
            index = i
    return index + 1

#Neural Network##################################################
class Net(nn.Module):
    def __init__(self, vocabulary_length):
        super().__init__()
        self.fc1 = nn.Linear(vocabulary_length, 100)
        self.fc2 = nn.Linear(100, 17)
    def forward(self, x):
        x = self.fc2(F.relu(self.fc1(x)))
        return x

def validate(values):
    # Unroll the dictionary
    learning_rate = values["learning_rate"]
    epochs = values["epochs"]
    batch_size = values["batch_size"]
    start = values["start"]
    end = values["end"]
    file = values["file"]
    nodes = values["nodes"]

def onehot(issue):
    onehot = np.zeros(6)
    if issue == "guns":
        onehot[0] = 1 
    elif issue == "isis":
        onehot[4] = 1 
    elif issue == "aca":
        onehot[1] = 1 
    elif issue == "immigration" or issue == "immig":
        onehot[3] = 1 
    elif issue == "abortion" or issue == "abort":
        onehot[5] = 1 
    else:
        onehot[2] = 1
    return onehot, int(np.where(onehot == 1)[0])

def represent(vocabulary, file, start=0, end=30000):
    df = pd.read_csv(file)
    if file == "unlabeled.csv":
        start = 0
        end = len(df.text)

    issues = list()
    representations = list()
    for i in range(start, end):
        representation = np.zeros(len(vocabulary))
        content = df.text[i].split(" ")
        content = [x for x in content if x]
        for j in content:
            if j in vocabulary:
                representation[vocabulary.index(j)] += 1

        if df.author[i] == "democrat":
            representation = np.append(representation, [1])
        else:
            representation = np.append(representation, [0])

        issue_vec, issue = onehot(df.issue[i])
        representation = np.append(representation, issue_vec)
        representation = np.append(representation, [1])
        representations.append(representation)
        issues.append(issue)
    return representations, issues



#################################################################
# Main stuff
# def main():
#     print("Here we go")
#     # df = pd.read_csv("train.csv")
#     # print("solve mystery", df.label)

#     # build representation
#     start = 0
#     end = 900
#     test_start = 900
#     test_end = 1200
#     vocabulary, vocabulary_length, representations, labels = representation("train.csv", start = start, end = end)
#     # print(len(representations), len(representations[0]))
#     # sys.exit()
#     representations = torch.tensor(np.array(representations))
#     representations = [[representations[i], torch.tensor(np.array([labels[i]-1]))] for i in range(end - start)]
#     print("here1")
#     print(representations)
#     print(len(representations))


#     # with open("training_data.txt", "w") as f:
#     #     # f.write(pickle.dumps(representations))
#     #     f.write(pickle.dumps(representations))
#     torch.save(representations, "representations.txt")

#     # For testing
#     test_representations = torch.tensor(np.array(representTest(file = "train.csv", vocabulary = vocabulary, start = test_start, end = test_end)))

#     # build a dict
#     print("here12")
#     values = {"learning_rate": 0.01, "epochs": 25, "batch_size": 50, "start": 0, "end": 900, "file": "train.csv", "nodes": vocabulary_length, \
#         "test_material": test_representations, "test_start":test_start, "test_end":test_end}

#     print("should be..", representations[10])

#     # Train the model
#     trainNet(values)


def build_vocab(file1, start, end):
    vocabulary = list()

    df1 = pd.read_csv(file1)
    texts = [df1.text[i] for i in range(start, end)]
    df2 = pd.read_csv("unlabeled.csv")
    for i in range(len(df2.text)):
        texts.append(df2.text[i])
    sb = SnowballStemmer("english")

    for i in texts:
        content = i.lower().split(" ")
        content = [sb.stem(x) for x in content if x]
        for word in content:
            if not (word in vocabulary):
                vocabulary.append(word)

    return len(vocabulary), vocabulary



# Helper class: Dataset #  recurrent neural network (RNN)
class Dataset(torch.utils.data.Dataset): # define a grapg dataset
    def __init__(self, representations, solutions, file, flag):
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
        self.solutions = solutions
        self.dataLen = len(solutions)
        self.representations = representations
        self.file =file
        self.flag = flag
        self.test = [np.array([4, 7, 23, 55, 40, 0, 2, 32, 10, 0, 4, 46, 20, 0, 1, 13, 8]), \
                    np.array([65, 9, 6, 28, 24, 0, 3, 128, 21, 3, 18, 116, 174, 2, 21, 100, 15]), \
                    np.array([2, 2, 37, 16, 30, 21, 93, 8, 36, 14, 49, 166, 65, 0, 5, 55, 147]), \
                    np.array([16, 7, 6, 6, 42, 3, 15, 0, 29, 19, 7, 81, 52, 1, 1, 32, 2]), \
                    np.array([0, 0, 9, 99, 23, 2, 2, 3, 10, 17, 7, 39, 14, 1, 2, 11, 48]), \
                    np.array([6, 4, 46, 3, 11, 10, 115, 1, 6, 13, 14, 69, 68, 35, 6, 99, 57])]

    def __getitem__(self, index: int):
        # with open("training.txt", "rb") as file:
        #     train = pickle.loads(file.read())
        representation = torch.tensor(self.representations[index])
        if self.file == "train.csv":
            solution = torch.tensor(self.solutions[index])
        else:
            if self.flag == "softmax":
                solution = torch.tensor(softmax(self.test[self.solutions[index]]))
            else:
                solution = torch.tensor(percentage(self.test[self.solutions[index]]))

        return representation, solution

    def __len__(self):
        return self.dataLen



#Neural Network##################################################
# def trainNet_new(net, values):
#     learning_rate = values["learning_rate"]
#     epochs = values["epochs"]
#     batch_size = values["batch_size"]
#     start = values["start"]
#     end = values["end"]
#     file = values["file"]
#     nodes = values["nodes"]
#     test_start = values["test_start"]
#     test_end = values["test_end"]
#     test_material = values["test_material"]
    #Trains the neural network

def pretrainNet(values, representations, issues):
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
    flag = values["flag"]

    net = Net(vocabulary_length = nodes)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate,  weight_decay=0.0001)
    criterion = nn.MSELoss()
    dataset = Dataset(representations = representations, solutions = issues, file="unlabeled.csv", flag = flag)
    loader = DataLoader(dataset, batch_size = 50)
    


    test_solution = [findMax(net(j.float()).tolist()) for j in torch.tensor(test_material)]
    test_accuracy = findAccuracy(solution = test_solution, start = test_start, end = test_end, file = file)
    print("Test accuracy is ", test_accuracy, " at epoch {}".format(0))


    #List of accuracies for training and testing data:
    # accuracies = list()
    # accuracies.append(accuracy)
    test_accuracies = list()
    losses = list()
    test_accuracies.append(test_accuracy)

    # Begin training
    for i in trange(epochs):
        for train, target in loader:
            # To be modified
            score = net(train.float())
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()
        #Let's see the accuracy
        ###########################################################################
        if i % 2 == 0 or i == epochs - 1:
            # solution = [findMax(net(j.float()).tolist()) for j in torch.tensor(representations)]
            # accuracy = findAccuracy(solution = solution, start = start, end = end, file = file)
            # print("Accuracy is ", accuracy, " at epoch {}".format(i+1))
            test_solution = [findMax(net(j.float()).tolist()) for j in torch.tensor(test_material)]
            test_accuracy = findAccuracy(solution = test_solution, start = test_start, end = test_end, file = file)
            print("Test accuracy is ", test_accuracy, " at epoch {}".format(i+1))
            #accuracies.append(accuracy)
            test_accuracies.append(test_accuracy)
            losses.append(loss)

    #save model
    print("Successfully trained/saved PRETRAINING model.")
    torch.save(net.state_dict(),"project.pt")
    return losses, test_accuracies
    #return accuracies, test_accuracies


def trainNet(values, representations, net):
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
    flag = values["flag"]

    #敢叫日月换新天
    #net = Net(vocabulary_length = nodes)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate,  weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()

    
    df = pd.read_csv(file)
    solutions = list(df.label[start : end])
    print(solutions[:30], "old")
    for i in range(len(solutions)):
        solutions[i] -= 1
    print(solutions[:30], "new")

    dataset = Dataset(representations = representations, solutions = solutions, file="train.csv", flag = flag)
    loader = DataLoader(dataset, batch_size = 50)

    # Before training
    solution = [findMax(net(j.float()).tolist()) for j in torch.tensor(representations)]
    accuracy = findAccuracy(solution = solution, start = start, end = end, file = file)
    print("Accuracy is ", accuracy, " at epoch {}".format(0))
    test_solution = [findMax(net(j.float()).tolist()) for j in torch.tensor(test_material)]
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
            solution = [findMax(net(j.float()).tolist()) for j in torch.tensor(representations)]
            accuracy = findAccuracy(solution = solution, start = start, end = end, file = file)
            print("Accuracy is ", accuracy, " at epoch {}".format(i+1))
            test_solution = [findMax(net(j.float()).tolist()) for j in torch.tensor(test_material)]
            test_accuracy = findAccuracy(solution = test_solution, start = test_start, end = test_end, file = file)
            print("Test accuracy is ", test_accuracy, " at epoch {}".format(i+1))
            accuracies.append(accuracy)
            test_accuracies.append(test_accuracy)

    #save model
    print("Successfully trained/saved model.")
    torch.save(net.state_dict(),"project_2.pt")
    return accuracies, test_accuracies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='softmax')
    args = parser.parse_args()
    flag = args.type
    if flag and flag != "softmax" and flag != "percentage":
        print("Please choose between 'softmax' and 'percentage' for --type!")
        sys.exit()


    df = pd.read_csv("train.csv")
    # issues = list()
    # for i in list(df.issue):
    #     if i not in issues:
    #         issues.append(i)
    old_issue = ['guns', 'aca', 'lgbt', 'immig', 'isis', 'abort']
    new_issue = ['aca', 'isis', 'immigration', 'guns', 'lgbtq', 'abortion']

    vocab_len, vocabulary = build_vocab("train.csv", 0, 1000)
    print("new vocab lenght = ", vocab_len)
    with open("vocab.txt", "wb") as f:
        f.write(pickle.dumps(vocabulary))
    with open("vocab.txt", "rb") as f:
        vocabulary = pickle.loads(f.read())

    representations, issues = represent(vocabulary, "train.csv", 0, 1000)
    representations_new, issues_new = represent(vocabulary, "unlabeled.csv")

    test_representations, _ = represent(vocabulary, "train.csv", 1000, 1200)
    values = {"learning_rate": 0.0001, "epochs": 37, "batch_size": 20, "start": 0, "end": 1000, "file": "train.csv", "nodes": vocab_len + 8, \
        "test_material": test_representations, "test_start":1000, "test_end":1200, "flag":flag}
    losses, pretest = pretrainNet(values, representations_new, issues_new)
    print("Pretraining finished!")

    net = Net(vocab_len + 8)
    net.load_state_dict(torch.load("project.pt"))
    values = {"learning_rate": 0.001, "epochs": 46, "batch_size": 50, "start": 0, "end": 1000, "file": "train.csv", "nodes": vocab_len + 8, \
        "test_material": test_representations, "test_start":1000, "test_end":1200, "flag":flag}
    accuracies, test_accuracies = trainNet(values, representations, net)

    x1 = np.array([0, 1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37])
    x2 = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 46])
    x2 = x2 + 37
    x3 = np.concatenate((x1, x2), axis =0)


    # plotting the line 1 points 
    plt.plot(x1, pretest, label = "Test Accuracy for stage 1")
    plt.plot(x2, test_accuracies, label = "Test Accuracy for stage 2")
    plt.plot(x2, accuracies, label = "Train Accuracy for stage 2")
    plt.xlabel('No.epochs')
    # Set the y axis label of the current axis.
    plt.ylabel('Accuracy')
    # Set a title of the current axes.
    plt.title('Accuracy for both stage vs epochs')
    # show a legend on the plot
    plt.legend()
    # Display a figure.
    plt.savefig("final.png")
    plt.show()




if __name__ == "__main__":
    main()




    # print(issues)
    #main()