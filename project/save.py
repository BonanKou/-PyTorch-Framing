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
torch.manual_seed(1)
import re
from main import *
from torch.utils.data import DataLoader, Sampler
from torchvision import datasets,transforms

def softmax(x):
    #x = [i - max(x) for i in x]
    return (np.exp(x) / np.exp(x).sum()).tolist()
def percentage(x):
    return (x / x.sum()).tolist()
def SaveFile(solution, output_csv_file):
    df = pd.read_csv("test.csv")
    with open(output_csv_file, mode='w', encoding='utf-8',newline='') as out_csv:
        writer = csv.writer(out_csv, delimiter=',', quoting=csv.QUOTE_NONE)
        writer.writerow(["tweet_id", "issue", "text", "author", "label"])
        for i in range(len(solution)):
            writer.writerow([df.tweet_id[i], df.issue[i].encode("utf-8").decode("utf-8"), df.text[i], df.author[i], solution[i]])
            #writer.writerow([df.tweet_id[i], df.issue[i], "test", df.author[i], solution[i]])

if __name__ == "__main__":
    # test = [np.array([4, 7, 23, 55, 40, 0, 2, 32, 10, 0, 4, 46, 20, 0, 1, 13, 8]), \
    #         np.array([65, 9, 6, 28, 24, 0, 3, 128, 21, 3, 18, 116, 174, 2, 21, 100, 15]), \
    #         np.array([2, 2, 37, 16, 30, 21, 93, 8, 36, 14, 49, 166, 65, 0, 5, 55, 147]), \
    #         np.array([16, 7, 6, 6, 42, 3, 15, 0, 29, 19, 7, 81, 52, 1, 1, 32, 2]), \
    #         np.array([0, 0, 9, 99, 23, 2, 2, 3, 10, 17, 7, 39, 14, 1, 2, 11, 48]), \
    #         np.array([6, 4, 46, 3, 11, 10, 115, 1, 6, 13, 14, 69, 68, 35, 6, 99, 57])]
    # for i in test:
    #     print(softmax(i))
    # for i in test:
    #     print(percentage(i))

    # print(pd.read_csv("train.csv"))


    net = Net(8120)
    net.load_state_dict(torch.load("project_2.pt"))
    with open("vocab.txt", "rb") as f:
        vocabulary = pickle.loads(f.read())
    representations, _ = represent(vocabulary, "test.csv", 0, 820)
    solutions = [findMax(net(torch.tensor(representations[i]).float())) for i in range(820)]
    SaveFile(solutions, "test_proj.csv")


 