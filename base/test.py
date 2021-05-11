# def SaveFile(solution, output_csv_file):
# 	df = pd.read_csv("test.csv")
# 	content = df.text
# 	with open(output_csv_file, mode='w') as out_csv:
# 		writer = csv.writer(out_csv, delimiter=',', quoting=csv.QUOTE_NONE)
# 		writer.writerow(["tweet_id", "issue", "text", "author", "label"])
# 		for i in range(len(solution)):
# 			print(i, "is ok")
# 			#writer.writerow([df.tweet_id[i], df.issue[i], df.text[i].encode('utf-8'), df.author[i], solution[i]])
# 			writer.writerow([df.tweet_id[i], df.issue[i], content[i].encode('utf-8').replace(',', '\,').bytes(), df.author[i], solution[i]])

# if __name__ == '__main__':
# 	df = pd.read_csv("test.csv")
# 	SaveFile([0 for i in range(820)], 'lala3.csv')

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
from torch.utils.data import DataLoader, Sampler
from torchvision import datasets,transforms


####################################################################
# with open('data.json', 'r') as fi:
#     [tweet_ids, tweet_id2topic]=json.load(fi)

# variables = list()
# for i in tweet_id2topic:
#     if tweet_id2topic[i] not in variables:
#         variables.append(tweet_id2topic[i])

# print(variables)
#####################################################################

import csv
from random import randrange


def ReadFile(input_csv_file):
    # Reads input_csv_file and returns four dictionaries tweet_id2text, tweet_id2issue, tweet_id2author_label, tweet_id2label

    tweet_id2text = {}
    tweet_id2issue = {}
    tweet_id2author_label = {}
    tweet_id2label = {}
    f = open(input_csv_file, "r")
    csv_reader = csv.reader(f)
    row_count=-1
    for row in csv_reader:
        row_count+=1
        if row_count==0:
            continue

        tweet_id = int(row[0])
        issue = str(row[1])
        text = str(row[2])
        author_label = str(row[3])
        label = row[4]
        tweet_id2text[tweet_id] = text
        tweet_id2issue[tweet_id] = issue
        tweet_id2author_label[tweet_id] = author_label
        tweet_id2label[tweet_id] = label

    #print("Read", row_count, "data points...")
    return tweet_id2text, tweet_id2issue, tweet_id2author_label, tweet_id2label


def SaveFile(tweet_id2text, tweet_id2issue, tweet_id2author_label, tweet_id2label, output_csv_file):

    with open(output_csv_file, mode='w') as out_csv:
        writer = csv.writer(out_csv, delimiter=',', quoting=csv.QUOTE_NONE)
        writer.writerow(["tweet_id", "issue", "text", "author", "label"])
        for tweet_id in tweet_id2text:
            writer.writerow([tweet_id, tweet_id2issue[tweet_id], tweet_id2text[tweet_id], tweet_id2author_label[tweet_id], tweet_id2label[tweet_id]])

def LR():

    # Read training data
    train_tweet_id2text, train_tweet_id2issue, train_tweet_id2author_label, train_tweet_id2label = ReadFile('train.csv')
    print(train_tweet_id2text)
    sys.exit()
    '''
    Implement your Logistic Regression classifier here
    '''

    # Read test data
    test_tweet_id2text, test_tweet_id2issue, test_tweet_id2author_label, test_tweet_id2label = ReadFile('test.csv')

    # Predict test data by learned model

    '''
    Replace the following random predictor by your prediction function.
    '''

    for tweet_id in test_tweet_id2text:
        # Get the text
        text=test_tweet_id2text[tweet_id]

        # Predict the label
        label=randrange(1, 18)

        # Store it in the dictionary
        test_tweet_id2label[tweet_id]=label

    # # Save predicted labels in 'test_lr.csv'
    # SaveFile(test_tweet_id2text, test_tweet_id2issue, test_tweet_id2author_label, test_tweet_id2label, 'test_lr.csv')
if __name__ == '__main__':
    df = pd.read_csv("train.csv")
    print(df)