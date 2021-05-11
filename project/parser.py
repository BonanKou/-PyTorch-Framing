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
import os


# Target
targets = dict()
found = 0
data_frame = pd.read_csv("unlabeled.csv")

#TODO: set this directory to whatever directory of the data you have locally
data_dir = "data/"
out_file = "./unlabeled.csv"
#TODO: wget https://raw.githubusercontent.com/alexlitel/congresstweets-automator/master/data/users.json
#TODO: wget https://raw.githubusercontent.com/alexlitel/congresstweets-automator/master/data/historical-users.json
#TODO: maybe find a better way to process commas and quotes

#setup users2author_label
with open('users.json', 'r') as fi:
    entities=json.load(fi)
#user2author_label
id2party = {}

#party_set
party = set()


#parse users:
for e in entities:
#parse party for author_label
    if 'party' in e:
        p = e['party']
    else:
        p = 'None'

    #expand party set
    if p not in party:
        party.add(p)

    #append user2author_label
    if 'accounts' in e:
        for a in e['accounts']:
            id2party[a['id']] = p

#setup users2author_label
with open('./historical-users.json', 'r') as fi:
    entities=json.load(fi)

#parse users:
for e in entities:
#parse party for author_label
    if 'party' in e:
        p = e['party']
    else:
        p = 'None'

    #expand party set
    if p not in party:
        party.add(p)

    #append user2author_label
    if 'accounts' in e:
        for a in e['accounts']:
            id2party[a['id']] = p

#party2author_label
#calculated after run 1 time
party2author_label = {
'D' : "democrat",
'R' : "republican",
'I' : "independent",
'L': "libertarian",
'None': None,
'N/A': None,
}

with open('data.json', 'r') as fi:
    [tweet_ids, tweet_id2topic]=json.load(fi)

# get text
files = os.listdir(data_dir)

#final tweet output_file
out = open(out_file, 'w+', encoding = "utf8")

#header:
out.write("tweet_id,issue,text,author,label,time\n")
label = None # no labels currently :(

#setup read_files to append
read_files = open("read_files", "a+")

#go through all other files
tweets = 0
for path in files:
    #open and save status
    fname = data_dir + path
#TODO: if you think its taking too long, see progress
    #print("Processing {}".format(fname))
    with open(fname, 'r', encoding="utf8") as f:
        data = json.load(f)
######################REVISIT
    for tweet in data:
        if 'id' in tweet:
            if tweet['id'] in tweet_id2topic:
                tweets += 1
                string = "{},{}"
                #convert to author_label
                u_id = tweet['user_id']
                p = id2party[u_id]
                author_label = party2author_label[p]

                #other stuff
                t_id = tweet['id']
                issue = tweet_id2topic[t_id]
                text = tweet["text"]
                time = tweet["time"]
                if tweets % 10 == 0:
                    print(time)

                toRemove = "`~!@#$%^*\(\)_+\}\{<>?:\",./;'\[\]\\-=\'1234567890\n"
                for i in toRemove:
                    if i in text:
                        text = text.replace(i, " ")
                # content = text.split(" ")
                # content = [x for x in content if x]

                #sanity check, remove all newlines and commas
                #TODO: add this in, its pretty easy :) 

                #add line to csv
                string = "{},{},{},{},{},{}\n".format(t_id, issue, text, author_label, label, time)

                #write to output csv
                #out.write(string.encode('cp850','replace').decode('cp850'))
                out.write(string)
                #save status:
                read_files.write("{}\n".format(fname))

        if "id" in tweet and tweet["id"] in list(data_frame.tweet_id):
            print("I found one!")
            found += 1
            targets[tweet["id"]] = tweet["time"]

print(found)
with open("target_time.txt", "wb") as file:
    file.write(pickle.dumps(targets))
#sanity check:
print("NUMBERS SHOULD MATCH:")
print(tweets)
print(len(tweet_id2topic))
