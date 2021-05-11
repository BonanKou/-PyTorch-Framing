import numpy as np
import pandas as pd
import re
import csv
from datetime import datetime

classCount = 17
stepSize = 3
epochs = 100
l2 = 0.1

def modelInitialize(wordCount):
	mu, sigma = 0, 0.1
	# Whether to +1?
	model = np.random.normal(mu, sigma, [wordCount, classCount])
	return model

# Takes in an array and returns the softmaxed result
def softmax(x):
	#x = [i - max(x) for i in x]
	return np.exp(x) / np.exp(x).sum()
####################
#def SSM(x):
	#raise_exception("NOT IMPLEMENTED YET.")


def oneHotIt(x):
	index = x.indexOf(x.max())
	result = [0 for i in len(x)]
	result[index] = 1
	return result

# Calculate the predicted value for a given vector
def forward(x, model):
	#########################################
	################BIAS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	# The one added at the end is the bias
	#x = np.hstack([x, [1]])
	z = np.dot(x, model)
	return softmax(z)

def forwardNaked(x, model):
	z = np.dot(x, model)
	z = [i - max(z) for i in z]
	return [np.exp(i) for i in z]

# Calculate the cross entrophy loss
def loss(y, truth):
	logY = np.log(y)
	return -np.sum(logY * truth)

# Calcualte the gradient of the function
# 17 gradient vectors to be added for one observation.
def getGradient(x, truth, model):
	predictY = forward(x, model)
	gradient = list()
	for i in range(len(truth)):
		truthTemp = truth[i]
		predictTemp = predictY[i]
		gradient.append([x[j] * (predictTemp - truthTemp) for j in range(len(x))])
	return np.array(gradient)

def generateTruth(frame):
	result = [0 for i in range(17)]
	result[frame - 1] = 1
	return result

def representation(file, start, end):
	df = pd.read_csv(file)
	twitters = df.text[start:end]
	vocabulary = list()
	representations = list()
	for twitter in twitters:
		toRemove = "`~!@#$%^*\(\)_+\}\{<>?:\",./;'\[\]\\-=\'1234567890"
		for i in toRemove:
			if i in twitter:
				twitter = twitter.replace(i, " ")
		content = twitter.split(" ")
		content = [x for x in content if x]
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
	return vocabulary, len(representations[0]), representations

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

def findMax(result):
	index = 0
	for i in range(len(result)):
		if result[i] > result[index]:
			index = i
	return index + 1

def findAccuracy(solution, start, end, file):
	print("solution length", len(solution))
	df = pd.read_csv(file)
	final = 0
	for i in range(len(solution)):
		if solution[i] == df.label[i + start]:
			final += 1
	return final / len(solution)

def doLR(start, end, stepSize, epochs, l2, testStart, testEnd, testFile):
	#Bag of Words representations of all twitters.
	vocabulary, wordCount, representations = representation("train.csv", start, end)
	representationsTest = representTest(vocabulary, testFile, testStart, testEnd)
	trainLen = end - start

    ##########################################################################################################
    ####Use this line to train from a fresh start.
	#model = modelInitialize(wordCount)
    ##########################################################################################################
    ####Comment out this line to train from a fresh start
	model = np.loadtxt("model.txt")
	#np.savetxt('model.txt', model)
	df = pd.read_csv("train.csv")
	test = 0
	truthList = [generateTruth(i) for i in range(1, 18)]
	solution = list()
	for a in range(epochs):
		#Training
		#for each, update model.
		#gradientSum = np.array([[0.0 for i in range(wordCount)] for i in range(classCount)])
		print(a+1, " out of ", epochs, " iteration begins")
		newModel = [[0 for i in range(len(model[0]))] for j in range(len(model))]
		for i in range(len(model)):
			for j in range(len(model[i])):
				newModel[i][j] = model[i][j]
		for i in range(len(representations)):
			truth = truthList[df.label[i] - 1]
			gradient = getGradient(representations[i], truth, model)
			for j in range(len(model)):
				for k in range(len(model[j])):
					# if j != (len(model) - 2):
					# 	# Except for the bias term, do l2.
					# 	newModel[j][k] -= stepSize * (gradient[k][j] + l2 * model[j][k]) / trainLen
					# else:
					newModel[j][k] -= stepSize * (gradient[k][j]) / trainLen
		model = newModel
		print(a+1, " out of ", epochs, " iteration complete")
	solution = list()
	trainsolution = list()
	for represent in representationsTest:
		result = forward(represent, model)
		solution.append(findMax(result))
	lossval = list()
	for represent in representations:
	 	result = forward(represent, model)
	 	trainsolution.append(findMax(result))
	 	lossval.append(loss(result, generateTruth(df.label[representations.index(represent)])))
	print("total loss is", sum(lossval))
	trainAccu = findAccuracy(trainsolution, start, end, "train.csv")
	print("solution for train", trainAccu)
	logString = "TotalLose: {}, TrainAccracy:{}\n".format(sum(lossval), trainAccu)
	logFile = open('log.txt', 'a')
	logFile.write(logString)
	iterationString = "{} iterations completed\n".format(a)
	logFile.write(iterationString)
	logFile.close()
	np.savetxt('model.txt', model)
	return solution

def SaveFile(solution, output_csv_file):
	df = pd.read_csv("test.csv")
	with open(output_csv_file, mode='w') as out_csv:
		writer = csv.writer(out_csv, delimiter=',', quoting=csv.QUOTE_NONE)
		writer.writerow(["tweet_id", "issue", "text", "author", "label"])
		for i in range(len(solution)):
			#writer.writerow([df.tweet_id[i], df.issue[i], df.text[i], df.author[i], solution[i]])
			writer.writerow([df.tweet_id[i], df.issue[i], "text", df.author[i], solution[i]])

if __name__ == '__main__':
	solution = doLR(0, 1231, 3, 5, 0.1, 960, 1231, "test.csv")
    ##################################################################
    ########Code for some 5-fold validation###########################
    # solution = doLR(0, 960, 3, 1, 0.1, 960, 1231, "train.csv")
    # solution = doLR(0, 960, 3, 10, 0.1, 960, 1231, "train.csv")
    # solution = doLR(0, 960, 3, 100, 0.1, 960, 1231, "train.csv")
    # solution = doLR(0, 960, 3, 50, 0.1, 960, 1231, "train.csv")
    # solution = doLR(0, 960, 3, 30, 0.1, 960, 1231, "train.csv")
    # solution = doLR(0, 960, 3, 20, 0.1, 960, 1231, "train.csv")
    # solution = doLR(0, 960, 3, 15, 0.1, 960, 1231, "train.csv")
    # solution = doLR(0, 960, 3, 86, 0.1, 960, 1231, "train.csv")
	SaveFile(solution, "testlr.csv")


