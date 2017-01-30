# Larissa Clopton
# CMSC 25400

# binary classification
# a batch implementation of the perceptron algorithm to classify each image as a 3 or 5
# each row of the data represents a 28x28 image of white (0) or black (1) pixels

import numpy as np
import matplotlib.pyplot as plt

# retrieve the training data and respective labels
with open('train35.digits') as input:
	train_values = np.array([list(map(int, line.split())) for line in input])
labels = open('train35.labels','r')
train_labels = map(int, labels.readlines())

# retrieve the testing data
with open('test35.digits') as input:
	test_values = np.array([list(map(int, line.split())) for line in input])


def Perceptron(train,labels,test):

	# duplicate the data 5 times
	train_data = np.concatenate((train,train,train,train,train),axis=0)
	train_labels = labels + labels + labels + labels + labels
	test_data = test

	pixels = train_data.shape[1] # number of pixels in each image
	train_trials = train_data.shape[0] # number of examples
	w = np.zeros((1,pixels)) # the initial hyperplane
	mistakes = [0]*(train_trials+1) # mistakes as function of number of examples seen
	t = 0 # initial example number
	cum_error = 0 # cumulative "mistakes"

	while (t < train_trials):
		if (np.dot(w,train_data[t,:]) >= 0):
			prediction = 1
			if (prediction != train_labels[t]):
				w = w - train_data[t,:]
				cum_error += 1 
		else:
			prediction = -1
			if (prediction != train_labels[t]):
				w = w + train_data[t,:]
				cum_error += 1
		mistakes[t+1] = cum_error/float(t+1)
		t += 1

	# plot the cumulative number of "mistakes" as a function of number of examples seen
	plt.figure()
	plt.plot(mistakes)
	plt.xlabel('Number of Examples')
	plt.ylabel('Cumulative Mistakes / Examples')
	plt.title('Proportion of Cumulative Mistakes vs. Number of Examples')
	plt.savefig('CumMistakesFunc.png')

	# use final result to predict test data labels
	test_trials = test_data.shape[0]
	predictions = [0]*test_trials
	
	for i in range(test_trials):
		if (np.dot(w,test_data[i,:]) >= 0):
			predictions[i] = 1
		else:
			predictions[i] = -1	
	
	# output predictions to a file, one prediction per line
	np.savetxt('test35.predictions',predictions,fmt='%i')	


Perceptron(train_values,train_labels,test_values)
