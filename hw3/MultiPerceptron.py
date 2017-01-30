# Larissa Clopton
# CMSC 25400

# multi-class classification
# a batch implementation of a multi-class perceptron algorithm
# to classify images as a 0, 1, 2, 3, or 4
# each row of the data represents a 28x28 image of white (0) or black (1) pixels

import numpy as np
import random
import matplotlib.pyplot as plt

# retrieve the training data and respective labels
with open('train01234.digits') as input:
	train_values = [list(map(int, line.split())) for line in input]
labels = open('train01234.labels','r')
train_labels = map(int, labels.readlines())

# bind the data into tuples, shuffle, and then unbind
pairs = [(train_values[i],train_labels[i]) for i in range(len(train_labels))]
random.shuffle(pairs)
train_values = np.array([x[0] for x in pairs])
train_labels = [x[1] for x in pairs]

# retrieve the testing data
with open('test01234.digits') as input:
	test_values = np.array([list(map(int, line.split())) for line in input])

def MultiPerceptron(train,labels,test):

	# duplicate the data 5 times
        train_data = np.concatenate((train,train,train,train,train),axis=0)
	train_labels = labels + labels + labels + labels + labels
	test_data = test

        pixels = train_data.shape[1] # number of pixels in each image
        train_trials = train_data.shape[0] # number of examples
        mistakes = [0]*(train_trials+1) # mistakes as function of number of examples
	t = 0 # initial example number
        cum_error = 0 # cumulative "mistakes"

	# initial weight vectors for classes 0,1,2,3,4
	w = np.zeros((5,pixels))
	
	while (t < train_trials):
		# calculate dot product with each unique weight vector
		products = [np.dot(w[i,:],train_data[t,:]) for i in range(w.shape[0])]
		# predict class with the highest dot product
		predicted = products.index(max(products))
		actual = train_labels[t]
		
		if (predicted != actual):
			# subtract feature vector from weight vector of predicted class
			w[predicted,:] = w[predicted,:] - train_data[t,:]		
			# add feature vector to weight vector of correct class
		 	w[actual,:] = w[actual,:] + train_data[t,:]
			# record a "mistake"
			cum_error += 1

		mistakes[t+1] = cum_error/float(t+1)
		t += 1

	# plot the cumulative number of "mistakes" as a function of number of examples seen
        plt.figure()
        plt.plot(mistakes)
        plt.xlabel('Number of Examples')
        plt.ylabel('Cumulative Mistakes / Examples')
        plt.title('Proportion of Cumulative Mistakes vs. Number of Examples')
	plt.savefig('CumMistakesFuncEC.png')

	# use the final result to predict test data labels
	test_trials = test_data.shape[0]
	predictions = [0]*test_trials

	for i in range(test_trials):
		# calculate dot product with each unique weight vector
                products = [np.dot(w[j,:],test_data[i,:]) for j in range(w.shape[0])]
                # predict class with the highest dot product
                predictions[i] = products.index(max(products))

	# output predictions to a file, one prediction per line
        np.savetxt('test01234.predictions',predictions,fmt='%i')


MultiPerceptron(train_values,train_labels,test_values)
