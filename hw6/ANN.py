# Larissa Clopton
# CMSC 25400

# an implementation of a 3-layer, fully connected, feedforward neural network
# with a  single hidden layer

import numpy as np
import random
import matplotlib.pyplot as plt

def ANN_train(data,labels,hidden,rate,epochs):
	# hidden - number of hidden units
	# rate - learning rate
	# epochs - number of times to feed in the data

	# define vectors to hold outputs at each layer
	# output of last in x0 and x1 always 1 (bias neurons)
	x0 = np.zeros(784 + 1)
	x1 = np.zeros(hidden + 1)
	x2 = np.zeros(10)

	# define weight matrices (fully connected ANN)
	# initialize with random weights
	w1 =  np.random.uniform(-1,1,(hidden, 784 + 1))
	w2 = np.random.uniform(-1,1,(10, hidden + 1))

	# define the standard vectors for each class
	E = np.identity(10)

	for ep in range(epochs):
		for i in range(data.shape[0]):
			# set the input layer as the value of the pixels
			x0[:784] = data[i,:]
			x0[784] = 1 # bias neuron
			
			# compute activations and outputs of the hidden layer
			for j in range(hidden):
				# activation
				act = np.dot(x0,w1[j,:]) 
				# output
				x1[j] = 1/(1 + np.exp(-1*act))
			x1[hidden] = 1 # bias neuron	

			# compute activations and outputs of the final layer
			for k in range(10):
				# activation
				act = np.dot(x1,w2[k,:])
				# output
				x2[k] = 1/(1 + np.exp(-1*act))

			# back-propagate to update the weights
			actual = E[labels[i,0],:]
			# compute deltas for the final layer
			delta2 = np.zeros(10)
			for k in range(10):
				delta2[k] = (x2[k]-actual[k])*x2[k]*(1-x2[k]) 				
			# compute deltas for the hidden layer
			delta1 = np.zeros(hidden)
			for j in range(hidden):
				delta1[j] = x1[j]*(1-x1[j])*np.dot(np.transpose(w2[:,j]),delta2)	
			
			# update w2 weights
			for k in range(10):
				w2[k,:] -= rate*delta2[k]*x1
			# update w1 weights
			for j in range(hidden):
				w1[j,:] -= rate*delta1[j]*x0

	return (w1, w2)


def ANN_test(data,hidden,w1,w2):
	# hidden - number of hidden nodes
	# w1 - weight matrix, x0 to x1
	# w2 - weight matrix, x1 to x2

	# define vectors to hold outputs at each layer
        # output of last in x0 and x1 always 1 (bias neurons)
        x0 = np.zeros(784 + 1)
        x1 = np.zeros(hidden + 1)
        x2 = np.zeros(10)

	# store the predictions of each testing example
	predict = [0]*data.shape[0]

	for i in range(data.shape[0]):
		# set the input layer as the value of the pixels
                x0[:784] = data[i,:]
                x0[784] = 1 # bias neuron

		# compute activations and outputs of the hidden layer
                for j in range(hidden):
                	# activation
                        act = np.dot(x0,w1[j,:])
                        # output
                        x1[j] = 1/(1 + np.exp(-1*act))
                x1[hidden] = 1 # bias neuron
                
		# compute activations and outputs of the final layer
                for k in range(10):
                	# activation
                        act = np.dot(x1,w2[k,:])
                        # output
                        x2[k] = 1/(1 + np.exp(-1*act))	

		# predict the class as the index of maximum output in the final layer
		predict[i] = np.argmax(x2)		
		
	return np.transpose(np.asarray(predict))


""" NOTE - FILES HAVE ALREADY BEEN DECOMPRESSED """
""" NOTE - THE CODE BELOW HAS BEEN MOVED TO OTHER FILES """

"""# STEP 1 - optimize parameters for a holdout set

# load the training data and respective labels
with open('TrainDigitX.csv') as input:
        train_values = np.array([list(map(float, line.split(","))) for line in input])
with open('TrainDigitY.csv') as input:
	train_labels = np.array([list(map(float, line.split(","))) for line in input])
train_labels = train_labels.astype(int)

# pick a holdout set (10-20%)
holdout_size = int(0.1*train_values.shape[0])
holdout_idxs = random.sample(range(train_values.shape[0]),holdout_size)
subset_idxs = list(set(range(train_values.shape[0])) - set(holdout_idxs))
train_subset = np.take(train_values,subset_idxs,axis=0)
labels_subset = np.take(train_labels,subset_idxs,axis=0)
holdout_set = np.take(train_values,holdout_idxs,axis=0)
holdout_labels = np.take(train_labels,holdout_idxs,axis=0)

# define ranges for the three parameters
hidden = [32, 64, 128, 256]
rate = [0.15, 0.2, 0.25, 0.35]
epoch = [4, 6, 8]

# compute errors for various combinations of hidden nodes, learning rates, and epochs
err = np.zeros((len(hidden),len(rate),len(epoch)))
for i,h in enumerate(hidden):
	print(h)
	for j,r in enumerate(rate):
		print(r)
		for k,ep in enumerate(epoch):
			print(ep)
			# train on training subset
			w1,w2 = ANN_train(train_subset,labels_subset,h,r,ep)
			# test on holdout set
			predictions = ANN_test(holdout_set,h,w1,w2)
			err[i,j,k] = (1/float(holdout_size))*np.sum([predictions[a] != holdout_labels[a,0] for a in range(holdout_size)]) 

# find combination of parameters with lowest error
min_val = 1
min_hidden = 0
min_rate = 0
min_epoch = 0
for i in range(len(hidden)):
	val = np.amin(err[i,:,:])
	if val < min_val:
		min_val = val
		print(min_val)
		(j,k) = np.unravel_index(err[i,:,:].argmin(),err[i,:,:].shape)
		min_hidden = i
		min_rate = j
		min_epoch = k
print(hidden[min_hidden])
print(rate[min_rate])
print(epoch[min_epoch])

# STEP 2 - vary parameters one at a time on test set, keepting others optimal

# load the testing data and respective labels
with open('TestDigitX.csv') as input:
        test_values = np.array([list(map(float, line.split(","))) for line in input])
with open('TestDigitY.csv') as input:
        test_labels = np.array([list(map(float, line.split(","))) for line in input])
test_labels = test_labels.astype(int)

# define the optimal parameters (holdout set - 3.22% error)
h_opt = 128
r_opt = 0.35
ep_opt = 8

# define the manipulation paramters
hidden = [32, 64, 128, 256]
rate = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
epoch = [2, 4, 6, 8, 10]

# compute error rates for each parameter manipulation
h_err = [0]*len(hidden)
r_err = [0]*len(rate)
ep_err = [0]*len(epoch)
for i,h in enumerate(hidden):
	print(h)
	# train on training set
        w1,w2 = ANN_train(train_values,train_labels,h,r_opt,ep_opt)
        # test on testing set
        predictions = ANN_test(test_values,h,w1,w2)
        h_err[i] = (1/float(test_values.shape[0]))*np.sum([predictions[a] != test_labels[a,0] for a in range(test_values.shape[0])])
for j,r in enumerate(rate):
	print(r)
	# train on training set
        w1,w2 = ANN_train(train_values,train_labels,h_opt,r,ep_opt)
        # test on testing set
        predictions = ANN_test(test_values,h_opt,w1,w2)
        r_err[j] = (1/float(test_values.shape[0]))*np.sum([predictions[a] != test_labels[a,0] for a in range(test_values.shape[0])])
for k,ep in enumerate(epoch):
        print(ep)
	# train on training set
        w1,w2 = ANN_train(train_values,train_labels,h_opt,r_opt,ep)
        # test on testing set
        predictions = ANN_test(test_values,h_opt,w1,w2)
        ep_err[k] = (1/float(test_values.shape[0]))*np.sum([predictions[a] != test_labels[a,0] for a in range(test_values.shape[0])])
print(h_err)
print(r_err)
print(ep_err)

# plot the error rates for each parameter manipulation
plt.figure()
plt.plot(hidden,h_err)
plt.xlabel('Hidden nodes')
plt.ylabel('Error rate')
plt.title('Error rate vs hidden nodes')
plt.savefig('HiddenError.png')
plt.figure()
plt.plot(rate,r_err)
plt.xlabel('Learning rate')
plt.ylabel('Error rate')
plt.title('Error rate vs learning rate')
plt.savefig('RateError.png')
plt.figure()
plt.plot(epoch,ep_err)
plt.xlabel('Epochs')
plt.ylabel('Error rate')
plt.title('Error rate vs epochs')
plt.savefig('EpochError.png')

# STEP 3 - run with optimal parameters on two test sets, save predictions

# TestDigitX
w1,w2 = ANN_train(train_values,train_labels,h_opt,r_opt,ep_opt)
predictions = ANN_test(test_values,h_opt,w1,w2)
np.savetxt('TestDigitYClopton.csv.gz',predictions,fmt='%i')
# TestDigitX2
with open('TestDigitX2.csv') as input:
        test2_values = np.array([list(map(float, line.split(","))) for line in input])
predictions = ANN_test(test2_values,h_opt,w1,w2)
np.savetxt('TestDigitY2Clopton.csv.gz',predictions,fmt='%i')"""
