# find the optimal learning rate, number of hidden nodes, and number of epochs on the holdout set

import ANN
import numpy as np
import random

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
                        w1,w2 = ANN.ANN_train(train_subset,labels_subset,h,r,ep)
                        # test on holdout set
                        predictions = ANN.ANN_test(holdout_set,h,w1,w2)
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
