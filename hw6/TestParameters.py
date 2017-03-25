# manipulate each parameter individually, producing error plots for each

import ANN
import numpy as np
import matplotlib.pyplot as plt

# load the training data and respective labels
with open('TrainDigitX.csv') as input:
        train_values = np.array([list(map(float, line.split(","))) for line in input])
with open('TrainDigitY.csv') as input:
        train_labels = np.array([list(map(float, line.split(","))) for line in input])
train_labels = train_labels.astype(int)

# load the testing data and respective labels
with open('TestDigitX.csv') as input:
        test_values = np.array([list(map(float, line.split(","))) for line in input])
with open('TestDigitY.csv') as input:
        test_labels = np.array([list(map(float, line.split(","))) for line in input])
test_labels = test_labels.astype(int)

# define the optimal parameters (holdout set - 3.22% error)
h_opt = 256
r_opt = 0.35
ep_opt = 8

# define the manipulation paramters
hidden = [32, 64, 128, 256]
rate = [0.2, 0.3, 0.4, 0.5]
epoch = [2, 4, 6, 8, 10]

# compute error rates for each parameter manipulation
h_err = [0]*len(hidden)
r_err = [0]*len(rate)
ep_err = [0]*len(epoch)
for i,h in enumerate(hidden):
        print(h)
        # train on training set
        w1,w2 = ANN.ANN_train(train_values,train_labels,h,r_opt,ep_opt)
        # test on testing set
        predictions = ANN.ANN_test(test_values,h,w1,w2)
        h_err[i] = (1/float(test_values.shape[0]))*np.sum([predictions[a] != test_labels[a,0] for a in range(test_values.shape[0])])
for j,r in enumerate(rate):
        print(r)
        # train on training set
        w1,w2 = ANN.ANN_train(train_values,train_labels,h_opt,r,ep_opt)
        # test on testing set
        predictions = ANN.ANN_test(test_values,h_opt,w1,w2)
        r_err[j] = (1/float(test_values.shape[0]))*np.sum([predictions[a] != test_labels[a,0] for a in range(test_values.shape[0])])
for k,ep in enumerate(epoch):
        print(ep)
        # train on training set
        w1,w2 = ANN.ANN_train(train_values,train_labels,h_opt,r_opt,ep)
        # test on testing set
        predictions = ANN.ANN_test(test_values,h_opt,w1,w2)
        ep_err[k] = (1/float(test_values.shape[0]))*np.sum([predictions[a] != test_labels[a,0] for a in range(test_values.shape[0])])

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
