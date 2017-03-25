# save predictions on the two testing sets

import ANN
import numpy as np

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
h_opt = 128
r_opt = 0.35
ep_opt = 8

# TestDigitX
w1,w2 = ANN.ANN_train(train_values,train_labels,h_opt,r_opt,ep_opt)
predictions = ANN.ANN_test(test_values,h_opt,w1,w2)
np.savetxt('TestDigitYClopton.csv.gz',predictions,fmt='%i')
# TestDigitX2
with open('TestDigitX2.csv') as input:
        test2_values = np.array([list(map(float, line.split(","))) for line in input])
predictions = ANN.ANN_test(test2_values,h_opt,w1,w2)
np.savetxt('TestDigitY2Clopton.csv.gz',predictions,fmt='%i')
