import numpy as np

# load actual labels
with open('TestDigitY.csv') as input:
        test_labels = np.array([list(map(float, line.split(","))) for line in input])
test_labels = test_labels.astype(int)

# load predicted labels
with open('TestDigitYClopton.csv') as input:
        predict = np.array([list(map(float, line.split(","))) for line in input])
predict = predict.astype(int)

err = (1/float(test_labels.shape[0]))*np.sum([predict[a,0] != test_labels[a,0] for a in range(test_labels.shape[0])])
print(err)
