# Larissa Clopton
# CMSC 25400

# an implementation of the Viola Jones face detection cascade

import numpy as np
from PIL import Image, ImageDraw
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import datetime

# define global variables

# width and height of the image
width = 64
height = 64

# store the iimage representation for all images
nimages = 500 # the number of images, CHANGE FOR DEBUGGING
iimages = np.zeros((nimages*2,width*height)) # first half holds face, second half holds background

# store definitions of the rectangles
nfeatures = 8448 # the number of features, CHANGE FOR DEBUGGING
featuretbl = np.zeros((nfeatures,8,2))

# training set labels
labels = [1]*nimages + [0]*nimages
inverted_labels = [0]*nimages + [1]*nimages

# images included in the first stage of the classification cascade
# updated after each stage of the classification cascade
curr_images = [i for i in range(nimages*2)]

# computation of iimage
def iimageRep(img):
        # sum of pixel values in image i in the rectangle extending from (0,0) to (x,y)
        # i.e. sum of all pixels above and to the left
        ii = np.zeros(width*height)
        s = np.zeros(width*height)

        for y in range(width):
                for x in range(height):
                        if x == 0:
                                s[(y*width)+x] = img[(y*width)+x]
                        else:
                                s[(y*width)+x] = s[(y*width)+x-1] + img[(y*width)+x]
                        if y == 0:
                                ii[(y*width)+x] = s[(y*width)+x]
                        else:
                                ii[(y*width)+x] = ii[((y-1)*width)+x] + s[(y*width)+x]
        return ii

# feature set generation
def featureGenerator():

	stride = 8 # CHANGE FOR DEBUGGING
	max_width = 32 # CHANGE FOR DEBUGGING
	max_height = 16 # CHANGE FOR DEBUGGING

	n = 0
	for i in range(2,max_width,2):
		for j in range(2,max_height,2):
			for i2 in range(i,width,stride):
				for j2 in range(j,height-j,stride):
					# horizontal rectangles, and flip
					featuretbl[n,:,:] = np.array([(i2-i,j2-j), (i2,j2-j), (i2-i,j2), (i2, j2), \
								(i2-i,j2), (i2,j2), (i2-i,j2+j), (i2,j2+j)])
					###featuretbl[n+1,:,:] = np.array([(i2-i,j2), (i2,j2), (i2-i,j2+j), (i2,j2+j), \
								###(i2-i,j2-j), (i2,j2-j), (i2-i,j2), (i2, j2)])				
					# vertical rectangles, and flip
					featuretbl[n+1,:,:] = np.array([(i2-i,j2-j), (i2-i/2,j2-j), (i2-i,j2+j), (i2-i/2,j2+j), \
								(i2-i/2,j2-j), (i2,j2-j), (i2-i/2,j2+j), (i2,j2+j)])
					###featuretbl[n+3,:,:] = np.array([(i2-i/2,j2-j), (i2,j2-j), (i2-i/2,j2+j), (i2,j2+j), \
								###(i2-i,j2-j), (i2-i/2,j2-j), (i2-i,j2+j), (i2-i/2,j2+j)])
					n += 2

# load each face image, convert to greyscale, and compute iimage representation
for image in range(nimages):
	x = Image.open('faces/face%i.jpg' % image, 'r')
	x = x.convert('L') # makes it greyscale
	y = np.asarray(x.getdata(),dtype=np.float64).reshape(x.size[0]*x.size[1])
	iimages[image,:] = iimageRep(y)

# load each background image, convert to greyscale, and compute iimage representation
for image in range(nimages):
	x = Image.open('background/%i.jpg' % image, 'r')
        x = x.convert('L') # makes it greyscale
        y = np.asarray(x.getdata(),dtype=np.float64).reshape(x.size[0]*x.size[0])
	iimages[image+nimages,:] = iimageRep(y)

# generate the feature set
featureGenerator()

def computeFeature(img,f):
	# compute feature value for feature f of image img
	# intensity of black rectangle - intensity of white rectangle (black always first)
	
	points = featuretbl[f,:4,:]	
	x1 = points[0,0]
        y1 = points[0,1]
        x2 = points[1,0]
        y2 = points[1,1]
        x3 = points[2,0]
        y3 = points[2,1]
        x4 = points[3,0]
        y4 = points[3,1]
        value1 = iimages[img,y4*width+x4] + iimages[img,y1*width+x1] - iimages[img,y2*width+x2] - iimages[img,y3*width+x3]

	points = featuretbl[f,4:,:]
	x1 = points[0,0]
        y1 = points[0,1]
        x2 = points[1,0]
        y2 = points[1,1]
        x3 = points[2,0]
        y3 = points[2,1]
        x4 = points[3,0]
        y4 = points[3,1]
        value2 = iimages[img,y4*width+x4] + iimages[img,y1*width+x1] - iimages[img,y2*width+x2] - iimages[img,y3*width+x3]

	return value1-value2

def boost(n):
	
	# compute number of negatives in the sample
        negatives = n - nimages

	# initialize the weight vector
	w = np.zeros(n)
	w[:nimages].fill(1/float(2*nimages))
	w[nimages:].fill(1/float(2*negatives))

	# hold the final classifier
	final = []

	# iterate until false positive rate below 30%
	t = 0
	fp = 1 # initialize false positive rate
	while fp > 0.3:
		print('boosting')
		# normalize the weights to make a probability distribution
		w = np.true_divide(w,np.sum(w))

		# pick the best classifier, i.e. select a single feature
		# f is the feature, theta the threshold, p the polarity, err the error, and misses the misclassifications 
		f,theta,p,err,misses = bestLearner(w,n)

		# update the weights based on the chosen classifier
		beta = err/(1-err)
		alpha = np.log(1/beta)
		e = misses
		for i in range(n):
			w[i] = w[i]*(beta**(1-e[i]))
		
		# update the final classifier with the hypothesis and alpha			
		if t == 0:
			final = [(f,theta,p,alpha)]
		else:
			final.append((f,theta,p,alpha))

		# compute the final theta
		pos_results = [0]*nimages # results of positives with the current final classifier
		for i in range(nimages):
			result = np.sum([h[3]*int(h[2]*computeFeature(curr_images[i],h[0]) < h[2]*h[1]) for h in final])
			pos_results[i] = result
		final_theta = min(pos_results) 

		# compute the false positive rate
		false_positives = 0 
		for i in range(nimages,n):
			result = np.sum([h[3]*int(h[2]*computeFeature(curr_images[i],h[0]) < h[2]*h[1]) for h in final])
			if result >= final_theta:
				false_positives += 1
		fp = float(false_positives/negatives) 
		t += 1

	# determine correctly identified negatives by the final classifier
	correct_negs = []
	for i in range(nimages,n):
		# make prediction with final classifier
		result = np.sum([h[3]*int(h[2]*computeFeature(curr_images[i],h[0]) < h[2]*h[1]) for h in final]) - final_theta
		if result < 0: 
			if not correct_negs:
				correct_negs = [curr_images[i]]
			else:
				correct_negs.append(curr_images[i])	

	# determine the training error of the classifier
	loss = 0
	for i in range(n):
        	result = np.sum([h[3]*int(h[2]*computeFeature(curr_images[i],h[0]) < h[2]*h[1]) for h in final]) - final_theta
        	predict = int(result > 0)
        	if predict != labels[curr_images[i]]:
                	loss += 1
	train_error = loss/float(nimages*2)
	
	print('boosting completed')
	print(train_error)

	# return the final classifier, final theta, and indices of correct negatives
	return (final,final_theta,correct_negs)

def bestLearner(w,n):

	# return the feature, threshold, polarity, error, and missclassifications
	# of the best learner on weight vector w

	# determine values for the first feature
	f = 0
	theta,p,err = hypothesis(0,w,n)	

	# compare to rest of the features to determine the best one (lowest err)
	for i in range(1,nfeatures):
		theta_tmp,p_tmp,err_tmp = hypothesis(i,w,n)
		if err_tmp < err:
			f = i
			theta = theta_tmp
			p = p_tmp
			err = err_tmp

	# determine misclassifications for the best learner
	misses = [0]*n # 0 for correct predictions
	for i in range(n):
		predict = int(p*computeFeature(curr_images[i],f) < p*theta)
		if predict != labels[curr_images[i]]:
			misses[i] = 1 # 1 for incorrect predictions
	
	return (f,theta,p,err,misses)

def hypothesis(f,w,n):
	# f is the feature, w is the weight vector

	# compute the feature values on the current set of images 
	values = np.array([computeFeature(curr_images[i],f) for i in range(n)])

	# find indices of the images when sorted by this feature
	order = np.argsort(values)

	# hold errors and polarity for each value of j
	errors = [0]*(n)
	polarity = [1]*(n)

	# total weight of + examples
        Tplus = np.sum([labels[curr_images[i]]*w[i] for i in range(n)])
        # total weight of - examples
	Tminus = np.sum([inverted_labels[curr_images[i]]*w[i] for i in range(n)])

	Splus = 0 # total weight of + examples to the left
	Sminus = 0 # total weight of - examples to the left

	# compute the error and polarity for all values j
	for j in range(n):
		if labels[curr_images[order[j]]]:
			Splus += w[order[j]]
		else:
			Sminus += w[order[j]]
		
		# compute the error and polarity by which side is smaller
		side = int(Splus+Tminus-Sminus > Sminus+Tplus-Splus)
		if side == 0:
			polarity[j] = -1 # otherwise, keep at 1
			errors[j] = Splus+Tminus-Sminus
		else:
			errors[j] = Sminus+Tplus-Splus

	# find j with the minimum error and its polarity
	best_j = errors.index(min(errors))
	err = errors[best_j]
	p = polarity[best_j]	

	# set threshold between feature value at ordered index best_j and feature value at ordered index best_j+1
	if (best_j+1) == n:
		theta = values[order[best_j]]
	else: 
		theta = values[order[best_j]] + (values[order[best_j+1]]-values[order[best_j]])/2.0

	# return threshold, polarity, and err
	return (theta,p,err)

def classifierCascade():

	classifiers = [] # holds the cascade of classifiers

	n = nimages*2 # run the first round of boosting on all training images

	# iterate until 5% of incorrectly classified negatives remain
	while n > (nimages + nimages/20):
		print('new cascade stage')
		ts = time.time()
		st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
		print(st)		

		classifier, theta, correct_negs = boost(n)

		# add the classifier to the cascade
		if not classifiers:
			classifiers = [(classifier,theta)]
		else:
			classifiers.append((classifier,theta))

		# remove the correctly identified negatives from curr_images
		for image in correct_negs:
			curr_images.remove(image)

		# subtract the number of correctly identified negatives from n
		n -= len(correct_negs)
		print(n)

	return classifiers

# obtain the classifier cascade from the training data
classifiers = classifierCascade()

classifiers = np.array(classifiers)
np.save('classifiers', classifiers)

# functions specific to the test data

def computeFeature_test(iimage,f):
        # compute feature value for feature f of the test iimage
        # intensity of black rectangle - intensity of white rectangle (black always first)
       
	points = featuretbl[f,:4,:]
        x1 = points[0,0]
        y1 = points[0,1]
        x2 = points[1,0]
        y2 = points[1,1]
        x3 = points[2,0]
        y3 = points[2,1]
        x4 = points[3,0]
        y4 = points[3,1]
        value1 = iimage[y4*width+x4] + iimage[y1*width+x1] - iimage[y2*width+x2] - iimage[y3*width+x3]

        points = featuretbl[f,4:,:]
        x1 = points[0,0]
        y1 = points[0,1]
        x2 = points[1,0]
        y2 = points[1,1]
        x3 = points[2,0]
        y3 = points[2,1]
        x4 = points[3,0]
        y4 = points[3,1]
        value2 = iimage[y4*width+x4] + iimage[y1*width+x1] - iimage[y2*width+x2] - iimage[y3*width+x3]

        return value1-value2

# load the test image
x = Image.open('class.jpg')
x = x.convert('L')
im = np.array(x, dtype=np.uint8)
fig,ax = plt.subplots(1)
ax.imshow(im, cmap ='gray')

test_width = x.size[0]
test_height = x.size[1]

###detections = []

# slide a 64x64 window over the test image
for i in range(width,test_width,16):
        for j in range(height,test_height,16):
                # compute the iimage representation of the patch
                image = x.copy()
                selected_patch = image.crop((i-width,j-height,i,j))
                y = np.asarray(selected_patch.getdata(),dtype=np.float64).reshape(width*height)
                iimage = iimageRep(y)

                # run through the cascade of classifiers
                face = True
                for (classifier,theta) in classifiers:
                        result = np.sum([h[3]*int(h[2]*computeFeature_test(iimage,h[0]) < h[2]*h[1]) for h in classifier]) - theta
                        if result < 0:
                                face = False
                                break
                        # otherwise, continue through the cascade
                        # only if complete the cascade, count as a face

                if face:
                        # draw a square
			window = [(i-64,j-64), (i,j-64), (i-64,j), (i,j)]
			###detections += [window]
                        rect = patches.Rectangle((i-64,j-64),width,height,linewidth=3,edgecolor='red',facecolor='none')
        		ax.add_patch(rect)

"""# create a dictionary to hold the sets
sets = {}

for index,window in enumerate(detections):
        placed = False
        if not sets:
                sets[index] = [window]
                placed = True
        else:
                for key,windows in sets.iteritems():
                        overlap = True
                        # iterate through windows to see if overlaps with all of them
                        for element in windows:
                                if element[1][0] < window[0][0] or element[3][1] < window[1][1]:
                                        # the windows do not overlap
                                        overlap = False
                                        break
                        if overlap:
                                sets[key] = windows + [window]
                                placed = True
                        if placed:
                                break
        if not placed:
                # create a new set
                sets[index] = [window]

# for each set, average the overlapping values
for key,windows in sets.iteritems():
        l = [i[0][0] for i in windows]
        x1 = sum(l)/float(len(l))
	l = [i[0][1] for i in windows]
        y1 = sum(l)/float(len(l))
        l = [i[1][0] for i in windows]
        x2 = sum(l)/float(len(l))
        l = [i[2][1] for i in windows]
        y3 = sum(l)/float(len(l))

	rect = patches.Rectangle((x1,y1),x2-x1,y3-y1,linewidth=2,edgecolor='red',facecolor='none')
        ax.add_patch(rect)"""

fig.savefig('FaceDetectionResults.jpg')	
