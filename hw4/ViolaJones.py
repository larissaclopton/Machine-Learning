# Larissa Clopton
# CMSC 25400

# an implementation of the Viola Jones face detection cascade

import numpy as np
from PIL import Image
import math

# define global variables

# the iimage representation for all images
nimages = 2000 # the number of images
iimages = np.zeros((nimages*2,64**2)) # first half holds background, second half holds faces

# store definitions of the rectangles
nfeatures = 540672 # the number of features
featuretbl = np.zeros((nfeatures,8))

# width and height of the image
width = 64
height = 64

# computation of iimage
def iimage_rep(img):
        # sum of pixel values in image i in the rectangle extending from (0,0) to (x,y)
        # i.e. sum of all pixels above and to the left
        ii = np.zeros(width*height)
        s = np.zeros(width*height)

        for y in range(height):
                for x in range(width):
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
def feature_generator:
	n = 0
	for i in range(2,64+2,2):
		for j in range(2,32+2,2):
			for i2 in range(i,64+2,2):
				for j2 in range(j,64-j+2,2):
					# horizontal rectangles, and flip
					featuretbl[n,:] = np.array([(i2-i,j2-j), (i2,j2-j), (i2-i,j2), (i2, j2),\
								(i2-i,j2), (i2,j2), (i2-i,j2+j), (i2,j2+j)])
					featuretbl[n+1,:] = np.array([(i2-i,j2), (i2,j2), (i2-i,j2+j), (i2,j2+j), \
								(i2-i,j2-j), (i2,j2-j), (i2-i,j2), (i2, j2)])				
					# vertical rectangles, and flip
					featuretbl[n+2,:] = np.array([(i2-i,j2-j), (i2-i/2,j2-j), (i2-i,j2+j), (i2-i/2,j2+j), \
								(i2-i/2,j2-j), (i2,j2-j), (i2-i/2,j2+j), (i2,j2+j)])
					featuretbl[n+3,:] = np.array([(i2-i/2,j2-j), (i2,j2-j), (i2-i/2,j2+j), (i2,j2+j), \
								(i2-i,j2-j), (i2-i/2,j2-j), (i2-i,j2+j), (i2-i/2,j2+j)])
					n = n + 4

# load each background image, convert to greyscale, and compute iimage representation
for image in range(nimages):
	x = Image.open('background/%i.jpg' % image, 'r')
	x = x.convert('L') # makes it greyscale
	y = np.asarray(x.getdata(),dtype=np.float64).reshape(x.size[0]*x.size[1])
	iimages[image,:] = iimage_rep(y)

# load each face image, convert to greyscale, and compute iimage representation
for image in range(nimages):
	x = Image.open('faces/face%i.jpg' % image, 'r')
        x = x.convert('L') # makes it greyscale
        y = np.asarray(x.getdata(),dtype=np.float64).reshape(x.size[0]*x.size[0])
	iimages[image+nimages,:] = iimage_rep(y)

# define labels for the training set
labels = [0]*(nimages*2)
labels[nimages:] = 1
inverted_labels = [1]*(nimages*2)
inverted_labels[nimages:] = 0

# generate the feature set
# featuretbl = generate_feature_set()

def rectangleIntensity(points,img):
	# compute the sum of the pixels within the defined rectangle	
	# extract the coordinates of each vertex of the rectangle
	x1 = points[0][0]
	y1 = points[0][1]
	x2 = points[1][0]
	y2 = points[1][1]
	x3 = points[2][0]
	y3 = points[2][1]
	x4 = points[3][0]
	y4 = points[3][1]

	value = iimages[img,y4*width+x4] + iimages[img,y1*width+x1] - iimages[img,y2*width+x2] - iimages[img,y3*width+x3]
	return value

def computeFeature(i,f):
	# compute feature value for feature f of image i
	# intensity of black rectangle - intensity of white rectangle (black always first)
	coordinates = featuretbl[f,:]	

	value = rectangleIntensity(coordinates[:4],i) - rectangleIntensity(coordinates[4:],i)
	return value

def boost(n,flag_array):

	# initialize the weight vector
	w = np.zeros(n)
	w.fill(float(1/n)
	
	# iterate until false positive rate below 30%
	t = 0
	false_pos = 1
	while false_pos >= 0.3:
		# normalize the weights to make a probability distribution
		w = np.true_divide(w,np.sum(w))

		# pick the best classifier, each restricted to a single feature
		# f is the feature, theta the threshold, p the polarity, err the error, and misses the misclassifications 
		f,theta,p,err,misses = bestLearner(w,flag_array,n)

		# update the weights based on the chosen classifier
		beta = err/(1-err)
		alpha = math.log(1/beta)
		e = [0]*(n) # 0 for correct predictions
		e[misclassifications] = 1 # 1 for incorrect predictions
		for i in range(n):
			w[i] = w[i]*(beta**(1-e[i]))
		
		
		# UPDATE the final classifier with the hypothesis and alpha			
		# COMPUTE the new theta (minimum final classifier value for + examples)
		# COMPUTE the false positive rate (fraction of - examples for which final classifier >= theta)

		t += 1

	# RETURN the final strong classifier
	# RETURN how many incorrect - classifications were updated

def bestLearner(w,flag_array,n):

	# return the feature, threshold, polarity, error, and missclassifications
	# of the best learner on weight vector w

	# determine values for the first feature
	f = 0
	theta,p,err = hypothesis(0,w,flag_array,n)	

	# compare to rest of the features to determine the best one (lowest err)
	for i in range(1,nfeatures):
		theta_tmp,p_tmp,err_tmp = hypothesis(i,w,flag_array,n)
		if err_tmp < err:
			f = i
			theta = theta_tmp
			p = p_tmp
			err = err_tmp

	# determine misclassifications for the best learner
	misses = []
	subset_index = -1
	for i in range(nimages*2):
		if flag_array[i]:
			subset_index += 1
			if p*computeFeature(i,f) < p*theta:
				predict = 1
			else:
				predict = 0
			if predict != labels[i]:
				if not misses:
					misses = [subset_index]
				else:
					misses.append(subset_index)		

	return (f,theta,p,err,misses)

def hypothesis(f,w,flag_array,n):
	# f is the feature, w is the weight vector

	# find the indices of the subset of images and compute their feature values
	subset = [] 
	for i in range(nimages*2):
		if flag_array[i]:
			if not subset:
				subset = [computeFeature(i,f)]
			else:
				subset.append(computeFeature(i,f))	
	
	values = np.array(subset)

	# find indices of the images when sorted by this feature
	order = np.argsort(values)

	# hold errors for each value of j
	errors = [0]*(n)
	polarity = [1]*(n)

	# compute the error and polarity for all values j
	for j in range(n):
		proper_indices = [getIndex(order[k]) for k in range(j)]
		# total weight of + examples to the left
		Splus = np.sum([labels[proper_indices[k]]*w[order[k]] for k in range(j)])		
		###Splus = np.sum([labels[order[k]]*w[order[k]] for k in range(j)]) 
		# total weight of - examples to the left
		Sminus = np.sum([inverted_labels[proper_indices[k]]*w[order[k]] for k in range(j)])
		###Sminus = np.sum([inverted_labels[order[k]]*w[order[k]] for k in range(j)]) 
		
		proper_indices = [getIndex(order[k]) for k in range(n)]
		# total weight of + examples
		Tplus = np.sum([labels[proper_indices[k]]*w[order[k]] for k in range(n)])
		###Tplus = np.sum([labels[order[k]]*w[order[k]] for k in range(n)]) 
		# total weight of - examples
		Tminus = np.sum([inverted_labels[proper_indices[k]]*w[order[k]] for k in range(n)])
		###Tminus = np.sum([inverted_labels[order[k]]*w[order[k]] for k in range(n)])
		
		# compute the error and polarity by which side is smaller
		vals = [Splus+Tminus-Sminus, Sminus+Tplus-Splus]
		errors[j] = min(values)
		side = vals.index(min(vals))
		if side == 0:
			polarity[j] = -1

	# find j with the minimum error and its polarity
	best_j = errors.index(min(errors))
	err = errors[best_j]
	p = polarity[best_j]	

	# set threshold between feature at ordered index j and feature at ordered index j+1
	theta = values[order[best_j]] + (values[order[best_j+1]]-values[order[best_j]])/2.0

	# return threshold, polarity, and err
	return (theta,p,err)
	 
def getIndex(relative_index):
	# return the proper index of the subset in the entire training set	
