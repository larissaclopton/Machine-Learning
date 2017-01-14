# A python implementation of the k-means++ algorithm,
# an augmented version of the k-means algorithm

# Centers chosen at random from data points, but data points are 
# weighed according to distance from closest center already chosen

# See Arthur and Vassilbitskii (2007)

import kmeans as km
import numpy as np
import random

def kmeansplus(dataset,k):

	# choose m1 uniformly at random from data points
	init_centers = random.sample(dataset, 1)

	# choose following centers based on probability weighted by distance from centers
	for i in range(1,k):
		dist_centers = np.array([min([np.linalg.norm(point-center)**2 \
				for center in init_centers]) for point in dataset])

		init_centers.append(dataset[choose_next_center(dist_centers)])

	clusters,distortion_function = km.kmeans(dataset,k,init_centers)

	return (clusters,distortion_function)

def choose_next_center(dist_centers):
	dist_sum = np.sum(dist_centers)
	probs = [dist/dist_sum for dist in dist_centers]

	cumprobs = np.cumsum(probs)
	r = random.random() # a random probability

	index = np.where(cumprobs >= r)[0][0]
	
	return index		
	
