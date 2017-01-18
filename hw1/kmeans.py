# Larissa Clopton
# CMSC 25400

# A python implementation of the kmeans algorithm

import numpy as np
import random

def kmeans(dataset, k, init_centers=None):

	# initialize k random centers
	if init_centers is None:
		centers = random.sample(dataset, k)
	else:
		centers = init_centers

	centers_buff = [[] for i in range(k)]

	# track distortion_function across iterations
	distortion_function = []

	while not (has_converged(centers, centers_buff)):

		# assign data points to clusters
		clusters = [[] for i in range(k)]
		clusters,distortion = build_clusters(dataset, k, centers, clusters)

		try:
			distortion_function.append(distortion)
		except KeyError:
			distortion_function = [distortion]	

		index = 0
		for cluster in clusters:
			# store previous centers
			centers_buff[index] = centers[index]
			# recalculate centers
			centers[index] = np.mean(cluster, axis=0).tolist()
			index += 1

	return (clusters,distortion_function) 
	

def build_clusters(dataset, k, centers, clusters):

	distortion = 0 # sum of squared distances

	for point in dataset:
		# find closest center
		closest_index = min([(i[0], np.linalg.norm(point-centers[i[0]])) \
							for i in enumerate(centers)], \
							key=lambda t:t[1])[0]
		try:
			clusters[closest_index].append(point)
		except KeyError:
			clusters[closest_index] = [point]

		distortion += np.linalg.norm(point-centers[closest_index],ord=2)**2


	return (clusters,distortion)


def has_converged(centers, centers_buff):

	# algorithm has converged when centers do not change
	return (set([tuple(a) for a in centers]) == set([tuple(b) for b in centers_buff]))

