# CMSC 25400
# Larissa Clopton

# dimensionality reduction
# apply PCA and Isomap to map 3D data to 2D data

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math

# read in the data
with open('3Ddata.txt') as input:
	data3D = np.array([list(map(float,line.split())) for line in input])

def PCA(dataset):

	# organize the data
	type = dataset[:,3] # for visualization purposes
	dataset = np.transpose(dataset[:,[0,1,2]])

	# center data around the mean
	dataset[0,:] = dataset[0,:] - np.mean(dataset[0,:])
	dataset[1,:] = dataset[1,:] - np.mean(dataset[1,:])
	dataset[2,:] = dataset[2,:] - np.mean(dataset[2,:])

	# compute sample covariance matrix
	sample_cov = np.zeros((3,3))
	for i in range(dataset.shape[1]):
		sample_cov += (dataset[:,i].reshape(3,1)).dot((dataset[:,i].reshape(3,1)).T)
	sample_cov = np.true_divide(sample_cov,dataset.shape[1])	

	# form and sort (eig_val, eig_vec) tuples
	eig_val, eig_vec = np.linalg.eig(sample_cov)
	eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_val))]
	eig_pairs.sort(key=lambda x: x[0], reverse=True)

	# isolate eigenvectors with top 2 eigenvalues
	feature_matrix = np.hstack((eig_pairs[0][1].reshape(3,1), eig_pairs[1][1].reshape(3,1)))

	# project dataset onto the 2D subspace
	data_trans = feature_matrix.T.dot(dataset)

	# plot the transformed data
	colors = cm.rainbow(np.linspace(0, 1, 4))
	for i, c in enumerate(type):
		plt.scatter(data_trans[0,i], data_trans[1,i], color = colors[c-1])
	plt.xlabel('x')
	plt.ylabel('y')
	plt.axis([-3, 3, -1.5, 1.5])
	plt.title('PCA: Transformed 2D data')
	plt.savefig('PCA-transform.png')

def Isomap(dataset):

	# organize the data
        type = dataset[:,3] # for visualization purposes
        dataset = np.transpose(dataset[:,[0,1,2]])
	n = dataset.shape[1] # the number of points
	k = 10 # number of nearest neighbors

	# center data around the mean
        dataset[0,:] = dataset[0,:] - np.mean(dataset[0,:])
        dataset[1,:] = dataset[1,:] - np.mean(dataset[1,:])
        dataset[2,:] = dataset[2,:] - np.mean(dataset[2,:])

	# initialize the squared distance matrix
	D = np.zeros((n,n))
	D[:,:] = sys.maxint

	# create knn graph for k = 10 nearest neighbors
	for i in range(n):
		sq_distances = [np.linalg.norm(dataset[:,i] - dataset[:,j]) for j in range(n)]
		# sort to find closest k points
		knn_index = (np.argsort(sq_distances))[:k+1]
		# place these values in D
		for j in range(k+1):
			D[i,knn_index[j]] = sq_distances[knn_index[j]]

	# ensure the distance matrix is symmetric
	D = np.minimum(D, np.transpose(D))

	# use Floyd-Warshall algorithm for shortest paths	
	G = np.square(shortest_paths(D))

	# obtain the centered Gram matrix
	P = np.eye(n) - np.ones((n,n))/n
	B = -P.dot(G).dot(P)/2

	# take eigendecomposition of B
	dimension = 2
	eig_val, eig_vec = np.linalg.eigh(B)
	eig_val = np.absolute(eig_val)
	idx = np.argsort(eig_val)[::-1]
	eig_val = eig_val[idx]
	eig_vec = eig_vec[:,idx]

	# isolate top 2 dimensions 
	feature_matrix = np.zeros((n,dimension))
	feature_matrix[:,0] = eig_vec[:,0]
	feature_matrix[:,1] = eig_vec[:,1]
	l1_sqrt = math.sqrt(eig_val[0])
	l2_sqrt = math.sqrt(eig_val[1])
	lambda_sqrt = np.array([[l1_sqrt, 0],[0, l2_sqrt]])

	# project the data
	data_trans = (feature_matrix.dot(lambda_sqrt)).T 

	# plot the transformed data
	colors = cm.rainbow(np.linspace(0, 1, 4))
        for i, c in enumerate(type):
                plt.scatter(data_trans[0,i], data_trans[1,i], color = colors[c-1])	
	plt.xlabel('x')
	plt.ylabel('y')
	plt.title('Isomap: Transformed 2D data')
	plt.savefig('Isomap-transform.png')

def shortest_paths(distances):

	# Floyd-Warshall algorithm
	# distances[i][j] -> shortest path from vertex i to j
	n = distances.shape[1]
	for k in range(n):
		for i in range(n):
			for j in range(n):
				if distances[i,j] > (distances[i,k] + distances[k,j]):
					distances[i,j] = distances[i,k] + distances[k,j]

	return distances

plt.figure(1)
PCA(data3D)
plt.figure(2)
Isomap(data3D)
