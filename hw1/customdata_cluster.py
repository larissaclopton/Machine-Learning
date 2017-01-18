# Larissa Clopton
# CMSC 25400

""" NOTE: This code does not always run, I'm still trying to fix the bug.
    It normally runs within a few attempts. When it does run, it produces
    the expected result. """

# Create a custom dataset for which kmeans++ improves kmeans 
# by large factor in terms of distortion function

import kmeans as km
import kmeansplus as kmplus
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import random

# set the number of clusters
k = 7

# create k points (centers) that are uniformly distributed
centers = np.array([(random.uniform(0,100),random.uniform(0,100)) for i in range(k)])

min_distances = [0]*k
for index in range(len(centers)):

	# remove current center from list
	curr_center = centers[0]
	np.delete(centers,0)
		
	# find closest center to curr_center
	closest_center = min([(i[0], np.linalg.norm(curr_center-centers[i[0]])) \
                                                        for i in enumerate(centers)], \
                                                        key=lambda t:t[1])[0] 	
	
	# record distance to this center
	min_dist = np.linalg.norm(curr_center-centers[closest_center]) 

	# put current center back in list
        np.append(centers,curr_center)

	min_distances[index] = min_dist

# buffer of epsilon between circles of points
ep = 1 # what value should this be?

# points per cluster, total datapoints = ppc*k
# should this vary per cluster based on size?
ppc = 20

# to hold the custom dataset
custom_data = []

for index,min_dist in enumerate(min_distances):

	r = min_dist/2 - ep # circle radius
	xcenter,ycenter = centers[index] # circle center
		
	# draw ppc points uniformly from the defined circle
	points = []
	for i in range(ppc):
		# random angle
		theta = 2*np.pi*random.random()
		# random radius
		rad = r*random.random()
		# coordinates
		x = r*np.cos(theta)+xcenter
		y = r*np.sin(theta)+ycenter

		try:
			points.append((x,y))
		except KeyError:
			points = [(x,y)]

	custom_data = custom_data + points	

data = np.array(custom_data)

plt.figure(0)
plt.xlim(0,5)
plt.ylim(0,80000)
plt.xlabel('Iteration')
plt.ylabel('Distortion')
plt.title('Kmeans: distortion vs iteration (custom dataset)')

for i in range (0,20):

        clusters,distortion_function = km.kmeans(data,k)

        # plot index (iteration) against value (distortion)
        plt.plot(distortion_function)

plt.figure(0)
plt.savefig('CustomKmeansDistortion.png')

plt.figure(1)
plt.xlim(0,5)
plt.ylim(0,8000)
plt.xlabel('Iteration')
plt.ylabel('Distortion')
plt.title('Kmeans++: distortion vs iteration (custom dataset)')

for i in range (0,20):

        clusters,distortion_function = kmplus.kmeansplus(data,k)
        plt.plot(distortion_function)

plt.savefig('CustomKmeans++Distortion.png')
