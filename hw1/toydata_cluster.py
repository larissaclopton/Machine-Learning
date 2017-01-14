import kmeans as km
import kmeansplus as kmplus
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# retrieve dataset and desired number of clusters
k = 3

with open('toydata.txt') as input:
	toy_data = np.array([tuple(map(float, line.split())) for line in input])


# For each run of the kmeans algorithm:
# plot final result of clustering where point color indicates assignment
# plot distortion function as function of iteration
# COMMENT on the distortion function plot

colors = mpl.cm.rainbow(np.linspace(0, 1, k))

plt.figure(0)
plt.xlabel('Iteration') 
plt.ylabel('Distortion')
plt.title('Kmeans: distortion vs iteration')

plt.figure(1)
plt.figure(1).subplots_adjust(hspace=1)
plt.title('Kmeans Clustering')


for i in range (0,20):

	clusters,distortion_function = km.kmeans(toy_data,k)

	plt.figure(1)
	plt.subplot(4,5,i+1)
        plt.title('Trail (%d)' % (i+1))	

	for index, cluster in enumerate(clusters):
		plt.scatter(*zip(*cluster),color = colors[index])
	
	# plot index (iteration) against value (distortion)
	plt.figure(0)
	plt.plot(distortion_function)

plt.figure(0)
plt.savefig('KmeansDistortion.png')

plt.figure(1)
plt.savefig('KmeansClusters.png')


# For each run of the kmeans++ algorithm:
# plot distortion function as function of iteration
# COMMENT on convergence of kmeans++ vs kmeans

plt.figure(2)
plt.xlabel('Iteration')
plt.ylabel('Distortion')
plt.title('Kmeans++: distortion vs iteration')

for i in range (0,20):

        clusters,distortion_function = kmplus.kmeansplus(toy_data,k)
        plt.plot(distortion_function)

plt.savefig('Kmeans++Distortion.png')
