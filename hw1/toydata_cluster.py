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

colors = mpl.cm.rainbow(np.linspace(0, 1, k))

plt.figure(0)
plt.figure(0).subplots_adjust(hspace=0.5)
plt.title('Kmeans Clustering (toy dataset)')

plt.figure(1)
plt.figure(1).subplots_adjust(hspace=0.5)
plt.title('Kmeans Clustering (toy dataset)')

plt.figure(2)
plt.figure(2).subplots_adjust(hspace=0.5)
plt.title('Kmeans Clustering (toy dataset)')

plt.figure(3)
plt.figure(3).subplots_adjust(hspace=0.5)
plt.title('Kmeans Clustering (toy dataset)')

plt.figure(4)
plt.figure(4).subplots_adjust(hspace=0.5)
plt.title('Kmeans Clustering (toy dataset)')

plt.figure(5)
plt.xlabel('Iteration')
plt.ylabel('Distortion')
plt.title('Kmeans: distortion vs iteration (toy dataset)')

for i in range(0,20):

	clusters,distortion_function = km.kmeans(toy_data,k)

	plt.figure(i/4)
	plt.subplot(2,2,i%4+1)
        plt.title('Trial (%d)' % (i+1))	

	for index, cluster in enumerate(clusters):
		plt.scatter(*zip(*cluster),color = colors[index])
	
	# plot index (iteration) against value (distortion)
	plt.figure(5)
	plt.plot(distortion_function)

plt.figure(5)
plt.savefig('ToyKmeansDistortion.png')

plt.figure(0)
plt.savefig('ToyKmeansClusters1_4.png')

plt.figure(1)
plt.savefig('ToyKmeansClusters5_8.png')
plt.figure(2)
plt.savefig('ToyKmeansClusters9_12.png')
plt.figure(3)
plt.savefig('ToyKmeansClusters13_16.png')
plt.figure(4)
plt.savefig('ToyKmeansClusters17_20.png')

# For each run of the kmeans++ algorithm:
# plot distortion function as function of iteration

plt.figure(6)
plt.xlabel('Iteration')
plt.ylabel('Distortion')
plt.title('Kmeans++: distortion vs iteration (toy dataset)')

for i in range (0,20):

        clusters,distortion_function = kmplus.kmeansplus(toy_data,k)
        plt.plot(distortion_function)

plt.savefig('ToyKmeans++Distortion.png')
