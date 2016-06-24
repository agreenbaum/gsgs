import numpy as np
import matplotlib.pyplot as plt

def makecirc(origin,d,nrings=5):
	'''---------------------------------------------------
	makehex.py - populates a hexagonal grid by iteratively
	bisecting baselines between points already on the 
	grid.
	niter = number of times to iterate
	anglefrac = fraction of 180 degrees to define symmetry
	---------------------------------------------------'''
	#common quantities

	r = d/2.5
	theta=np.pi/10. #(30./180.* np.pi)     #angle when moving diagonally across segments
	dct=d*np.cos(theta) 
	dst=d*np.sin(theta) 

	# initial point
	# create the six border points

	npoints = 100000 # some stupidly large number
	nangles = 10

	points = np.zeros((npoints,2))

	i = 0 
	points[0,:] = origin
	i += 1
	tol = 1e-4*d

	for k,l in enumerate(np.linspace(0,d,nrings)):
		for angle in np.linspace(0,2*np.pi,k*nangles):
			newpoint = np.array([l/2.2*np.cos(angle),l/2.2*np.sin(angle)])+origin
			dists = np.sqrt(np.sum((points-newpoint)**2,axis=1))
			if np.min(dists) >= tol:
				points[i,:] = newpoint
				i += 1
		imax=i

	'''------------------------
	Now find the unique set of 
	points to return
	------------------------'''	

	fullset = np.array(list(set(tuple(p) for p in points)))

	plt.clf()
	plt.scatter(fullset[:,0],fullset[:,1])
	plt.axis('equal')
	plt.xlabel('x (m)')
	plt.ylabel('y (m)')
	plt.title('Dragonfly MEMS Mirror Model')
	plt.show()

	return fullset


