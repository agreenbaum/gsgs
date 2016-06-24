import numpy as np
import matplotlib.pyplot as plt

def makehex(origin,d,niter=2,anglefrac=6):
	'''---------------------------------------------------
	makehex.py - populates a hexagonal grid by iteratively
	bisecting baselines between points already on the 
	grid.
	niter = number of times to iterate
	anglefrac = fraction of 180 degrees to define symmetry
	---------------------------------------------------'''
	#common quantities

	r = d/2.5
	theta=np.pi/anglefrac #(30./180.* np.pi)     #angle when moving diagonally across segments
	dct=d*np.cos(theta) 
	dst=d*np.sin(theta) 
	nsteps = 5

	# initial point
	# create the six border points

	npoints = 100000 # some stupidly large number

	points = np.zeros((npoints,2))

	i = 0 
	points[0,:] = origin
	i += 1

	step = r/float(nsteps)
	a = np.array([step,0])
	b = np.array([step*np.cos(theta),step*np.sin(theta)])

	for angle in theta*2*np.arange(6):
		points[i,0] = origin[0] + d/2*np.cos(angle)
		points[i,1] = origin[1] + d/2*np.sin(angle)
		i += 1
	imax=i

	'''------------------------
	Now find the midpoints of
	all the separate baselines
	------------------------'''
	tol = 1e-3*d

	for it in range(niter):
		for j in range(imax):
			for k in range(imax):
				newpoint = (points[j,:]+points[k,:])/2
				dists = np.sqrt(np.sum((points-newpoint)**2,axis=1))
				if np.min(dists) >= tol:
					points[i,:] = newpoint
					i += 1
		imax = i

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


