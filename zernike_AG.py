#! /usr/bin/env python

"""
Alex's zernike script, with reference to 
http://www.staff.science.uu.nl/~werkh108/docs/teach/2011b_python/course/python102/python_102-print.pdf
by Tim Werkhoven
"""

import numpy as np
import pylab as pl
import sys,os
from scipy.misc import factorial as fac
import math
import AG_utils as AG

# Coordinate grid:
N = 64
y,x = np.indices((N,N)) - 32
rho = np.sqrt( (x*x) + (y*y)) /32
theta = np.arctan2(y,x)
mask = rho<=30

def zernike(m,n,rho,theta):
	"""
	Calculate Zernike polynomial (m, n) given a grid of radial
	coordinates rho and azimuthal coordinates theta.
	Adapted from Werkhoven, modified to the orthonormal basis from Noll 1976
	"""
	if (m > 0):
		#print zernike_rad(abs(m),n,rho)
		#print np.cos(abs(m)*theta)
		return np.sqrt(n+1)*zernike_rad(abs(m),n,rho)*np.sqrt(2)*np.cos(abs(m)*theta)
	if (m < 0):
		#print zernike_rad(abs(m),n,rho)
		#print np.sin(abs(m)*theta)
		return np.sqrt(n+1)*zernike_rad(abs(m),n,rho)*np.sqrt(2)*np.sin(abs(m)*theta)
	else:
		return np.sqrt(n+1)*zernike_rad(0, n, rho)

def wolf_zernike(m,n,rho,theta):
	if m%2:
		return None
	
def zernike_rad(m,n,rho):
	"""radial component of zernike (m,n), given radial coordinates rho"""
	wf = np.zeros(rho.shape) # grid of radial coords
	if (np.mod(n-m, 2) ==1):
		return rho*0.0
	for k in range((n-m)/2+1):
		wf += rho**(n-(2.0*k)) * ((-1.0)**k) * fac(n-k) / \
		( fac(k) * fac( (n+m)/2.0 - k ) * \
		fac( (n-m)/2.0 - k ) )
		return wf

def zernikel(i, rho, theta):
	"""find a way to do single indexing for zernikes
	   can be described by the zernike pyramid.
	http://www.opt.indiana.edu/VSG/Library/VSIA/VSIA-2000_taskforce/TOPS4_2.html"""
	# j = (n(n+2) + m ) / 2
	n = int(math.ceil((-3+np.sqrt(9+(8*i)))/2))
	m = int(2*i - n*(n+2))
	#print "m:", m, "n:", n
	return zernike(m, n, rho, theta)

def zernikemodel(vector, rho, theta):
	model = np.zeros((len(vector), rho.shape[0]))
	for q,vec in enumerate(vector):
		model[q,:] = zernikel(q, rho,theta)
		print "model slice size", model[q,:].shape
	print "total model shape",model.shape
	return model

def scipymodel(vector, rho, theta):
	model = np.zeros((rho.shape[0], rho.shape[1]))
	for q,vec in enumerate(vector):
		model[:,:] += vec*zernikel(q, rho,theta)
	return model

if __name__ == "__main__":

	support = AG.makedisk(rho.shape[0], R=32, array="EVEN")
	for q in range(20):
		print (q+5)//5, (q+5)%5
		ax=pl.subplot2grid((4,5),((q)//5,(q+5)%5))
		ax.set_title("mode "+str(q))
		test = support*zernikel(q,rho, theta)
		ax.imshow(test, interpolation='nearest')
		pl.axis('off')
	#pl.show()
	#sys.exit()

	# make a list of the first 25 terms
	zern_list = [zernikel(i, rho, theta)*mask for i in range(25)]
	print "zernike rad (m,n) indexed"
	print zernike_rad(1,1, rho)
	print "zernike (m,n) indexed"
	print zernike(3,3,rho, theta)
	print "zernike l indexed"
	print zernikel(9,rho, theta)

	print "zern_list shape", np.shape(zern_list)

	# try to make a test surface
	vec = np.random.random(25)
	surf = sum(val*zernikel(i, rho, theta) for (i,val) in enumerate(vec) )
	#surf = surf/surf.max()

	# cov matrix
	cov = np.array([ [(zerni*zernj*mask).sum() for zerni in zern_list]\
			for zernj in zern_list ])

	"""
	b is (#datapoints x 1) length
	x must be (param x 1)
	A is (#datapoints x param)

	A x = b
	(dpx25).(25x1) = dpx1
	(25xdp).(dpx25).(25x1) = (25xdp).(dpx1) = 25x1
	
	x = A-1 b, but A is not square
	
	b is the test suface I made
	A is the zernike components
	x is the coefficients I want to know.
	"""
	zmodel = zernikemodel(vec, rho, theta)
	flatdata = surf.reshape(surf.shape[0]*surf.shape[1])
	print "surf shape:", surf.shape
	flatmodel = np.zeros((flatdata.shape[0],len(zmodel)))
	print "flatmodel zeros:", flatmodel.shape
	print "flatdata shape:", flatdata.shape
	print "shape to equal:", zmodel.shape[1]*zmodel.shape[2]
	for q in range(flatmodel.shape[1]):
		flatmodel[:,q]= zmodel[q,:,:].reshape(zmodel.shape[1]*zmodel.shape[2])
	print "model created", flatmodel.shape
	modeltransp = flatmodel.transpose()
	print "transpose of A done..."
	print "transpose shape:", modeltransp.shape
	print "now taking the dot product of array size", modeltransp.shape, "and array size", flatmodel.shape
	At_b = np.dot(modeltransp, flatdata)
	At_A = np.dot(modeltransp, flatmodel)
	print 'At_b computed...'
	print 'At_b shape:', At_b.shape
	Ainv = np.linalg.inv(At_A)
	print 'Ainv computed...'
	print 'Ainv shape:', Ainv.shape
	x = np.dot(Ainv, At_b)
	print 'dot product projection computed...'
	residualvec = x-vec
	print "real vector:", vec
	print "calculated vector:", x
	print "residual vector:", residualvec
	residimg = np.dot(flatmodel, x) - flatdata
	residimg = residimg.reshape(rho.shape[0], rho.shape[1])
	## compare to pseudo inverse:
	print "==============NOW COMPARING TO PSEUDO INVERSE==============="
	Apinv = np.linalg.pinv(flatmodel)
	xp = np.dot(Apinv,flatdata)
	presvec = xp-vec
	print "calculated vector:", xp
	print "residual vector:", presvec
	precsurf = np.dot(flatmodel, x).reshape(rho.shape[0], rho.shape[1])
	presidual = precsurf - flatdata.reshape(rho.shape[0], rho.shape[1])

	recsurf = np.zeros((rho.shape[0], rho.shape[1]))
	for q in range(len(vec)):
		recsurf += x[q]*zmodel[q,:,:]

	print "==============NOW COMPARING TO SCIPY's==============="
	curve_fit()

	pl.figure()
	pl.imshow(surf, interpolation='nearest', cmap='bone')
	pl.colorbar()
	pl.figure()
	pl.imshow(recsurf, interpolation='nearest', cmap='bone')
	pl.colorbar()
	pl.figure()
	pl.imshow(residimg, interpolation='nearest', cmap='bone')
	pl.colorbar()

	pl.figure()
	pl.imshow(surf, interpolation='nearest', cmap='bone')
	pl.colorbar()
	pl.figure()
	pl.imshow(precsurf, interpolation='nearest', cmap='bone')
	pl.colorbar()
	pl.figure()
	pl.imshow(presidual, interpolation='nearest', cmap='bone')
	pl.colorbar()
	pl.show()

	sys.exit()
	
	#tempcov = np.zeros((4,4))
	#for i,zi in enumerate(zern_list[:4]):
	#	for j, zj in enumerate(zern_list[:4]):
	#		print "i:",i, zi.shape, zi.sum()
	#		print "j:",j, zj.shape, zi.sum()
	#		print "product ij:",zi*zj
	#		print "normed sum of product:", np.sum(zi*zj) /(np.sum(zi)*np.sum(zj))
	#		tempcov[i,j] = np.sum(zi*zj)/(np.sum(zi)*np.sum(zj))
	#pl.imshow(tempcov, interpolation='nearest', cmap='bone')
	#pl.show()
	# Our old friend the pseudo inverse
	cov_inv = np.linalg.pinv(cov)

	# inner product:
	innerprod = np.array([np.sum(surf*zerni) for zerni in zern_list])

	# solve for vector coefficients:
	solve = np.dot(cov_inv, innerprod)

	# Reconstruct surface
	recon = sum(val*zernikel(i,rho, theta) for (i, val) in enumerate(solve))

	# And check the solution, e.g. residual
	# np.allclose returns True if two arrays are element-wise equal 
	# within a tolerance given by keyword rtol, default 1.0e-5.
	print "residual:", vec - solve
	print "original vector:", vec
	print "solution:",solve
	print "cov", cov
	pl.imshow(cov, interpolation="nearest", cmap ='bone')
	pl.colorbar()
	pl.show()
	print "Do the vectors match?", np.allclose(vec, solve)
	print "Do the surfaces match?", np.allclose(surf, recon)
	pl.figure()
	pl.imshow(surf*mask, interpolation="nearest", cmap ='bone')
	pl.figure()
	pl.imshow((surf-recon)*mask,interpolation="nearest", cmap ='bone')
	pl.colorbar()
	pl.figure()
	pl.imshow(zern_list[4], interpolation="nearest", cmap ='bone')
	pl.show()
