#! /usr/bin/env python

import numpy as np
import sys,os
import zerniketools as zern


def makedisk(N, R, ctr=(0,0)):
	if N%2 == 1:
		M = (N-1)/2
		xx = np.linspace(-M-ctr[0],M-ctr[0],N)
		yy = np.linspace(-M-ctr[1],M-ctr[1],N)
	if N%2 == 0:
		M = N/2
		xx = np.linspace(-M-ctr[0],M-ctr[0]-1,N)
		yy = np.linspace(-M-ctr[1],M-ctr[1]-1,N)
	(x,y) = np.meshgrid(xx, yy.T)
	r = np.sqrt((x**2)+(y**2))
	array = np.zeros((N,N))
	array[r<R] = 1
	return array


def simulate_zern(fov = 80, mode=5, livepupilrad=None, mask=None):
	"""
	Default Astig_1 0.1 radians
	"""
	# Define our coordinates for wavefront aberration of choice
	y,x = np.indices((fov,fov)) - fov/2
	if livepupilrad is not None:
		rho = np.sqrt( (x*x) + (y*y) ) / livepupilrad
	else:
		rho = np.sqrt( (x*x) + (y*y) ) / (fov/2) 
	theta = np.arctan2(y,x)
	if hasattr(mode, "__iter__"):
		aberr = np.zeros(rho.shape)
		frac = 1/len(mode)
		for md in mode:
			aberr += zern.zernikel(md,rho,theta)
	else:
		aberr = zern.zernikel(mode, rho, theta)
	if mask is not None:
		mask = mask
	else:
		mask = np.ones((fov,fov))
	mask_aberr = mask * aberr
	# print out rms of this aberration over circular pupil?  - AS
	print np.var(aberr[rho<1])
	return mask_aberr, aberr

def avg_piston(aberr, positions, mask, R = 20, point=False):
	"""
	aberr is the full square array aberration

	positions are approximaete coordinates of mask holes recorded by eye

	mask is the NRM or other pupil mask for over which to compute average phases

	R - radius to enclose mask holes around coordinate points (also by eye for now)
	    purpose to choose holes one by one. 

	point -- should this return average phase values in a single pixel (TRUE)
		 or over the whole mask support (FALSE). Recommended to remain FALSE.
	"""
	fov = aberr.shape[0]
	avgphase = np.zeros(aberr.shape) #radians
	holemask = avgphase.copy()
	x,y = np.indices(holemask.shape)
	debug = avgphase.copy()
	new_support = avgphase.copy()
	for ii,pos in enumerate(positions):
		holemask = makedisk(fov, R, ctr = (pos[0] - fov/2, pos[1] - fov/2))
		debug+=holemask
		holemask[abs(holemask*mask)>0] = 1 
		holemask[abs(holemask*mask)==0] = 0
		if not point:
			avgphase[holemask==1] = aberr[holemask==1].sum() / len(aberr[holemask==1])
		else:
			#int(np.sum(holemask*x)/holemask.sum()),int(np.sum(holemask*y)/holemask.sum())
			avgphase[int(np.sum(holemask*x)/holemask.sum()),\
			 int(np.sum(holemask*y)/holemask.sum())] = \
					aberr[holemask==1].sum() / len(aberr[holemask==1])
		#plt.figure()
		#plt.imshow(avgphase)
		#plt.figure()
		#plt.imshow(debug)
		#plt.show()
	new_support[abs(avgphase)>0] = 1
	return avgphase, new_support
