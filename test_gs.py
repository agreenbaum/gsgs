#! /usr/bin/env python

"""
by A. Greenbaum agreenba@pha.jhu.edu 2015

Using the NRM phases to feed into a GS phase retrieval algorithm.

Will need
-------------
P : pupil support file -- self generated, what is our pupil?

phi_0 : starting phases (0's, e.g.)

Psi_i's : 7 NRM-derived phases, zero mean

dpsf : data psf -- cleaned up lab exposures

Procedure
-------------
P,phi_0 ---FT---> a,psi replace a with sqrt(dpsf), sqrt(dpsf),psi ---iFT---> P', phi' replace P' with P
phi' with NRM phases only in nrm support --- repeat
"""
HOME = "/Users/agreenba/"
GS = HOME+"Dropbox/AlexNRM/makidonlab/GSreduction/"

import sys,os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import zernike_AG as zern
from matrixDFT import matrix_dft as mft
from simtools import *

nrm_support = fits.open("test_nrm_support.fits")[0].data
nrm_aberr, aberr = simulate_zern(fov=nrm_support.shape[0], mask = nrm_support)

# provide rough hole positions by eye
pos_byeye = [[147,255], [179,289], [222, 274], [276,218], [285,174], [220,113], [134,139]]
avgpist = avg_piston(nrm_aberr, pos_byeye, nrm_support)

fullpupmask = makedisk(nrm_support.shape[0], R = 128, array='EVEN')
fullpupaberr = fullpupmask*np.exp(1j*aberr)
npix=400
fullE = mft(fullpupaberr, npix/2,npix)
fullpsf = abs(fullE)**2

def test_simple_GS(psf, pupilsupport, aberr):
	""" First test normal gerchberg-saxton without noise to see if things 
	    are going the right way
	"""
	psf = fullpsf.copy()
	pupil = fullpupmask.copy() # pupil support shape
	masked_aberr = aberr*fullpupmask
	plt.figure()
	plt.title("Pupil Support")
	plt.imshow(pupilsupport)
	plt.show()
	npix = pupil.shape[0]
	u = npix/2
	for ii in range(20):
		a = mft(pupil, u, npix)
		amp = np.sqrt(psf)
		pha = np.angle(a)
		pupil = mft(amp*np.exp(1j*pha), u, npix, inverse=True)*pupilsupport
		# Just to make sure abs() and np.angle() are working as we expect
		# When I use the amplitude of pupil support here my GS loop
		# converges on the wrong answer -- some weird quad-symmetric thing
		# need to use abs(pupil) even if it's not even illumination
		# just constrain the support by multiplying at the end
		pupil = abs(pupil)*np.exp(1j*np.angle(pupil))*pupilsupport
		#if ii%15 == 0:
			#plt.figure()
			#plt.title("simple GS pupil iteration # {0}".format(ii+1))
			#plt.imshow(np.angle(pupil))
			#plt.colorbar()
			#plt.show()
	plt.figure()
	plt.title("original aberration")
	plt.imshow(masked_aberr)
	plt.colorbar()
	plt.figure()
	plt.title("simple GS pupil iteration # {0}".format(ii+1))
	plt.imshow(np.angle(pupil))
	plt.colorbar()
	plt.show()

def test_constrained_GS():
	
	gs = NRM_GS(fullpsf, fullpupmask, avgpist, nrm_support)
	gs.orig_aberr = aberr
	gs.find_wavefront()

if __name__ == "__main__":

	test_simple_GS()
	test_constrained_GS()

