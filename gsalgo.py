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
from scipy import signal
import scipy.ndimage.filters as imfilter
from simtools import makedisk
from ZernikeFitter import ZernikeFitter
from HexikeFitter import HexikeFitter

def makedisk(N, R, ctr=(0,0), array="ODD"):
	if array == "ODD":
		M = (N-1)/2
		xx = np.linspace(-M-ctr[0],M-ctr[0],N)
		yy = np.linspace(-M-ctr[1],M-ctr[1],N)
	if array == "EVEN":
		M = N/2
		xx = np.linspace(-M-ctr[0],M-ctr[0]-1,N)
		yy = np.linspace(-M-ctr[1],M-ctr[1]-1,N)
	(x,y) = np.meshgrid(xx, yy.T)
	r = np.sqrt((x**2)+(y**2))
	array = np.zeros((N,N))
	array[r<R] = 1
	return array


def simulate_zern(fov = 80, mode=5, zcoeffs=None,livepupilrad=None, mask=None):
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
	if zcoeffs is not None:
		aberr = np.zeros((rho.shape))
		for md in range(len(zcoeffs)):
			aberr+=zcoeffs[md]*zern.zernikel(md, rho, theta)
	elif hasattr(mode, "__iter__"):
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
	aberr = aberr - np.mean(aberr[rho<=1])
	mask_aberr = mask * aberr
	# print out rms of this aberration over circular pupil?  - AS
	print np.var(aberr[rho<1])
	return mask_aberr, aberr

def avg_piston(aberr, positions, mask, R = 20, point=False, error=None):
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

		aberrval = aberr[holemask==1].sum() / len(aberr[holemask==1])
		# make the fringe phases noisy or not:
		if error is not None:
			aberrval = np.random.normal(aberrval, error*abs(aberrval))
			#noise = np.random.normal(scale=error*abs(aberrval))
		else:
			noise = 0.0

		if not point:
			#avgphase[holemask==1] = aberr[holemask==1].sum() / len(aberr[holemask==1])
			avgphase[holemask==1] = aberrval
		else:
			#int(np.sum(holemask*x)/holemask.sum()),int(np.sum(holemask*y)/holemask.sum())
			avgphase[int(np.sum(holemask*x)/holemask.sum()),\
			 int(np.sum(holemask*y)/holemask.sum())] = aberrval #+ noise#\
			#		aberr[holemask==1].sum() / len(aberr[holemask==1]) 
		
	new_support[abs(avgphase)>0] = 1
	return avgphase, new_support

def fit_zernikes_to_surface(zern_list_, surface, grid_rho_, grid_phi_):
	"""
	Also from Tim van Werkhoven's zernike code
	"""

	# Calculate covariance between all Zernike polynomials
	cov_mat = np.array([[np.sum(zerni * zernj) for zerni in zern_list_] for zernj in zern_list_])
	# Invert covariance matrix using SVD
	cov_mat_in = np.linalg.pinv(cov_mat)
	# Calculate the inner product of each Zernike mode with the test surface
	wf_zern_inprod = np.array([np.sum(surface * zerni) for zerni in zern_list_])

	# Given the inner product vector of the test wavefront with Zernike basis,
	# calculate the Zernike polynomial coefficients
	rec_wf_pow_ = np.dot(cov_mat_in, wf_zern_inprod)
	print "recov Z coeffts:\n", rec_wf_pow_
	
	# Reconstruct surface from Zernike components
	rec_wf_ = sum(val * zern.zernikel(i, grid_rho_, grid_phi_) for (i, val) in enumerate(rec_wf_pow_))
	
	return rec_wf_pow_, rec_wf_

class NRM_GS:
	def __init__(self, dpsf, pupsupport, pha_constraint, c_support, segmentwhich=None, god=None):
		"""
		requires both full pupil and nrm data and pupil supports for each
		which array needs to be specified for segmented telescope
		"""
		self.dpsf = dpsf
		self.pupsupport = pupsupport
		# The "constraint" replaces values inside support of pupil phase 
		# each iteration
		self.constraint = pha_constraint
		self.c_support = c_support # tells where to replace phase values
		self.npix_img = self.dpsf.shape[0]
		self.npix_pup = self.pupsupport.shape[0]
		# When should the loop stop?
		self.orig_aberr=None
		# set fractional change in phase between iterations to stop
		self.condition = 1.0e-5 #default condition, user can manually change
		self.condition_c = 1.0e-2 #default condition, user can manually change
		self.metriclist = []
		self.errorlist = []
		self.nitermax_c = 100
		self.nitermax = 500
		self.zsmooth=True
		self.segmentwhich = segmentwhich

		if god is not None:
			# give it the true phase to constrain
			self.god=god
		self.damping=False

		if segmentwhich is None:
			side = self.pupsupport.shape[0]
			radsupport = self.pupsupport[:,side/2].sum() // 2
			self.residmask = makedisk(side, radsupport-2,array="EVEN")
		else:
			self.residmask = self.pupsupport

	def find_wavefront(self):
		self.i = 0
		# Give it a value so to be true on first iteration
		self.metric = 1
		if hasattr(self, "fitscubename"):
			self.fitscube = np.zeros((10, self.pupsupport.shape[0],\
						self.pupsupport.shape[1]*2))
		self.make_iter_step(self.pupsupport)
		# for testing purposes, set the number of iterations, check it every so often
		while abs(self.metric) > self.condition_c:
		#for ii in range(100):
			self.make_iter_step(self.pupil_i)
			print "Iteration no. ", self.i
			print "Value of difference metric:", self.metric
			if (self.i==105 & hasattr(self, "fitscubename")):
				fits.PrimaryHDU(data=self.fitscube).writeto(self.fitscubename,\
						clobber=True)
			if self.i > 100:
				print "Reached max iterations"
				break

		# Now lift the constraint to finish off
		self.c_support = np.zeros((self.c_support.shape))
		self.c_conv_i = self.i
		while abs(self.metric) > self.condition:
			print "Iteration no. ", self.i
			if (hasattr(self, "fitscubename") and self.i==105):
				fits.PrimaryHDU(data=self.fitscube).writeto(self.fitscubename,\
					clobber=True)
			self.make_iter_step(self.pupil_i)
			print "Iteration max:", self.nitermax
			print "Max reached?", self.nitermax - self.i, "iterations left"
			print "Value of difference metric:", self.metric
			if self.i==self.nitermax:
				print "REACHED 500 ITERATIONS, THIS SHOULD STOP"
			if self.i > self.nitermax:
				print "Reached max iterations"
				break

		return self.pupil_i

	def make_iter_step(self, pupil):
		"""
		As described at top of this file
		signal in each plane is complex
		"""
		self.i += 1
		self.pup_in = pupil
		# constrain
		#self.pup_i[abs(self.constraint)>0] = 0
		self.puppha_i = np.angle(self.pup_in)
		if self.c_support.sum()>0:
			print "using phase constraint"
			self.puppha_i[abs(self.c_support)>0] =0
			self.puppha_i = self.puppha_i + self.constraint
		else:
			print "no phase constraint specified"
		if (self.i>1 & self.damping==True):
			damp = 0.2
			print "damping correction:", damp
			# 0.1 damping correction
			phaseupdate = (1-damp)*(self.puppha_i) + (damp)*(np.angle(self.pupil_i))
			self.pup_i = abs(self.pup_in)*(np.exp(1j*phaseupdate)*self.pupsupport)
		else:
			self.pup_i = abs(self.pup_in)*np.exp(1j*(self.puppha_i))*self.pupsupport

		if hasattr(self, "fitscubename"):
			# writes a cube every 10 iterations up to i=100, useful for checking progress
			# this half of the array is the (i-1)th pupil
			if ((np.mod((self.i), 10)<1) & (self.i < 105)):
				self.fitscube[(self.i-1)//10, :,:self.pupsupport.shape[0]] =\
					 np.angle(self.pup_i)*self.pupsupport

		self.field_i = mft(self.pup_i, self.nlamD, self.npix_img)
		# swap out a (electric field magnitude) with root(dpsf)
		self.amp = np.sqrt(self.dpsf)
		# use phase psi calculated
		self.psi_i = np.angle(self.field_i)
		# The corrected field
		self.fieldc_i = self.amp*np.exp(1j*self.psi_i)
		#self.pupil_i = mft.perform(self.fieldc_i, self.npix/4, self.npix, inverse=True)
		self.pupil_i = mft(self.fieldc_i, self.nlamD, self.npix_pup, inverse=True)*self.pupsupport
		# subtract the mean from the undersized pupil
		zmeanphase = np.angle(self.pupil_i) - \
				np.mean(np.angle(self.pupil_i)[self.pupsupport==1])
		#		np.mean(np.angle(self.pupil_i)[self.residmask==1])
		self.pupil_i = abs(self.pupil_i)*np.exp(1j*zmeanphase)

		if self.zsmooth==True:
			# smooth each iterations by fitting zernikes, gets rid of unphysical
			# high frequency artifacts
			self.fit_zernikes()
			self.pupil_i = abs(self.pupil_i)*np.exp(1j*self.full_rec_wf)

		self.currentpsf = mft(self.pupil_i, self.nlamD, self.npix_img)

			pupilpha = imfilter.uniform_filter(paddedpha)#, nsig)
			self.pupil_i = abs(self.pupil_i)*np.exp(1j*pupilpha*self.pupsupport)*self.pupsupport
		self.metriclist.append(self.compute_metric())

		if hasattr(self,"god"):
			# self.god defined (as the input aberration), this will also calculate the true error 
			self.errorlist.append(self.truerror)

		if hasattr(self, "fitscubename"):
			# As about writes a cube every 10 iterations up to i=100
			# this half of the array is the ith pupil
			if ((np.mod((self.i), 10)<1) & (self.i < 105)):
				self.fitscube[(self.i-1)//10, :, self.pupsupport.shape[0]:] =\
					 np.angle(self.pupil_i)*self.pupsupport

		return self.pupil_i

	def compute_metric(self):
		''' what is the difference between my original PSF and the current iteration? '''

		self.residual = ((self.puppha_i*self.residmask) \
				- (np.angle(self.pupil_i)*self.residmask)) \
				/ (self.puppha_i*self.residmask).sum()
		self.residual *=self.residmask

		self.metric = np.std((self.puppha_i[self.residmask==1] - \
			      np.angle(self.pupil_i)[self.residmask==1]))\
		if hasattr(self, "god"):
			# self.god is the true input aberration. If defined this will compute the 
			# True residual
			# make a slightly undersized support to compute an appropriate residual
			# (that neglects the really big errors only on the edge)

			self.truresid = (self.god*self.residmask - np.angle(self.pupil_i)*self.residmask)
			self.truresid = (self.truresid - np.mean(self.truresid[self.residmask==1]))*\
					self.residmask

			# overwrite the metric here.
			self.truerror = np.sqrt(np.var(self.truresid[self.residmask==1]))
			#print self.truerror

		return self.metric

	def fit_zernikes(self, nmodes=15):
		# Projects measured pupil phase onto set of nmodes zernike terms
		# default is 15
		hexsz = 116 # rough size of a hex segment in out JWST segment which array
		self.full_rec_wf = np.zeros(self.pupsupport.shape) # This will be the new pupil phase
		print self.segmentwhich
		if self.segmentwhich is not None:
			# If working with a segmented mirror, need a fits file labeling each mirror segement
			for seg in range(int(self.segmentwhich.max())):

				segmask = np.ones(self.segmentwhich.shape)*(self.segmentwhich==seg+1)
				segtrue = np.where(self.segmentwhich==seg+1)

				# This is a lot of silly stuff to isolate each segment,
				# it's probably written very poorly.
				# pick out the min/max position of the segment

				lox, hix = min(segtrue[0]), max(segtrue[0])
				loy, hiy = min(segtrue[1]), max(segtrue[1])
				xsize,ysize = hix-lox, hiy-loy
				padx = (hexsz-xsize)// 2 
				pady = (hexsz-ysize)// 2

				"""
				print lox, hix, loy, hiy
				print "total padding:",zernboxsize - xsize, zernboxsize-ysize
				print "hex sizes", xsize, ysize
				print "padding", padx, pady
				print lox-padx,lox+xsize+padx, loy-pady,loy+ysize+pady
				"""

				hexmask = segmask[lox-padx:lox+xsize+padx + (hexsz-xsize)%2,\
						  loy-pady:loy+ysize+pady+(hexsz-ysize)%2]
				hf = HexikeFitter(nhex = nmodes, narr = hexsz, extrasupportindex = (hexmask==0))

				#12/17/2015: added as listed below in Zernike version
				#hf.grid_mask = hexmask

				fullpupphase = (self.pupsupport*np.angle(self.pupil_i))
				pupseg = fullpupphase[lox-padx:lox+xsize+padx + (hexsz-xsize)%2,\
					 loy-pady:loy+ysize+pady + (hexsz-ysize)%2]*hexmask

				self.coeffs, self.rec_wf, self.hresid = hf.fit_hexikes_to_surface(pupseg)
				self.full_rec_wf[segmask==1] = self.rec_wf[hexmask==1]

			return self.coeffs


		else:
			# Edit, Dec 17, 2015:
			# Greater of x/y axis diameter. There must be a better way though.
			indarry, indarrx = np.indices(self.pupsupport.shape)
			indarrx[self.pupsupport==0] = 0
			indarry[self.pupsupport==0] = 0
			self.livepupilD = np.array([(indarrx.max()-indarrx[indarrx>0].min()),\
								indarry.max()-indarry[indarry>0].min()]).max() +1
			# commenting this out --- if you have any pupil obstructions this doesn't work!
			#self.livepupilD = np.array([self.pupsupport.sum(axis=1), self.pupsupport.sum(axis=0)]).max()
			print self.livepupilD, "size for live pupil support"
			fov = self.pupil_i.shape[0]
			print "size of full pupil"
			print self.full_rec_wf.shape
			sidesz= (fov - self.livepupilD) / 2

			# crop down to just the pupil
			croppedpupphases = (self.pupsupport*np.angle(self.pupil_i))[sidesz:-sidesz, \
										    sidesz:-sidesz]
			# initialize Zernike fitter
			zf = ZernikeFitter(nzern=nmodes, narr = croppedpupphases.shape[0],
					extrasupportindex=(self.pupsupport[sidesz:-sidesz,sidesz:-sidesz]==0))
			# cut out everything outside the live pupil

			# 12/17/15: But sometimes you have pupil obstructions and that's kind of important
			# so I want to update the grid_mask attribute so that I don't try to fit 
			# the part of the pupil that is zero due to the support
			# so I'm adding the following line to care of any discontinuities in the pupil.
			#zf.grid_mask[self.pupsupport[sidesz:-sidesz,sidesz:-sidesz]==0] = 0
			#zf.grid_outside[self.pupsupport[sidesz:-sidesz,sidesz:-sidesz]==0] = 1

			self.zcoeffs, self.rec_wf, self.zresid = \
				zf.fit_zernikes_to_surface(croppedpupphases)

			# some fits files for debugging
			"""
			fits.PrimaryHDU(data=self.pupsupport*\
				np.angle(self.pupil_i)).writeto("zf_pupilin.fits", clobber=True)
			fits.PrimaryHDU(data=self.zresid).writeto("zf_residual.fits", clobber=True)
			fits.PrimaryHDU(data=croppedpupphases).writeto("zf_croppedpupil.fits",\
									clobber=True)
			fits.PrimaryHDU(data=self.rec_wf*zf.grid_mask).writeto("zf_rec_wf.fits",\
										clobber=True)
			"""

			# Put it back into the big array
			"""
			# More fits files for debugging
			fits.PrimaryHDU(data=self.full_rec_wf).writeto("zf_full_rec_wf.fits", clobber=True)
			fits.PrimaryHDU(data=zf.grid_mask*\
				np.ones(self.rec_wf.shape)).writeto("zf_gridmask.fits", clobber=True)
			print "==========================="
			print "recovered zernike coefficients"
			print self.zcoeffs
			sys.exit()
			"""
			self.full_rec_wf[sidesz:-sidesz, sidesz:-sidesz] = zf.grid_mask*self.rec_wf
			return self.zcoeffs

def test_simple_GS(psf, pupilsupport, aberr):
	""" First test normal gerchberg-saxton without noise to see if things 
	    are going the right way
	"""
	pupil = pupilsupport.copy()
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
	plt.imshow(aberr)
	plt.colorbar()
	plt.figure()
	plt.title("simple GS pupil iteration # {0}".format(ii+1))
	plt.imshow(np.angle(pupil))
	plt.colorbar()
	plt.show()

def test_constrained_GS():
	nrm_support = fits.open(GS+"test_nrm_support.fits")[0].data
	nrm_aberr, aberr = simulate_zern(fov=nrm_support.shape[0], mask = nrm_support)

	# provide rough hole positions by eye
	pos_byeye = [[147,255], [179,289], [222, 274], [276,218], [285,174], [220,113], [134,139]]
	avgpist = avg_piston(nrm_aberr, pos_byeye, nrm_support)

	#fullpupfgeom = geom(D = 6, ctrs = [(0,0),])
	fullpupmask = makedisk(nrm_support.shape[0], R = 128, array='EVEN')
	fullpupaberr = fullpupmask*np.exp(1j*aberr)
	npix=400
	fullE = mft(fullpupaberr, npix/2,npix)
	fullpsf = abs(fullE)**2

	import matplotlib.pyplot as plt
	plt.figure()
	plt.imshow(np.sqrt(fullpsf))

	plt.show()

if __name__ == "__main__":
	# Define our "known" phi_i's that we would measure with NRM
	# state phi_i UNITS - AS
	#nrm_support = fits.open(GS+"JWST_g7s6_polyNoah6_outerrad14_mag1_array400_rotdeg14.75.fits")[0].data
	nrm_support = fits.open(GS+"test_nrm_support.fits")[0].data
	nrm_aberr, aberr = simulate_zern(fov=nrm_support.shape[0], mask = nrm_support)

	# provide rough hole positions by eye
	pos_byeye = [[147,255], [179,289], [222, 274], [276,218], [285,174], [220,113], [134,139]]
	avgpist = avg_piston(nrm_aberr, pos_byeye, nrm_support)

	#fullpupfgeom = geom(D = 6, ctrs = [(0,0),])
	fullpupmask = makedisk(nrm_support.shape[0], R = 128, array='EVEN')
	fullpupaberr = fullpupmask*np.exp(1j*aberr)
	npix=400
	fullE = mft(fullpupaberr, npix/2,npix)
	fullpsf = abs(fullE)**2
	plt.figure()
	plt.imshow(np.sqrt(fullpsf))

	"""
	test = mft(fullpsf, 50, 400)
	plt.figure()
	plt.title("transform amplitude")
	plt.imshow(test.real)
	plt.figure()
	plt.title("transform phase")
	plt.imshow(np.angle(test))
	plt.show()
	"""
	test_simple_GS(fullpsf, fullpupmask, aberr*fullpupmask)
	
	"""
	gs = NRM_GS(fullpsf, fullpupmask, avgpist, nrm_support)
	gs.orig_aberr = aberr
	gs.find_wavefront()
	"""
