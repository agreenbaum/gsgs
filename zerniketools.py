### Libraries

import sys
import astropy.io.fits as pyfits
import numpy as N
from scipy.misc import factorial as fac
from poppy import zernike as z

### Init functions
def zernike_rad(m, n, rho):
	"""
	Calculate the radial component of Zernike polynomial (m, n) 
	given a grid of radial coordinates rho.
	
	>>> zernike_rad(3, 3, 0.333)
	0.036926037000000009
	>>> zernike_rad(1, 3, 0.333)
	-0.55522188900000002
	>>> zernike_rad(3, 5, 0.12345)
	-0.007382104685237683
	"""
	
	if (n < 0 or m < 0 or abs(m) > n):
		raise ValueError
	
	if ((n-m) % 2):
		return rho*0.0
	
	pre_fac = lambda k: (-1.0)**k * fac(n-k) / ( fac(k) * fac( (n+m)/2.0 - k ) * fac( (n-m)/2.0 - k ) )
	
	return sum(pre_fac(k) * rho**(n-2.0*k) for k in xrange((n-m)/2+1))

def zernike(m, n, rho, phi):
	"""
	Calculate Zernike polynomial (m, n) given a grid of radial
	coordinates rho and azimuthal coordinates phi.
	
	>>> zernike(3,5, 0.12345, 1.0)
	0.0073082282475042991
	>>> zernike(1, 3, 0.333, 5.0)
	-0.15749545445076085
	"""
	if (m > 0): return zernike_rad(m, n, rho) * N.cos(m * phi)
	if (m < 0): return zernike_rad(-m, n, rho) * N.sin(-m * phi)
	return zernike_rad(0, n, rho)

def zernikel(j, rho, phi):
	"""
	Calculate Zernike polynomial with Noll coordinate j given a grid of radial
	coordinates rho and azimuthal coordinates phi.
	
	>>> zernikel(0, 0.12345, 0.231)
	1.0
	>>> zernikel(1, 0.12345, 0.231)
	0.028264010304937772
	>>> zernikel(6, 0.12345, 0.231)
	0.0012019069816780774
	"""
	n = 0
	while (j > n):
		n += 1
		j -= n
	
	m = -n+2*j
	return zernike(m, n, rho, phi)



class ZernikeFitter:
	"""
	Does Zernikes on a circular disk fitting just inside your array of size narr,
	so if your pupil is undersized within the pupil array, snip off padding before
	sending its wavefront into this object be fit.

	Usage:

		import ZernikeFitter as ZF
		zf = ZF.ZernikeFitter(nzern=10, narr=200) # sqaure array containing the disk (D)
		zcoeffs, fittedsurface, residualsurface = zf.fit_zernikes_to_surface(yoursurface)


	Zernikes break naturally at nzern = 0 1 3 6 10 15 21 28 36 45 55 66 78 91 105 ...   n*(n+1)/2
	N.B. These are Noll-numbered Zernikes     anand@stsci.edu 2015
	"""

	def __init__(self, nzern=15, narr=200, extrasupportindex=None):
		"""
		Input: nzern: number of Noll Zernikes to use in the fit
		Input: narr: the live pupil array size you want to use
		
		Sets up list of poly's and grids & support grids
		Makes coordinate grid for rho and phi and circular support mask
		Calculates 'overlap integrals' (covariance matrix) of the Zernike polynomials on your grid and array size
		Calculates the inverse of this matrix, so it's 'ready to fit' your incoming array
		
		"""
		self.narr = narr
		self.nzern = nzern  # tbd - allowed numbers from Pascal's Triangle sum(n) starting from n=1, viz. n(n+1)/2
		self.grid = (N.indices((self.narr, self.narr), dtype=N.float) - self.narr//2) / (float(self.narr)*0.5) 
		self.grid_rho = (self.grid**2.0).sum(0)**0.5
		self.grid_phi = N.arctan2(self.grid[0], self.grid[1])
		self.grid_mask = self.grid_rho <= 1
		self.grid_outside = self.grid_rho > 1

		if extrasupportindex is not None:
			self.grid_mask[extrasupportindex] = 0
			self.grid_outside[extrasupportindex] = 1
		else:
			pass
		# Compute list of explicit Zernike polynomials and keep them around for fitting
		self.zern_list = [zernikel(i, self.grid_rho, self.grid_phi)*self.grid_mask for i in xrange(self.nzern)]

		# Calculate covariance between all Zernike polynomials
		self.cov_mat = N.array([[N.sum(zerni * zernj) for zerni in self.zern_list] for zernj in self.zern_list])
		# Invert covariance matrix using SVD
		self.cov_mat_in = N.linalg.pinv(self.cov_mat)


	def fit_zernikes_to_surface(self, surface):
		"""
		Input: surface: input surface to be fit (2D array)
		Output: zcoeffs: 1d vector of coefficients of the fit (self.nzern in length)
		Output: rec_wf: the 'recovered wavefront' - i.e. the fitted zernikes, in same array size as surface
		Output: res_wf: surface - rec_wf, i.e. the residual error in the fit

		"""

		# Calculate the inner product of each Zernike mode with the test surface
		wf_zern_inprod = N.array([N.sum(surface * zerni) for zerni in self.zern_list])

		# Given the inner product vector of the test wavefront with Zernike basis,
		# calculate the Zernike polynomial coefficients
		zcoeffs = N.dot(self.cov_mat_in, wf_zern_inprod)
		print( "First 10 recovered Zernike coeffts:", zcoeffs[:10])
	
		# Reconstruct (e.g. wavefront) surface from Zernike components
		rec_wf = sum(val * zernikel(i, self.grid_rho, self.grid_phi) for (i, val) in enumerate(zcoeffs))
		rec_wf = rec_wf*self.grid_mask

		print( "Standard deviation of fit is %.3e" % (surface*self.grid_mask - rec_wf)[self.grid_mask].std())
		return zcoeffs, rec_wf, (surface - rec_wf)*self.grid_mask


class HexikeFitter:
	"""
	Does Zernikes on a circular disk fitting just inside your array of size narr,
	so if your pupil is undersized within the pupil array, snip off padding before
	sending its wavefront into this object be fit.

	Usage:

		import ZernikeFitter as ZF
		zf = ZF.ZernikeFitter(nzern=10, narr=200) # sqaure array containing the disk (D)
		zcoeffs, fittedsurface, residualsurface = zf.fit_zernikes_to_surface(yoursurface)


	Zernikes break naturally at nzern = 0 1 3 6 10 15 21 28 36 45 55 66 78 91 105 ...   n*(n+1)/2
	N.B. These are Noll-numbered Zernikes     anand@stsci.edu 2015
	"""

	def __init__(self, nhex=15, narr=200, extrasupportindex=None):
		"""
		Input: nzern: number of Noll Zernikes to use in the fit
		Input: narr: the live pupil array size you want to use
		
		Sets up list of poly's and grids & support grids
		Makes coordinate grid for rho and phi and circular support mask
		Calculates 'overlap integrals' (covariance matrix) of the Zernike polynomials on your grid and array size
		Calculates the inverse of this matrix, so it's 'ready to fit' your incoming array
		
		"""
		self.narr = narr
		self.nhex = nhex  # tbd - allowed numbers from Pascal's Triangle sum(n) starting from n=1, viz. n(n+1)/2
		self.grid = (N.indices((self.narr, self.narr), dtype=N.float) - self.narr//2) / (float(self.narr)*0.5) 
		self.grid_rho = (self.grid**2.0).sum(0)**0.5
		self.grid_phi = N.arctan2(self.grid[0], self.grid[1])
		self.grid_mask = self.grid_rho <= 1
		self.grid_outside = self.grid_rho > 1

		if extrasupportindex is not None:
			self.grid_mask[extrasupportindex] = 0
			self.grid_outside[extrasupportindex] = 1

		# Compute list of explicit Zernike polynomials and keep them around for fitting
		self.hex_list = z.hexike_basis(nterms = self.nhex, npix=self.narr)


		self.hex_list = z.hexike_basis(nterms = self.nhex, npix=self.narr)

		# Force hexikes to be unit standard deviation over hex mask
		for h, hfunc in enumerate(self.hex_list):
		    if h>0: self.hex_list[h] = (hfunc/hfunc[self.grid_mask].std()) * self.grid_mask
		    else: self.hex_list[0] = hfunc * self.grid_mask

		#### Write out a cube of the hexikes in here
		if 0:
			self.stack = N.zeros((self.nhex, self.narr, self.narr))
			for ii in range(len(self.hex_list)):
				self.stack[ii,:,:] = self.hex_list[ii]
		# Calculate covariance between all Zernike polynomials
		self.cov_mat = N.array([[N.sum(hexi * hexj) for hexi in self.hex_list] for hexj in self.hex_list])
		self.grid_mask = z.hex_aperture(npix=self.narr)==1
		self.grid_outside = self.grid_mask == False
		# Invert covariance matrix using SVD
		self.cov_mat_in = N.linalg.pinv(self.cov_mat)


	def fit_hexikes_to_surface(self, surface, choosemodes=False):
		"""
		Input: surface: input surface to be fit (2D array)
		Output: zcoeffs: 1d vector of coefficients of the fit (self.nzern in length)
		Output: rec_wf: the 'recovered wavefront' - i.e. the fitted zernikes, in same array size as surface
		Output: res_wf: surface - rec_wf, i.e. the residual error in the fit

		"""

		# Calculate the inner product of each Zernike mode with the test surface
		wf_hex_inprod = N.array([N.sum(surface * hexi) for hexi in self.hex_list])

		# Given the inner product vector of the test wavefront with Zernike basis,
		# calculate the Zernike polynomial coefficients
		hcoeffs = N.dot(self.cov_mat_in, wf_hex_inprod)
	
		# Reconstruct (e.g. wavefront) surface from Zernike components
		hexikes =  z.hexike_basis(nterms=len(hcoeffs), npix=self.narr)
		if choosemodes is not False:
			hcoeffs[choosemodes==0] = 0
		rec_wf = sum(val * hexikes[i] for (i, val) in enumerate(hcoeffs))

		if 0:
			print( "First 10 recovered Hernike coeffts:", hcoeffs[:10])
			print( "Standard deviation of fit is %.3e" % (surface*self.grid_mask - rec_wf)[self.grid_mask].std())
		return hcoeffs, rec_wf, (surface - rec_wf)*self.grid_mask


