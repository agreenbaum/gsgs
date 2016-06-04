"""
A driver to test the asymmetric pupil wavefront sensor with constrained GS phase retrieval
"""

import numpy as np
from gsalgo import *
import numpy.random as rand
import matplotlib.pyplot as plt
from AG_utils import rebin
from scipy.misc import comb
from astropy.io import fits

def rad2mas(rad):
	mas = rad*(3600*180/np.pi) / (10**(-3))
	return mas
def mas2rad(mas):
	rad = mas*(10**(-3)) / (3600*180/np.pi)
	return rad

################################
##### Telescope parameters #####
D = 6.5
pixscal=65
lam=4.8e-6
lamD_pix = pixscal/rad2mas(lam/D)
nlamD = lamD_pix*npix
################################


################################
##### Load in pupil files ######
pup_support = fits.open()

# get zernike normalization right
zfac=[]
nrow=4
for ii in range(nrow):
    for jj in range(ii+1):
        zfac.append(comb(ii,jj))
	
def photon_noise(img, nphot):
    nimg = img/img.sum()
    countimage = nimg*nphot
    photonnoise = rand.poisson(countimage)
    return countimage + photonnoise

def generate_data(aberration, support, band=None):
    npix = support.shape[0]
    pupE = suppport*np.exp(11j*aberration)
    imagE = mft(pupE, nlamD, npix)
    return abs(imageE)**2

if __name__ == "__main__":

    aberrfn = "z10-inputaberr.fits"

    if load_zernike==True:
        aberr=fits.getdata()
    else:
        nz = 10 # number of zmodes
        npix = 256
        aberr = make_aberr("z{0:02d}-inputaberr.fits".format(nz), nz, npix) 

    psf = generate_data(aberr, pup_support)

    estim_pupphase = apwfs()

    # GS takes: 
    #   data psf 
    #   pupil support, 
    #   an estimate of the phase in the pupil over some pupil geometry (e.g. and NRM)
    #   the support for your pupil phase estimate
    #   optionally you can provide the true phase for comparison with kw arg "god"
    gs = GS(psf, pup_support, estim_pupphase, pup_support, god=)

    # some stuff to set up
    gs.condition = 1.0e-6 # difference between phase estimate in consecutive iterations, 
                          # conditions for stopping
    gs.nlamD = nlamD
    gs.orig_aberr = aberr
    gs.damping = True
    gs.zsmooth = True

    # run the gs loop
    gswavefront = gs.find_wavefront()
    gswavefront[fullpupmask=0]=0 # just in case

    # Save some stuff we care about:
    np.savetxt(, gs.zcoeffs) # save final zernike coefficients
    np.savetxt(, gs.metriclist) # save the convergence criteria at each step
    np.savetxt(, gs.errorlist)
    
