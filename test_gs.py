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
from gsgs import gsalgo
from poppy.matrixDFT import matrix_dft as mft
from simtools import makedisk, mas2rad, baselinify, makeA


class maskobj:
    def __init__(self, ctrs=None, hdia=None):
        """
        If you don't have any NRM geometry handy, here are hole centers & size for
        NIRISS-like mask
        """
        if ctrs is None:
            """Default is NIRISS AMI hole centers rounded to nearest hundreth of m"""
            self.ctrs = np.array([[ 0.00,  -2.64],
            	                  [-2.29 ,  0.00],
	                              [ 2.29 , -1.32],
            	                  [-2.29 ,  1.32],
	                              [-1.14 ,  1.98],
             	                  [ 2.29 ,  1.32],
	                              [ 1.14 ,  1.98]] )
        else:
            self.ctrs = ctrs
        if hdia is None:
            """Default is NIRISS AMI hole diameter"""
            self.hdia = 0.80

def generate_aberration(npix, nz, pv, livepupilrad=None, debug=False, readin=False):
    """
    Generate a zero mean aberration (w/o piston) in radians
    """
    if readin is not False:
        pupmask = fits.getdata(readin)
        npix = pupmask.shape[0]
        livepupilrad = npix/2

    y,x = np.indices((npix,npix)) - npix/2.0
    if livepupilrad is not None:
        rho = np.sqrt( (x*x) + (y*y) ) / livepupilrad
    else:
        rho = np.sqrt( (x*x) + (y*y) ) / (npix/2.0)
    theta = np.arctan2(y,x)

    aberr = np.zeros(rho.shape)
    for z in range(nz):
        aberr += np.random.rand()*zern.zernikel(z+1,rho,theta)
        aberr[rho>1] = 0
        aberr[rho<0.2] = 0
        if debug:
            plt.imshow(aberr)
            plt.show()

    aberr = aberr - np.mean(aberr[rho<=1])
    # force P2V
    aberr = pv * aberr/(aberr.max() - aberr.min())
    print "rms pupil aberr:", np.sqrt(np.var(aberr[rho<1]))
    if readin is False:
        pupmask = np.ones(rho.shape)
        pupmask[rho>1] = 0
        pupmask[rho<0.2] = 0
    return aberr, pupmask

def make_PSF(pupmask, aberr, lam_c, telD, pixscale, bandwidth=0, debug=False):
    npix = pupmask.shape[0]
    nlamD = npix*pixscale/(lam_c/telD)
    if bandwidth == 0:
        pupil = abs(pupmask)*np.exp(1j*aberr)
        efield = mft(pupil, nlamD, npix)
        psf = abs(efield)**2

    else:
        nwavs = 10
        lambdas = np.linspace(lam_c - (bandwidth/2)*lam_c, lam_c + (bandwidth/2)*lam_c, num=nwavs)
        psf = np.zeros((npix,npix))
        for qq, lam in enumerate(lambdas):
            pupphase = aberr*lam_c/lam # scale the pupil phase by lambda
            pupil = np.abs(pupmask)*np.exp(1j*pupphase)
            efield = mft(pupil, nlamD, npix)
            psf += abs(efield)**2

    if debug:
        print "Number of pixels in image:", npix
        print "Number of lam/D in image:", nlamD
        print "Number of lam/D per pixel:", nlamD/npix
        plt.set_cmap("gray")
        plt.subplot(131)
        plt.title("pupil mask")
        plt.imshow(pupmask)
        plt.subplot(132)
        plt.title("aberration")
        plt.imshow(aberr)
        plt.subplot(133)
        plt.title("PSF")
        plt.imshow(abs(efield)[100:-100, 100:-100])
        plt.show()

    return psf

def make_mask(maskobj, npix, OD, debug=False):
    """
    OD will be set to half the array size to match defaults above
    """
    m2pix = (npix)/OD
    pupil = np.zeros((npix, npix))
    for holectr in maskobj.ctrs:
        #center = (0.5*npix + holectr[0]*m2pix -0.5, 0.5*npix + holectr[1]*m2pix -0.5)
        center = (holectr[0]*m2pix, holectr[1]*m2pix)
        pupil += makedisk(npix, R = 0.5*maskobj.hdia*m2pix, ctr=center)

    if debug:
       plt.title("making pupil mask")
       plt.imshow(pupil, cmap="gray")
       plt.show()

    return pupil

def measure_nrm_pistons(maskobj, nrmpsf, telD, lam, pscale, debug=False):

    baselines, lengths, labels = baselinify(maskobj.ctrs)
    #scalefac = 0.48*nrmpsf.shape[0]/telD
    # test tolerance:
    lamD_pix = (lam/telD)/pscale
    scalefac = 0.4*nrmpsf.shape[0]/telD
    scalefac = (nrmpsf.shape[0]/telD) / lamD_pix

    # calculate complex visibilities
    cvarray = mft(nrmpsf, nrmpsf.shape[0], nrmpsf.shape[0])
    #cvarray = mft(nrmpsf, 0.2*nrmpsf.shape[0], nrmpsf.shape[0])
    cvpha = np.angle(cvarray)

    fringephases = np.zeros(len(lengths))
    phaseloc = np.zeros((cvpha.shape[0], cvpha.shape[1]))
    regionmask = phaseloc.copy()
    for bl in range(len(lengths)):
        ctr = baselines[bl, 0]*scalefac, baselines[bl, 1]*scalefac
        # mask a small region
        region = makedisk(cvpha.shape[0], R=5, ctr=ctr)
        regionasymm = makedisk(cvpha.shape[0], R=5, ctr=(-ctr[0], -ctr[1]))
        regionmask += region-regionasymm

        fringephases[bl] = np.mean(cvpha[region==1])
        phaseloc[region==1] = fringephases[bl]
        phaseloc[regionasymm==1] = -fringephases[bl]

    bl_mat = makeA(len(maskobj.ctrs))
    hole_phases = np.dot(np.linalg.pinv(bl_mat), fringephases)

    if debug:
        plt.set_cmap("gray")
        plt.subplot(1,4,1)
        plt.title("visibility amp array")
        plt.imshow(np.sqrt(abs(cvarray)))
        plt.subplot(1,4,2)
        plt.title("visibility phase array")
        threshold = np.ones(cvpha.shape)
        threshold[abs(cvarray)<0.01] = 0
        plt.imshow(cvpha*threshold)
        plt.subplot(1,4,3)
        plt.title("selection")
        plt.imshow(phaseloc)
        plt.subplot(1,4,4)
        plt.title("PSF")
        plt.imshow(nrmpsf[100:-100, 100:-100])
        plt.show()

    print "Hole phases measured:", hole_phases
    return hole_phases

def create_pupilestim_array(maskobj, pistons, npix, OD, trueaberr = np.zeros((10,10)),debug=False):

    m2pix = 0.95*(npix)/OD
    pupilest = np.zeros((npix, npix))
    fullmask=pupilest.copy()
    for q, ctrloc in enumerate(maskobj.ctrs):
        #ctrs = cvpha.shape[0]/2. - ctrloc[0], cvpha.shape[0]/2. - ctrloc[1]
        ctrs = ctrloc[0]*m2pix, ctrloc[1]*m2pix
        mask = makedisk(npix, 8, ctr=ctrs)
        fullmask += mask
        #print q
        #print pistons
        pupilest[mask==1] = pistons[q]

    if debug:
        plt.title("pupil estimate")
        plt.subplot(131)
        plt.imshow(pupilest)
        plt.axis("off")
        plt.colorbar()
        plt.subplot(132)
        plt.title("What the mask sees")
        plt.imshow(trueaberr*fullmask)
        plt.axis("off")
        plt.colorbar()
        plt.subplot(133)
        plt.title("True aberration")
        plt.imshow(trueaberr)
        plt.axis("off")
        plt.colorbar()
        plt.show()

    return pupilest

def run_gsgs(psf, pupsupport, pupconstraint, D, lam, pscale):

    npix = pupsupport.shape[0]
    lamD_pix = pscale / (lam/D) # lam/D per pix
    nlamD = lamD_pix*npix
    
    print "npix:", npix, "gives", nlamD, "lam/D across the image"

    nrm_support = pupconstraint.copy()
    nrm_support[abs(pupconstraint)>0] = 1
    gs = gsalgo.NRM_GS(psf, pupsupport, pupconstraint, nrm_support, nz=15, watch=False)
    gs.nlamD = nlamD
    gs.damping = True
    gs.zsmooth=True
    gs.condition_c = 1.0e-2
    gs.nitermax = 500
    gs.nitermax_c = 25
    wavefront = gs.find_wavefront()

    print "==========================="
    print wavefront
    print np.nanmax(wavefront)
    print wavefront.imag
    print wavefront.real
    print "==========================="

    #np.savetxt(save.replace(".fits", "_convlist.txt"), gs.metriclist)
    #fits.PrimaryHDU(data=np.angle(wavefront), header=psfhdr).writeto(save, clobber=True)

    plt.set_cmap("BrBG")
    plt.subplot(121)
    plt.imshow(np.angle(wavefront))
    plt.axis("off")
    plt.colorbar()
    plt.subplot(122)
    plt.semilogy(np.arange(len(gs.metriclist)), gs.metriclist)
    plt.show()
    return wavefront, gs.pup_i

def simple_demo():

    # set up pupil/image params - matching NIRISS right here
    telD = 6.5
    lam1 = 4.3e-6
    telD = 6.5
    pscale = mas2rad(65)

    # Simulate an aberration and grab the pupil support array. 
    # npix are number of pixels in the image, nz is number of zernike terms
    # pv is the peak to valley amount of aberration (kept below phase wrapping here)
    # Have an option to read in a different pupil array here if you have one handy
    aberr, pupsupport = generate_aberration(npix=256, nz=10, pv=2.5)

    # Make a PSF from our example pupil and aberration, given the params we set up above
    # Optional "debug" keyword here to see that you made your pupil correctly
    # Setting debug=True will pop up a figure with your pupil, aberration, and PSF
    psf_430 = make_PSF(pupsupport, aberr, lam1, telD, pscale, bandwidth=0.05)

    # maskdef stores NRM geometry
    maskdef = maskobj()
    # Just in case. 
    maskdef.ctrs = np.array(maskdef.ctrs)

    # Makes the mask array the same size as our fake pupil from the geometry in maskdef
    # Turn on debug here to see the mask that is created
    nrmask = make_mask(maskdef, aberr.shape[0], telD)
    # Creates the NRM image (optional debug again)
    nrm_430 = make_PSF(nrmask, aberr, lam1, telD, pscale, bandwidth=0.05)#, debug=True)

    # Measure hole pistons from the NRM image. Turn on debug to see if it's working,
    # some plots will pop up showing how phases were selected from the visibility array
    pistons_430 = measure_nrm_pistons(maskdef, nrm_430,telD, lam1, pscale)#, debug=True)
    # If trueaberr is specified (i.e. if you are doing a simulation and you know what
    # aberration you put in) then debuf will show you the pupil estimate next to the 
    # aberration. These should match if things are working properly. 
    pupestim = create_pupilestim_array(maskdef, pistons_430, aberr.shape[0], telD, trueaberr=aberr, debug=True)

    # After all that setup now we can run gsgs!
    # This will measure the "recovered" wavefront.
    # Running this function will also pop up a plot showing the convergence
    # The algorithm should converge smoothly if things are going well.
    recovered, pup_i = run_gsgs(psf_430, pupsupport, pupestim, telD, lam1, pscale)

    # Some plots to see how we did. 
    print "************ TEST # 1: Does it work at all? *************"
    print "How did we do?", "rms error:", np.std(aberr - np.angle(recovered))
    vmax = aberr.max()
    vmin = aberr.min()
    plt.subplot(231)
    plt.title("Starting aberration")
    plt.imshow(aberr, vmax=vmax, vmin=vmin)
    plt.axis("off")
    plt.colorbar()
    plt.subplot(234)
    plt.title("Recovered aberration")
    plt.imshow(np.angle(recovered), vmax=vmax, vmin=vmin)
    plt.axis("off")
    plt.colorbar()
    plt.subplot(233)
    plt.title("Constraint")
    plt.imshow(pupestim, vmin=vmin, vmax=vmax)
    plt.axis("off")
    plt.colorbar()
    plt.subplot(232)
    plt.title("NRM")
    plt.imshow(nrmask)
    plt.axis("off")
    plt.colorbar()
    plt.subplot(236)
    plt.title("mask * aberration")
    plt.imshow(nrmask*aberr, vmax=vmax, vmin=vmin)
    plt.axis("off")
    plt.colorbar()
    plt.subplot(235)
    plt.title("residual aberration")
    plt.imshow((aberr - np.angle(recovered))*pupsupport)
    plt.axis("off")
    plt.colorbar()
    plt.show()

if __name__ == "__main__":

    simple_demo()

