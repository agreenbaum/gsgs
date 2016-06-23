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
from gsgs.NRM_mask_definitions import NRM_mask_definitions
from simtools import makedisk, mas2rad, baselinify, makeA


def generate_aberration(npix, nz, pv, livepupilrad=None, debug=False):
    """
    Generate a zero mean aberration (w/o piston) in radians
    """
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
        if debug:
            plt.imshow(aberr)
            plt.show()

    aberr = aberr - np.mean(aberr[rho<=1])
    # force P2V
    aberr = pv * aberr/(aberr.max() - aberr.min())
    print "rms pupil aberr:", np.sqrt(np.var(aberr[rho<1]))
    pupmask = np.ones(rho.shape)
    pupmask[rho>1] = 0
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

def measure_nrm_pistons(maskobj, nrmpsf, telD, debug=False):

    baselines, lengths, labels = baselinify(maskobj.ctrs)
    scalefac = 0.48*nrmpsf.shape[0]/telD

    # calculate complex visibilities
    cvarray = mft(nrmpsf, nrmpsf.shape[0], nrmpsf.shape[0])
    cvpha = np.angle(cvarray)

    fringephases = np.zeros(len(lengths))
    phaseloc = np.zeros((cvpha.shape[0], cvpha.shape[1]))
    for bl in range(len(lengths)):
        ctr = baselines[bl, 0]*scalefac, baselines[bl, 1]*scalefac
        # mask a small region
        region = makedisk(cvpha.shape[0], R=5, ctr=ctr)

        fringephases[bl] = np.mean(cvpha[region==1])
        phaseloc[region==1] = fringephases[bl]

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

def create_pupilestim_array(maskobj, pistons, npix, OD, debug=False):

    m2pix = 0.95*(npix)/OD
    pupilest = np.zeros((npix, npix))
    for q, ctrloc in enumerate(maskobj.ctrs):
        #ctrs = cvpha.shape[0]/2. - ctrloc[0], cvpha.shape[0]/2. - ctrloc[1]
        ctrs = ctrloc[0]*m2pix, ctrloc[1]*m2pix
        mask = makedisk(npix, 8, ctr=ctrs)
        #print q
        #print pistons
        pupilest[mask==1] = pistons[q]

    if debug:
        plt.title("pupil estimate")
        plt.imshow(pupilest)
        plt.show()

    return pupilest

def run_gsgs(psf, pupsupport, pupconstraint, D, lam, pscale):

    npix = pupsupport.shape[0]
    lamD_pix = pscale / (lam/D) # lam/D per pix
    nlamD = lamD_pix*npix
    
    print "npix:", npix, "gives", nlamD, "lam/D across the image"

    nrm_support = pupconstraint.copy()
    nrm_support[abs(pupconstraint)>0] = 1
    gs = gsalgo.NRM_GS(psf, pupsupport, pupconstraint, nrm_support)
    gs.nlamD = nlamD
    gs.damping = True
    gs.zsmooth=True
    gs.condition_c = 1.0e-2
    gs.nitermax = 500
    gs.nitermax_c = 500
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

    # set up pupil/image params
    telD = 6.5
    lam1 = 4.3e-6

    aberr, pupsupport = generate_aberration(npix=256, nz=5, pv=2.0)

    psf_430 = make_PSF(pupsupport, aberr, 4.3e-6, 6.5, mas2rad(65), bandwidth=0.05)

    maskdef = NRM_mask_definitions(maskname="jwst_g7s6c")
    maskdef.ctrs = np.array(maskdef.ctrs)

    nrmask = make_mask(maskdef, aberr.shape[0], telD)
    nrm_430 = make_PSF(nrmask, aberr, 4.3e-6, 6.5, mas2rad(65), bandwidth=0.05)

    pistons_430 = measure_nrm_pistons(maskdef, nrm_430,telD, debug=True)
    pupestim = create_pupilestim_array(maskdef, pistons_430, aberr.shape[0], telD)

    recovered, pup_i = run_gsgs(psf_430, pupsupport, pupestim, telD, lam1, mas2rad(65))

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
    plt.imshow(aberr - np.angle(recovered))
    plt.axis("off")
    plt.colorbar()
    plt.show()

if __name__ == "__main__":

    simple_demo()

