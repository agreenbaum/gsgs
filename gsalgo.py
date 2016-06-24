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


import sys,os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from poppy.matrixDFT import matrix_dft as mft
#from matrixDFT import matrix_dft as mft
from simtools import *
from zerniketools import ZernikeFitter, HexikeFitter

class NRM_GS:
    def __init__(self, dpsf, pupsupport, pha_constraint, c_support, segmentwhich=None, god=None,
                 watch=False):
        """
        requires both full pupil and nrm data and pupil supports for each
        which array needs to be specified for segmented telescope
        """
        self.dpsf = dpsf
        # A.G.: For real data - if you have any neg numbers set to zero (advice from J. Krist)
        self.dpsf[self.dpsf<0] = 0
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
        self.nitermax_c = 10
        self.nitermax = 20
        self.zsmooth=True
        self.segmentwhich = segmentwhich

        # Do you want to see what the algorithm is doing each iteration? Set to True.
        self.watch=watch

        if god is not None:
            # give it the true phase to constrain
            self.god=god
        self.damping=False

        if segmentwhich is None:
            side = self.pupsupport.shape[0]
            radsupport = self.pupsupport[:,side/2].sum() // 2
            self.residmask = makedisk(side, radsupport-2)#,array="EVEN")
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
            self.make_iter_step(self.pupil_i)
            print "Iteration no. ", self.i
            print "Value of difference metric:", self.metric
            if (self.i==105 & hasattr(self, "fitscubename")):
                fits.PrimaryHDU(data=self.fitscube).writeto(self.fitscubename,\
                        clobber=True)
            if self.i > self.nitermax_c:
                print "Reached max constraint iterations"
                break
        # Force this to stop without relaxing constraint
        if self.i > self.nitermax:
            self.metric = self.condition - 0.5*self.condition

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
            if self.i > self.nitermax:
                print "Reached 500 iterations, stopping..."
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

        """
        plt.figure()
        plt.title("support of phase constraint")
        plt.imshow(self.c_support+self.pup_in)
        plt.colorbar()
        plt.figure()
        plt.title("phase constraint")
        plt.imshow(self.constraint+self.puppha_i)
        plt.colorbar()
        plt.show()
        """
        if self.watch:
            plt.figure()
            if self.i>1:
                print "convergence:",self.metriclist[-1]
            #plt.imshow(self.puppha_i)
            plt.subplot(221)
            plt.title("Pupil amplitude, iteration {0}".format(self.i-1))
            plt.imshow(abs(self.pup_i))
            plt.subplot(222)
            if self.i>1:
                plt.title("current PSF, iteration {0}".format(self.i-1))
                plt.imshow(abs(self.currentpsf)[100:-100, 100:-100], cmap="gray")
            else:
                plt.title("data PSF, iteration {0}".format(self.i-1))
                plt.imshow(np.sqrt(self.dpsf)[100:-100, 100:-100], cmap="gray")
            plt.subplot(223)
            plt.title("Pupil wavefront, iteration {0}".format(self.i-1))
            plt.imshow(np.angle(abs(self.pup_in)*self.pupsupport*np.exp(1j*self.puppha_i)))
            plt.colorbar()
            plt.subplot(224)
            plt.title("data PSF".format(self.i-1))
            plt.imshow(np.sqrt(self.dpsf)[100:-100, 100:-100], cmap="gray")
            plt.show()

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
        #self.pupil_i = abs(self.pupil_i)*np.exp(1j*zmeanphase)
        self.pupil_i = abs(self.pupsupport)*np.exp(1j*zmeanphase)

        if self.zsmooth==True:
            # smooth each iterations by fitting zernikes, gets rid of unphysical
            # high frequency artifacts
            self.fit_zernikes()
            #self.pupil_i = abs(self.pupil_i)*np.exp(1j*self.full_rec_wf)
            self.pupil_i = abs(self.pupsupport)*np.exp(1j*self.full_rec_wf)

        self.currentpsf = mft(self.pupil_i, self.nlamD, self.npix_img)

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

        """
        plt.figure()
        print "DEBUG:",self.pup_i
        #plt.imshow(self.puppha_i)
        plt.subplot(131)
        plt.title("Image amplitude, iteration {0}".format(self.i-1))
        plt.imshow(abs(self.amp))
        plt.subplot(132)
        plt.title("Pupil support, iteration {0}".format(self.i-1))
        plt.imshow(self.pupsupport)
        plt.subplot(133)
        plt.title("Pupil wavefront, iteration {0}".format(self.i-1))
        plt.imshow(np.angle(self.pupil_i))
        plt.colorbar()
        plt.show()
        """

        return self.pupil_i

    def compute_metric(self):
        ''' what is the difference between my original PSF and the current iteration? '''

        self.residual = ((self.puppha_i*self.residmask) \
                - (np.angle(self.pupil_i)*self.residmask)) \
                / (self.puppha_i*self.residmask).sum()
        self.residual *=self.residmask

        self.metric = np.std((self.puppha_i[self.residmask==1] - \
                  np.angle(self.pupil_i)[self.residmask==1]))\
                # / (self.puppha_i*self.pupsupport).sum()
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

            fov = self.pupil_i.shape[0]
            sidesz= (fov - self.livepupilD) / 2

            # crop down to just the pupil
            # Edit June 2016: if sidesz is zero this breaks! Added in this if statement.
            if sidesz>0:
                croppedpupphases = (self.pupsupport*np.angle(self.pupil_i))[sidesz:-sidesz, \
                                            sidesz:-sidesz]
                # initialize Zernike fitter
                zf = ZernikeFitter(nzern=nmodes, narr = croppedpupphases.shape[0],
                        extrasupportindex=(self.pupsupport[sidesz:-sidesz,sidesz:-sidesz]==0))
            else:
                croppedpupphases = (self.pupsupport*np.angle(self.pupil_i))
                # initialize Zernike fitter
                zf = ZernikeFitter(nzern=nmodes, narr = croppedpupphases.shape[0],
                        extrasupportindex=(self.pupsupport==0))

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
            if sidesz > 0:
                self.full_rec_wf[sidesz:-sidesz, sidesz:-sidesz] = zf.grid_mask*self.rec_wf
            else:
                self.full_rec_wf = zf.grid_mask*self.rec_wf

            return self.zcoeffs


