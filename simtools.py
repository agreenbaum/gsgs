#! /usr/bin/env python

import numpy as np
import sys,os


def mas2rad(mas):
    rad = mas*(10**(-3)) / (3600*180/np.pi)
    return rad

def rad2mas(rad):
    mas = rad * (3600*180/np.pi) * 10**3
    return mas

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

def makeA(nh, verbose=False):
    """ 
    Writes the "NRM matrix" that gets pseudo-inverterd to provide
    (arbitrarily constrained) zero-mean phases of the holes.

    makeA taken verbatim from Anand's pseudoinverse.py

     input: nh - number of holes in NR mask
     input: verbose - True or False
     output: A matrix, nh columns, nh(nh-1)/2 rows  (eg 21 for nh=7)

    Ax = b  where x are the nh hole phases, b the nh(nh-1)/2 fringe phases,
    and A the NRM matrix

    Solve for the hole phases:
        Apinv = np.linalg.pinv(A)
        Solution for unknown x's:
        x = np.dot(Apinv, b)

    Following Noah Gamper's convention of fringe phases,
    for holes 'a b c d e f g', rows of A are 

        (-1 +1  0  0  ...)
        ( 0 -1 +1  0  ...)
    

    which is implemented in makeA() as:
        matrixA[row,h2] = -1
        matrixA[row,h1] = +1

    To change the convention just reverse the signs of the 'ones'.

    When tested against Alex'' NRM_Model.py "piston_phase" text output of fringe phases, 
    these signs appear to be correct - anand@stsci.edu 12 Nov 2014

    anand@stsci.edu  29 Aug 2014
        """

    print( "\nmakeA(): ")
    #                   rows         cols
    ncols = (nh*(nh-1))//2
    nrows = nh
    matrixA = np.zeros((ncols, nrows))
    if verbose: print( matrixA)
    row = 0
    for h2 in range(nh):
        if verbose: print()
        for h1 in range(h2+1,nh):
            if h1 >= nh:
                break
            else:
                if verbose:
                    print( "R%2d: "%row)
                    print( "%d-%d"%(h1,h2))
                matrixA[row,h2] = -1
                matrixA[row,h1] = +1
                row += 1
    if verbose: print()
    return matrixA


def makeK(nh, verbose=False):
    """ 
    As above, write the "kernel matrix" that converts fringe phases
    to closure phases. This can be psuedo-inverted to provide a 
    subset of "calibrated" fringe phases (hole-based noise removed)

     input: nh - number of holes in NR mask
     input: verbose - True or False
     output: L matrix, nh(nh-1)/2 columns, comb(nh, 3) rows  (eg 35 for nh=7)

    Kx = b, where: 
        - x are the nh(nh-1)/2 calibrated fringe phases 
        - b the comb(nh, 3) closure phases,
    and K the kernel matrix

    Solve for the "calibrated" phases:
        Kpinv = np.linalg.pinv(K)
        Solution for unknown x's:
        x = np.dot(Kpinv, b)

    Following the convention of fringe phase ordering above, which should look like:
    h12, h13, h14, ..., h23, h24, ....
    rows of K should look like:

        (+1 -1  0  0  0  0  0  0 +1 ...) e.g., h12 - h13 + h23
        (+1 +1  0 +1  ...)
    

    which is implemented in makeK() as:
        matrixK[n_cp, f12] = +1
        matrixK[n_cp, f13] = -1
        matrixK[n_cp, f23] = +1

    need to define the row selectors
     k is a list that looks like [9, 9+8, 9+8+7, 9+8+7+6, 9+8+7+6+5, ...] 
     -----up to nh*(nh-1)/2
     i is a list that looks like [0,9, 9+8, 9+8+7, 9+8+7+6, 9+8+7+6+5, ...] 
     -----up to nh*(nh-1)/2 -1
    because there are 9 fringe phases per single hole (decreasing by one to avoid repeating)
    hope that helps explain this!

    agreenba@pha.jhu.edu  22 Aug 2015
        """

    print( "\nmakeK(): ")
    nrow = comb(nh, 3)
    ncol = nh*(nh-1)/2

    # first define the row selectors
    # k is a list that looks like [9, 9+8, 9+8+7, 9+8+7+6, 9+8+7+6+5, ...] 
    # -----up to nh*(nh-1)/2
    # i is a list that looks like [0,9, 9+8, 9+8+7, 9+8+7+6, 9+8+7+6+5, ...] 
    # -----up to nh*(nh-1)/2 -1
    countk=[]
    val=0
    for q in range(nh-1):
        val = val + (nh-1)-q
        countk.append(val)
    counti = [0,]+countk[:-1]
    # MatrixK
    row=0
    matrixK = np.zeros((nrow, ncol))
    for ii in range(nh-2):
        for jj in range(nh-ii-2):
            for kk in range(nh-ii-jj-2):
                matrixK[row+kk, counti[ii]+jj] = 1
                matrixK[row+kk, countk[ii+jj]+kk] = 1
                matrixK[row+kk, counti[ii]+jj+kk+1] = -1
            row=row+kk+1
    if verbose: print()

    return matrixK

def baselinify(ctrs):
    N = len(ctrs)
    uvs = np.zeros((N*(N-1)//2, 2))
    label = np.zeros((N*(N-1)//2, 2))
    bllengths = np.zeros(N*(N-1)//2)
    nn=0
    for ii in range(N-1):
        for jj in range(N-ii-1):
            uvs[jj+nn, 0] = ctrs[ii,0] - ctrs[ii+jj+1,0]
            uvs[jj+nn, 1] = ctrs[ii,1] - ctrs[ii+jj+1,1]
            bllengths[jj+nn] = np.sqrt((ctrs[ii,0]-ctrs[ii+jj+1,0])**2 +\
                        (ctrs[ii,1]-ctrs[ii+jj+1,1])**2)
            label[jj+nn,:] = np.array([ii, ii+jj+1])
        nn = nn+jj+1
    return uvs, bllengths, label


#def simulate_zern(fov = 80, mode=5, livepupilrad=None, mask=None):
#   """
#   Default Astig_1 0.1 radians
#   """
#   # Define our coordinates for wavefront aberration of choice
#   y,x = np.indices((fov,fov)) - fov/2
#   if livepupilrad is not None:
#       rho = np.sqrt( (x*x) + (y*y) ) / livepupilrad
#   else:
#       rho = np.sqrt( (x*x) + (y*y) ) / (fov/2) 
#   theta = np.arctan2(y,x)
#   if hasattr(mode, "__iter__"):
#       aberr = np.zeros(rho.shape)
#       frac = 1/len(mode)
#       for md in mode:
#           aberr += zern.zernikel(md,rho,theta)
#   else:
#       aberr = zern.zernikel(mode, rho, theta)
#   if mask is not None:
#       mask = mask
#   else:
#       mask = np.ones((fov,fov))
#   mask_aberr = mask * aberr
#   # print( out rms of this aberration over circular pupil?  - AS)
#   print( np.var(aberr[rho<1]))
#   return mask_aberr, aberr
#
#def avg_piston(aberr, positions, mask, R = 20, point=False):
#   """
### aberr is the full square array aberration
#
#   positions are approximaete coordinates of mask holes recorded by eye
#
#   mask is the NRM or other pupil mask for over which to compute average phases
#
#   R - radius to enclose mask holes around coordinate points (also by eye for now)
#       purpose to choose holes one by one. 
#
#   point -- should this return average phase values in a single pixel (TRUE)
#        or over the whole mask support (FALSE). Recommended to remain FALSE.
#   """
#   fov = aberr.shape[0]
#   avgphase = np.zeros(aberr.shape) #radians
#   holemask = avgphase.copy()
#   x,y = np.indices(holemask.shape)
#   debug = avgphase.copy()
#   new_support = avgphase.copy()
#   for ii,pos in enumerate(positions):
#       holemask = makedisk(fov, R, ctr = (pos[0] - fov/2, pos[1] - fov/2))
#       debug+=holemask
#       holemask[abs(holemask*mask)>0] = 1 
#       holemask[abs(holemask*mask)==0] = 0
#       if not point:
#           avgphase[holemask==1] = aberr[holemask==1].sum() / len(aberr[holemask==1])
#       else:
#           #int(np.sum(holemask*x)/holemask.sum()),int(np.sum(holemask*y)/holemask.sum())
#           avgphase[int(np.sum(holemask*x)/holemask.sum()),\
#            int(np.sum(holemask*y)/holemask.sum())] = \
#                   aberr[holemask==1].sum() / len(aberr[holemask==1])
#       #plt.figure()
#       #plt.imshow(avgphase)
#       #plt.figure()
#       #plt.imshow(debug)
#       #plt.show()
#   new_support[abs(avgphase)>0] = 1
#"""
#   return avgphase, new_support
