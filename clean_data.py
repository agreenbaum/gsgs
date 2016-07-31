from make_splodges import Splodges
from astropy.io import fits
import numpy as np
from matrixDFT import matrix_dft as mft
import driverutils

path = "/Users/agreenba/data/gpi/"
fn = "S20140913E0038_spdc.fits"
writepath = "gpi_demo/"

def write_out_ft():
    splg = Splodges(fn, 15, path, writepath = "gpi_demo/", thresh=0.02, writeall=True)
    splg.make_splodges()

def center_and_crop(array, rad = 60):
    return driverutils.deNaN(3, driverutils.centerit(array, r=rad))

def clean_up_cv():
    # throw out everything outside support (inspected by eye -- ~95 pix radius)
    cv = fits.getdata(writepath+fn.replace(".fits", "_mod.fits"))
    f = fits.open(path+fn)
    cvhdr = f[1].header
    y,x = np.indices(cv.shape[1:])
    fov = cv.shape[1]
    r = np.sqrt((x-fov/2.0)**2+(y-fov/2.0)**2)
    cv[:,r>95] = 0
    cvmask = 0*cv.copy()
    cvmask[cv>0] = 1
    fits.PrimaryHDU(header = cvhdr, data = cvmask).writeto(writepath+"pupsupport.fits", clobber = True)

    cleaned_data = np.zeros((cv.shape[0], fov/2, fov/2))
    windowed_data = fits.getdata(writepath+fn.replace(".fits", "_window.fits"))
    #data = fits.getdata(path+fn)
    # basic cropping + deNaN
    fullcv = np.zeros(cv.shape)
    """
    import matplotlib.pyplot as plt
    dataslc = center_and_crop(data[25,:,:], rad = 20)
    dataslc[np.where(np.isnan(dataslc))] = 0
    plt.subplot(221)
    plt.title("Original PSF")
    plt.imshow(dataslc)
    plt.subplot(223)
    plt.title("Vis Amp")
    cvslc = mft(dataslc, dataslc.shape[0], cv.shape[1])
    plt.imshow(np.sqrt(abs(cvslc)))
    plt.subplot(222)
    plt.title("Cleaned Vis Amp")
    plt.imshow(np.sqrt(abs(cvslc*cvmask[25,:,:])))
    plt.subplot(224)
    plt.title("Cleaned PSF")
    plt.imshow(abs(mft(cvslc*cvmask[25,:,:], dataslc.shape[0], dataslc.shape[0], inverse=True)))
    plt.set_cmap = "gray"
    plt.show()
    """
    for ii in range(cv.shape[0]):
        fullcv[ii,:,:] = mft(windowed_data[ii, :,:], windowed_data.shape[1], cv.shape[1])
        #dataslc = center_and_crop(data[ii,:,:], rad = 20)
        #dataslc[np.where(np.isnan(dataslc))] = 0
        dataslc = windowed_data[ii,:,:]
        #plt.imshow(dataslc)
        #plt.show()
        fullcv[ii,:,:] = mft(dataslc, dataslc.shape[1], cv.shape[1])
        fullcv[ii, :,:] = fullcv[ii,:,:]*cvmask[ii,:,:]
        cleaned_data[ii,:,:] = mft(fullcv[ii,:,:], dataslc.shape[1], dataslc.shape[1], inverse=True)

    fits.PrimaryHDU(header = cvhdr, data = cleaned_data).writeto(writepath+fn.replace(".fits", "_ftcleaned.fits"), clobber = True)
    fits.PrimaryHDU(header = cvhdr, data = cvmask).writeto(writepath+fn.replace(".fits", "_cvmask.fits"), clobber = True)

if __name__ == "__main__":

    #write_out_ft()
    clean_up_cv()
