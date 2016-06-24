import numpy as np 
import matplotlib.pyplot as plt 
import pysco
import time

'''----------------------------------------------------
sense.py - script to run the eigenphase sensing code
on Dragonfly data.
----------------------------------------------------'''

ddir = './geometry/'
pupil = 'jwst'

'''----------------------------------------------------
Load the data as a row-phase object
----------------------------------------------------'''

tic = time.time()
a = pysco.wfs(ddir+pupil+'.pick',modes=400,weights=True,Ns=3.0)
toc = time.time()

a.save_to_file(ddir+pupil+'model.pick')