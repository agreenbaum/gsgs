# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 17:43:04 2013

@author: anthony
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from makehex import makehex
from makecirc import makecirc

#common quantities
d=1.32               #metres, the edge-to-edge segment diameter 
theta= (30./180.* np.pi)     #angle when moving diagonally across segments
dct=d*np.cos(theta) 
dst=d*np.sin(theta) 
origin = np.array([0.,0.])

'''---------------------
Define segment positions
---------------------'''

s1=[0,0] # center segment 
s2=[0,d] # inner loop segs 2-7
s3=[dct,dst]
s4=[dct,-dst]
s5=[0,-d]
s6=[-dct,-dst]
s7=[-dct,dst] #inner loop segs 2-7
s8=[0,2*d] # outer loop segs 8-19
s9=[dct,d+dst]
s10=[2*dct,2*dst]
s11=[2*dct,0]
s12=[2*dct,-2*dst]
s13=[dct,-d-dst]
s14=[0,-2*d]
s15=[-dct,-d-dst]
s16=[-2*dct,-2*dst]
s17=[-2*dct,0]
s18=[-2*dct,2*dst]
s19=[-dct,d+dst] #outer loop segs 8-19
all_segs=np.array([s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19])

indices = np.arange(19)+1

# cut out particular segments
cut = []
ncut = 1
arg = np.where(indices==cut)
all_segs = np.delete(all_segs,arg,axis=0)
indices = np.delete(indices,arg)


'''---------------------
Subsample the mirrors
---------------------'''

segments = (19)*10000#(19-ncut)*(7+6+6+6+6)

subsampled = np.zeros((segments,2))
newindices = np.zeros(segments)

hexpattern = makehex(origin,0.98*d,niter=1)# makecirc(origin,0.9*d,nrings=3)
hexsize = hexpattern.shape[0]

i = 0 # initialise
for j,seg in enumerate(all_segs):
	subsampled[i:i+hexsize,:] = hexpattern[:,]+seg
	newindices[i:i+hexsize] = indices[j]
	i += hexsize

indices = newindices[0:i]
subsampled = subsampled[0:i]

'''---------------------
Get rid of spiders
---------------------'''

xx,yy = subsampled[:,0],subsampled[:,1]

thick  = 0.05#0.14*2            # adopted spider thickness (meters)
beta   = 60*np.pi/180  # spider angle beta
epsi   = thick/(2*np.sin(beta))/10.
spiderangles = np.array([90,240,300])*np.pi/180.
sides = [1,-1,-1]
spiderindices = []

for j,spider in enumerate(spiderangles):
	side = sides[j]
	grad = np.tan(spider)
	for i in range(xx.size):
		dist = abs(grad*xx[i]-yy[i])/np.sqrt(1+grad**2)
		# xyangle = np.arctan(yy[i]/xx[i])
		if dist<thick/2. and yy[i]*side>0:
			spiderindices.append(i)  

notin = np.array([xx[spiderindices],yy[spiderindices]])
notin = notin.T
notinsegs = np.array([0,0])
cuts = []
xx = np.delete(xx,spiderindices)
yy = np.delete(yy,spiderindices)
indices = np.delete(indices,spiderindices)

subsampled = np.array([xx,yy])
subsampled = subsampled.T

'''---------------------
Plot
---------------------'''

r0 = d/np.sqrt(3)
th = 2*np.pi*np.arange(6)/6. + np.pi/6.

plt.clf()
# plt.scatter(all_segs[:,0],all_segs[:,1])
plt.scatter(subsampled[:,0],subsampled[:,1])
plt.scatter(notin[:,0],notin[:,1],c='r')
for j in range(all_segs.shape[0]):
    hx = all_segs[j,0] + r0 * np.cos(theta+th)
    hy = all_segs[j,1] + r0 * np.sin(theta+th)
    plt.fill(hx, hy, fc='none', linewidth=1)
plt.axis('equal')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title("JWST Mirror Model")
plt.draw()
plt.show()

'''---------------------
Now save the coords
---------------------'''

np.savetxt('./geometry/jwst.txt',subsampled)

data = {'mask'   : subsampled,
        'indices': indices,
        'cuts'	 : cuts,
        'centers': all_segs,
        'notin'	 : notin,
        'nocenters': notinsegs
		}

myf = open('./geometry/jwst.pick','w')
pickle.dump(data, myf, -1)
myf.close()

print 'Mask coordinates successfully saved'