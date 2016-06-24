#!/usr/bin/env python
import pdb
''' -------------------------------------------------------
    This procedure generates a coordinates file for a hex
    pupil made of an arbitrary number of rings.
    Additional constraints on the location of spiders make
    it look like the your favorite telescope primary mirror
    ------------------------------------------------------- '''

import numpy as np, matplotlib.pyplot as plt
import time
import pickle

nr     = 50              # rings within the pupil (should be ~> 50)
rmax   = 0.99*5.093/2.           # outer diameter:      5.093 m
# rmax   = 4.62 / 2.        # medium cross: 92 % of primary
rmin   = 0.36*rmax#2.5/2.           # central obstruction: 50 % with Med-cross!
# thick  = 0.257            # adopted spider thickness (meters)
# thick = 0.4

srad = 0.1        # segment "radius"
rad = np.sqrt(3)*srad # radius of the first hex ring in meters

xs = np.array(())
ys = np.array(())

fig = plt.figure(0, figsize=(6,6))
plt.clf()
ax = plt.subplot(111)
circ1 = plt.Circle((0,0), rmax, facecolor='none', linewidth=1)
circ2 = plt.Circle((0,0), rmin, facecolor='none', linewidth=1)
ax.add_patch(circ1)
ax.add_patch(circ2)
#plt.clf()
ax.axis([-rmax,rmax, -rmax,rmax], aspect='equal')

# x = np.linspace(-rmax,rmax,25)
# y = np.linspace(-rmax,rmax,25)

# xs,ys = np.meshgrid(x,y)

# xs, ys = xs.ravel(),ys.ravel()

for i in range(1-nr, nr, 1):
    for j in xrange(1-nr, nr, 1):
        x = srad * (i + 0.5 * j)
        y = j * np.sqrt(3)/2.*srad
        if (abs(i+j) < nr):
            xs = np.append(xs, x)
            ys = np.append(ys, y)

# modifications to match the actual telescope pupil (1): diameter constraints
# -----------------------------------------------------------------------
xx, yy = xs.copy(), ys.copy()        # temporary copies
xs, ys = np.array(()), np.array(())  # start from scratch again

osize = 0.01 * 2*rmax # oversize is a fixed % of total telescope size

for i in range(xx.size):
    thisrad = np.sqrt(xx[i]**2 + yy[i]**2)
    if ((rmin+osize) < thisrad < (rmax-osize)):# + 0.1*srad)):
        xs = np.append(xs, xx[i])
        ys = np.append(ys, yy[i])

# modifications to match the actual telescope pupil (2): spiders
# -----------------------------------------------------------
rm_spiders = True
spiderangles = np.pi/180.*np.array([90, 180, 322-90])
thicks = rmin*np.array([0.2,0.6,0.2])
spiderindices = []
sides = ['left','top','bottom']

if rm_spiders:
    xx, yy = xs.copy(), ys.copy()        # temporary copies
    xs, ys = np.array(()), np.array(())  # start from scratch again

    for j,spider in enumerate(spiderangles):
        thick = thicks[j]
        grad = np.tan(spider)

        for i in range(xx.size):

            if sides[j]=='left':
                condition = yy[i]>=0
            elif sides[j]=='top':
                condition = xx[i]>=0
            elif sides[j]=='bottom':
                condition = yy[i]<=0

            dist = abs(grad*xx[i]-yy[i])/np.sqrt(1+grad**2)

            if dist<thick and condition==False:
                spiderindices.append(i)

    xs = np.delete(xx,spiderindices)
    ys = np.delete(yy,spiderindices)  

ns = xs.size # number of samples

# final modification: global pupil rotation
# -----------------------------------------
th0 = -45.0 * np.pi / 180.0
rmat = np.matrix([[np.cos(th0), np.sin(th0)], [np.sin(th0), -np.cos(th0)]])

coords = np.array([xs, -ys])
coords2 = np.array(np.dot(rmat, coords))

xs = coords2[0,:]#np.reshape(coords2[0], ns)
ys = coords2[1,:]#np.reshape(coords2[1], ns)


# plot segments
# -------------
r0 = srad/np.sqrt(3)
th = 2*np.pi*np.arange(6)/6. + np.pi/6.

for i in range(xs.size):
    hx = xs[i] + r0 * np.cos(th + th0)
    hy = ys[i] + r0 * np.sin(th + th0)
    ax.fill(hx, hy, fc='none', linewidth=1)

ax.plot(xs, ys, 'r.')

np.savetxt("swiftmask.txt", np.transpose((xs,ys)), 
           fmt='%12.9f')

print "--------------------------------------------------"
print "%d pupil sample points were included in the pupil " % xs.size
print "--------------------------------------------------"

plt.show()

mask = np.array([xs,ys])
mask = mask.T

data = {'mask'   : mask
        }

myf = open('./geometry/swiftmask.pick','w')
pickle.dump(data, myf, -1)
myf.close()

np.savetxt('swiftmask.txt',mask)

print 'Mask coordinates successfully saved'
