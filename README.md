# gsgs - A Greenbaum-Sivaramakrishnan style Gerchberg-Saxton
GSGS depends on some functions and utilities in the poppy package:
Matrix_dft, zernike available at [https://github.com/mperrin/poppy](https://github.com/mperrin/poppy)

The core program here is gsalgo.py

Provided are also:

* zerniketools.py - for zernike smoothing an simulating aberrations. Mostly using programs from Tim van Werkhoven's [python](http://python101.vanwerkhoven.org)
* simtools.py - some stuff for generating fake data, pupils, etc.


I've written a short example driver, test_gs.py