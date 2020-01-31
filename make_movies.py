#!/usr/bin/env python
# coding: utf-8
from DELCgen import *
import scipy.stats as st
import matplotlib.pyplot as plt 
import matplotlib
from constants import *
from scipy.interpolate import interp1d
import naima
import astropy.units as u
import subroutines as sub
import os
import time
import pickle
from simulation import unit

matplotlib.use("TkAgg")

def power_threshold(rigidity, v_over_c=0.1, eta=0.1):
    power = (0.1 / eta) * (rigidity / 1e19)**2 * (0.1/v_over_c) * 1e44
    return power 

def max_energy(power, v_over_c=0.1, eta=0.1):
    return 1e19*np.sqrt( (power/1e44) * (v_over_c/0.1))

def get_lc(lognorm_params, PSD_params, tbin, Age):
    # Simulation params
    # let's do everything in units of kyr
    # run for 100 Myr (1e5 kyr) in bins of 0.1 Myr
    #lognorm_params = (1.5,0,np.exp(1.5))
    RedNoiseL,RandomSeed,aliasTbin = 100,12,100
    N = Age / tbin

    if lognorm_params[0] == 0.0: # dumb workaround 
        lognorm_params = (1.5,0,np.exp(1.5))
        lc = Simulate_DE_Lightcurve(BendingPL, PSD_params,st.lognorm,lognorm_params,
                                    RedNoiseL=RedNoiseL,aliasTbin=aliasTbin,randomSeed=RandomSeed,LClength=Age, tbin=tbin)
        lc.flux = np.ones_like(lc.flux)
    else:
        lc = Simulate_DE_Lightcurve(BendingPL, PSD_params,st.lognorm,lognorm_params,
                                    RedNoiseL=RedNoiseL,aliasTbin=aliasTbin,randomSeed=RandomSeed,LClength=Age, tbin=tbin)

    return (lc)

# since we're in Kyr units, set up conversion
MYR = 1e3

# set up light curve. pl_index of 1 means red noise.
pl_index = 1

# bend the power law at 1 Myr (1e-3 kyr^-1) and steeply decay it beyond that with index 10. 
# A,v_bend,a_low,a_high,c = 1, 1e-3, pl_index, 10, 1
PSD_params = (1, 1.0 * MYR, pl_index, 10, 1)
tbin = 0.1 * MYR   # 100 kyr 
Age = 100.0 * MYR
Length = int(Age / tbin)

# time is in kyr, so convert to Myr
times = np.arange ( 0, Length*tbin, tbin) 

energies = np.logspace(6,20.5,num=3000)
lognorm_params = (1.5,0,np.exp(1.5))
    
# paramaters for lc are lognorm parameters, PSD parameters, tbin and Length (Age is really number of points)
lc = get_lc(lognorm_params, PSD_params, tbin, Length)


# Set up arrays to loop over. 
# flux_scales contains the normalisations of our jet power 
# actually the median of the distribution, or the mean in log space.
# betas is the spectral index of the injected spectrum 
#flux_scales = np.logspace(43,45,num=10)
betas = [2,2.3,2.7]
flux_scales = np.logspace(42.5,44.5,num=10)
sigmas = np.linspace(0.5,3,num=10)
sigmas = [0.0]
betas = [2]
flux_scales = [7.389e43]


# In[10]:


energies = np.logspace(6,20.5,num=3000)


for i_sigma, SIGMA in enumerate(sigmas):
    
    lognorm_params = (SIGMA,0,np.exp(SIGMA))
    
    # paramaters for lc are lognorm parameters, PSD parameters, tbin and Length (Age is really number of points)
    lc = get_lc(lognorm_params, PSD_params, tbin, Length)
                      
    for i_flux, flux_scale in enumerate(flux_scales):
        # normalise the light curve
        flux = lc.flux * flux_scale 
    
        # loop over spectral indices
        for i_beta, BETA in enumerate(betas):
            
            print (i_beta)

            t1 = time.time()

            # initialise figure and movie file name
            fname = "movies/movie_beta{:.1f}q{:.1f}sig{:.1f}".format(BETA, np.log10(flux_scale),  SIGMA)
            print (fname)
            p, fig, ax1, ax2, ax3 = sub.init_fig(fname=fname, fmt="mp4", fps=6)

            # load the arrays calculated
            ncr_time = np.load("array_saves/ncr_beta{:.1f}q{:.1f}sig{:.1f}.npy".format(BETA, np.log10(flux_scale),  SIGMA))
            #import os
            #print (os.getcwd())
            dimensions = np.load("array_saves/dim_{:.1f}q{:.1f}sig{:.1f}.npy".format(BETA, np.log10(flux_scale),  SIGMA))
            with open("array_saves/jetstore_{:.1f}q{:.1f}sig{:.1f}.npy.pkl".format(BETA, np.log10(flux_scale),  SIGMA), "rb") as pickle_file:
                jet_store = pickle.load(pickle_file)

            jet = jet_store.jet
            # charges 
            z_elem = np.array([1,1,2,7,26])

            movie_counter = 0
            DELTA = 1
            print ("FRAMES:", len(dimensions[0,:,0])/DELTA)
            NPTS = len(dimensions[0,:,0])

            for i in range(1, NPTS, DELTA):            
                # this clears axes and plots the lightcurve 
                ax1.clear()
                sub.plot_lc(ax1, jet_store.time[:i+1] / unit.myr, jet_store.v_j[:i+1]/C)
                ax2.clear()
                ax3.clear()


                for j, frac in enumerate(z_elem):
                    sub.plot_spectra_morph(fig, ax2, ax3, energies, ncr_time[j,i,:], dimensions[:,i,:], j)

                # extract the image as an ARGB string
                fig.canvas.draw()
                string = fig.canvas.tostring_argb()

                # write to pipe
                p.stdin.write(string) 


            print ("BETA {:.1f} median lum {:8.4e} mean lum {:8.4e}".format(BETA, flux_scale, np.mean(flux)))
            # Finish up
            p.communicate()

            print ("{:.2f} seconds".format(time.time() - t1))


# import os
# import io
# import base64
# from IPython.display import HTML

# # for i_flux, flux_scale in enumerate(flux_scales):
# #     for i_beta, BETA in enumerate(betas):
# #         print (flux_scale, BETA)
# #         fname = "beta{:.1f}q{:.1f}".format(BETA, np.log10(flux_scale))
# #         #cmd = "ffmpeg -y -i spectra/{}/spectra%3d.png -r 6 movies/movie_{}.mp4".format(fname,fname)
# #         cmd = "ffmpeg -y -i spectra/{}/spectra%3d.jpg -r 6 movies/movie_{}.gif".format(fname,fname)
# #         os.system(cmd)
# #         time.sleep(3)
        
# # open the last movie?  
# moviename = "movies/movie_{}.mp4".format(fname)
# video = io.open(moviename, 'r+b').read()
# encoded = base64.b64encode(video)
# HTML(data='''<video alt="test" controls>
#                 <source src="data:video/mp4;base64,{0}" type="video/mp4" />
#              </video>'''.format(encoded.decode('ascii')))

# from IPython.display import Video

# Video(moviename )



