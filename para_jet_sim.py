#!/usr/bin/env python
# coding: utf-8
import scipy.stats as st
import matplotlib.pyplot as plt 
from scipy.interpolate import interp1d
import astropy.units as u
import subroutines as sub
import os
from msynchro.units import unit 
import simulation as sim
from mpi4py import MPI
from itertools import product
import time
from DELCgen import *

def write_variable(filename, variable_dict):
    keys =  list(variable.keys())
    filename.write("Run")
    for k in keys: filename.write("{} ".format(k))
    filename.write("\n")

    for i in range(len(variable[keys[0]])):
        for k in keys:
            filename.write("{} ".format(variable[k][i]))
        filename.write("\n")


# let's do this in parallel 
nproc = MPI.COMM_WORLD.Get_size()       # number of processes
my_rank = MPI.COMM_WORLD.Get_rank()     # The number/rank of this process
my_node = MPI.Get_processor_name()      # Node where this MPI process runs




# Now we've set up the basic functions. Let's initialise our parameters for the variable jet history and initialise the tau_loss dictionary which stores the loss times for the different species (calculated using CRPropa).  

# In[3]:


# since we're in Kyr units, set up conversion
MYR = 1e3

# set up light curve. pl_index of 1 means red noise.
pl_index = 1

# bend the power law at 1 Myr (1e-3 kyr^-1) and steeply decay it beyond that with index 20. 
# A,v_bend,a_low,a_high,c = 1, 1e-3, pl_index, 10, 1
PSD_params = (1, 1.0 * MYR, pl_index, 20, 0)
tbin = 0.1 * MYR   # 100 kyr 
Age = 100.0 * MYR
Length = int(Age / tbin)

# time is in kyr, so convert to Myr
times = np.arange ( 0, Length*tbin, tbin) 

elems = ["H", "He", "N", "Fe"]
tau_loss = dict()

energy_params = (14, 21, 3000)
# create energy bins 
energy_edges = np.logspace(energy_params[0], energy_params[1], energy_params[2] + 1)
energies = 0.5 * (energy_edges[1:] + energy_edges[:-1])

for i in range(len(elems)):
    tau_loss[elems[i]] = sub.Losses(elems[i])
    tau_loss[elems[i]].interpol(energies)
    


# Set up arrays to loop over. flux_scales contains the normalisations of our jet power $(\bar{Q})$.
# This is actually the median of the distribution, or the mean in log space.
# betas is the spectral index of the injected spectrum ($\beta$).
# flux_scales contains the normalisations of our jet power 
# actually the median of the distribution, or the mean in log space.
# betas is the spectral index of the injected spectrum 
#flux_scales = np.logspace(43,45,num=10)

NSIMS = 1000
powerlaw = sim.my_powerlaw(n=1.001, xmin=1e43, xmax=1e48)
Q = powerlaw.rvs(size=NSIMS)

variable = dict()
variable["runid"] = np.arange(0,NSIMS,1)
variable["betas"] = np.ones(NSIMS) * 2.0
variable["flux_scales"] = powerlaw.rvs(size = NSIMS)
variable["sigmas"] = np.random.random(size = NSIMS) * 3
variable["seeds"] = np.random.randint(0, 100.0 * NSIMS, size = NSIMS)

# generate a random lifetime in log space
variable["lifetime"] = 10.0 ** (np.random.random(size=NSIMS) * 2.5)

#indices = list(variable.values())
#parameter_keys = variable.keys()
#parameter_values = list(product(*indices))

logfile = open("logfile_{}.txt".format(my_rank), "w")
params = open("params_{}.txt".format(my_rank), "w")

write_variable(params, variable)
params.close()
# set the atmosphere
atmos = "UPP"

N_MODELS_TOTAL = NSIMS        # total number of models to run 
print ("Total models {}".format(N_MODELS_TOTAL))

# this MUST be an integer division so we get the remainder right 
n_models = N_MODELS_TOTAL // nproc       # number of models for each thread
remainder = N_MODELS_TOTAL - ( n_models * nproc )   # the remainder. e.g. your number of models may 

# arrays to store gamma ray and UHECR luminosities
# lgammas = np.zeros((len(sigmas), len(flux_scales), len(betas)))
# lcrs = np.zeros((len(sigmas), len(flux_scales), len(betas)))
# print (energies)

# little trick to spread remainder out among threads. If say you had 19 total models, and 4 threads
# then n_models = 4, and you have 3 remainder. This little loop would redistribute these three 
if remainder < my_rank + 1:
    my_extra = 0
    extra_below = remainder
else:
    my_extra = 1
    extra_below = my_rank

# where to start and end your loops for each thread
my_nmin = int((my_rank * n_models) + extra_below)
my_nmax = int(my_nmin + n_models + my_extra)

# total number you actually do
ndo = my_nmax - my_nmin


print ("This is thread {} calculating models {} to {}".format(my_rank, my_nmin, my_nmax))

# set barrier so print output doesn't look muddled
# just waits for other thread
MPI.COMM_WORLD.Barrier()

# start a timer for each thread
time_init = time.time()
tinit = time_init


for i in range(my_nmin, my_nmax):
    
    BETA = variable["betas"][i]
    flux_scale = variable["flux_scales"][i]
    SIGMA = variable["sigmas"][i]
    seed = variable["seeds"][i]
    tau_on = variable["lifetime"][i]
    savename = "run{}".format(variable["runid"][i])

    # get the lognormal parameters
    #mu = np.log(flux_scale)
    lognorm_params = (SIGMA,0,np.exp(np.log(1)))

    Length = int(tau_on / 0.1)
    
    # paramaters for lc are lognorm parameters, PSD parameters, tbin and Length (Age is really number of points)
    lc = sim.get_lc(lognorm_params, PSD_params, tbin, Length, RandomSeed=seed)
                      
    # normalise the light curve
    flux = lc.flux * flux_scale

    # this is roughly solar, but should probably be top heavy since it is easier to inject heavy ions, generally
    z = np.array([1,2,7,26])
    a = np.array([1,4,14,56])
    frac_elem = np.array([1.0,0.1,1e-4,3.16e-05]) * z * z * (a ** (BETA-2))
    elem = sim.get_elem_dict(beta = BETA)


    # NMAX 40,000 should limit array saves to under a GB in size
    ncr, escaping, lcr = sim.run_jet_simulation(energy_params, flux_scale, BETA, lc, tau_loss,
                                                elem=elem, plot_all=False, 
                                                sigma=SIGMA, R0=1e9, NRES = 20, NMAX=100000, 
                                                seed=seed, tau_on=tau_on, save_arrays=False,
                                                savename=savename)

    # get approximate gamma ray luminosity around 10 GeV
    select = (energies > 1e10) * (energies < 2e10)
    my_lgamma = np.fabs(np.trapz(energies[select] *unit.ev, energies[select]*ncr[0,-1,:][select]))

    # distance to Cen A
    distance = 3.7 * 3.086e18 * 1e6
    # lgammas[i_sigma,i_flux,i_beta] = my_lgamma * 1e-4 * 4.5e-26 * 0.5 * C / 4.0 / PI / distance / distance
    
    select = (energies > 6e19)

    # store the UHECR luminosity 
    # lcr = np.fabs(np.trapz(energies[select] * EV2ERGS, energies[select] * escaping[select]))
    # lcrs[i_sigma,i_flux,i_beta] = lcr
    lg = my_lgamma * 1e-4 * 4.5e-26 * 0.5 * unit.c / 4.0 / np.pi / distance / distance
    
    print ("sigma {:.1f} BETA {:.1f} median lum {:8.4e} mean lum {:8.4e} UHECR LUM: {:8.4e} {:8.4e}"
           .format(SIGMA, BETA, np.median(flux), np.mean(flux), lcr, lg))

    logfile.write("finished {} {} {}".format(SIGMA, BETA, flux_scale))


             
# get the time taken
time2 = time.time() 
time_tot = time2 - time_init

# set barrier so print output doesn't look muddled
if my_rank == 0: print ('Waiting for other threads to finish...')

# another barrier, wait for all to finish
MPI.COMM_WORLD.Barrier()

print ("Thread {} took {} seconds to calculate {} models".format(my_rank, time_tot, ndo))

# always call this when finishing up
MPI.Finalize()
