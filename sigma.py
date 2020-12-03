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
# import time
import pickle
from simulation import unit
from scipy.stats import lognorm
from jm_util import *
set_amy_defaults()


sigma = np.arange(0,3,0.01)
median = np.logspace(0,3,num=1000)
mu = np.log(1)
#mean = np.exp(mu + (sigma**2 / 2)) 

sf = 1.0 - np.array([lognorm.cdf(10, s, loc=mu, scale=np.exp(mu)) for s in sigma])

plt.figure(figsize=(7,5))
plt.plot(sigma, np.exp(mu + sigma**2 / 2), lw=3, label="Mean")
plt.plot(sigma, np.exp(mu - sigma*sigma / 2), lw=3, ls="--", label="Mode")
plt.semilogy()
plt.ylabel(r"Moment of jet power PDF ($\log_{10}$)", fontsize=18)
plt.xlabel(r"$\sigma$", fontsize=20)
plt.legend(loc=2, fontsize=16, frameon=False)
ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.plot(sigma, sf, c="k", lw=3, ls=":", label="Fraction of time $Q/Q_0>10$")
ax2.legend(fontsize=16, frameon=False)
ax2.set_ylabel("Fraction of time $Q/Q_0>10$", fontsize=18)
ax2.set_ylim(0,0.39)
ax2.set_yticks([0,0.1,0.2,0.3])
plt.xticks([0,1,2,3])

s = np.arange(-2,4)
ax1.set_yticks(10.0**s)
ax1.set_yticklabels([str(ss) for ss in s])
plt.xlim(0,3)
ax1.set_ylim(10.0**-2.2,10.0**2.2)
plt.subplots_adjust(top=0.98, left=0.1, right=0.9, bottom=0.12)
plt.savefig("sigma.png", dpi=300)

