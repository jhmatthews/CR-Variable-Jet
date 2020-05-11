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

plt.rcParams["text.usetex"] = "True"
plt.rcParams['xtick.labelsize']=14
plt.rcParams['ytick.labelsize']=14
plt.rcParams['font.serif']=['cm']
plt.rcParams['font.family']='serif'	
plt.rcParams["text.usetex"] = "True"
plt.rcParams["lines.linewidth"] = 2.5
plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams["xtick.major.width"] = 1.5
plt.rcParams["ytick.major.width"] = 1.5

sigma = np.arange(0,3,0.01)
median = np.logspace(42,45,num=1000)
mu = np.log(1e43)
#mean = np.exp(mu + (sigma**2 / 2)) 

sf = 1.0 - np.array([lognorm.cdf(1e44, s, loc=mu, scale=np.exp(mu)) for s in sigma])

plt.figure(figsize=(7,5))
plt.plot(sigma, np.exp(mu + sigma**2 / 2), lw=3, label="Mean")
plt.plot(sigma, np.exp(mu - sigma*sigma / 2), lw=3, ls="--", label="Mode")
plt.semilogy()
plt.ylabel("Moment of jet power distribution", fontsize=14)
plt.xlabel(r"$\sigma$", fontsize=18)
plt.legend(loc=2, fontsize=14)
ax2 = plt.gca().twinx()
ax2.plot(sigma, sf, c="k", ls=":", label="Fraction of time $Q>10^{44}$erg/s")
ax2.legend(fontsize=14)
ax2.set_ylabel("Fraction of time $Q_j>10^{44}$erg/s", fontsize=14)
ax2.set_ylim(0,0.4)
plt.xticks([0,1,2,3])
plt.xlim(0,3)
plt.subplots_adjust(top=0.98, left=0.1, right=0.9)
plt.savefig("sigma.png", dpi=300)

