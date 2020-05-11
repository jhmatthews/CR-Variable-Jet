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

B = np.logspace(-7,-4,num=1000)

tsync_gev = 600.0 / 1e9 / unit.ev / (B**2)  / unit.myr
nu_obs = 1.4e9 / 0.29
gamma = np.sqrt(2.0 * MELEC * C * nu_obs / (3.0 * E * B))
energy = gamma * MELEC * C * C

Bcrit = 4e13 
tsync2 = 9.0 / 8.0 / PI * 137.0 * 137.0 * H * Bcrit * Bcrit / B / B / energy / unit.myr
tsync2 = 3.0 / THOMPSON * np.sqrt(2.0 * PI * MELEC * C / B / B / B) * (1.4e9)**-0.5 / unit.myr

tsync = (energy / EV2ERGS / 1e9) * tsync_gev

print (tsync2, tsync)

tesc_ref = 10.0 
tesc =  tesc_ref * (B / 1e-5)

#plt.plot(B/1e-6, tsync, label="1.4 Ghz", lw=3)
plt.plot(B/1e-6, tsync2, label="1.4 Ghz")
# plt.plot(B/1e-6, (1e9 / 1e10) * tsync_gev, label="10 Gev")
# plt.plot(B/1e-6, (1e9 / 1e8) * tsync_gev, label="100 Mev")
plt.plot(B/1e-6, tesc, label="10 EeV escape", lw=3)
#plt.plot(B/1e-6, energy/EV2ERGS/1e9, label="10 EeV escape")
plt.legend()
plt.xlabel(r"$B (\mu G)$")
plt.ylabel(r"$\tau(Myr)$")
plt.loglog()
plt.show()
plt.clf()

ts1 = 3.0 / THOMPSON * np.sqrt(2.0 * PI * MELEC * C * E) * (B**-1.5) * (1.4e9**-0.5) / unit.myr
ts2 = 3.0 / THOMPSON * np.sqrt(2.0 * PI * MELEC * C * E) * (B**-1.5) * (1.44e8**-0.5) / unit.myr
plt.plot(B/1e-6, ts1, label="1.4 Ghz", lw=3)
plt.plot(B/1e-6, tsync2, label="1.4 Ghz")
# plt.plot(B/1e-6, (1e9 / 1e10) * tsync_gev, label="10 Gev")
# plt.plot(B/1e-6, (1e9 / 1e8) * tsync_gev, label="100 Mev")
plt.plot(B/1e-6, ts2, label="144 MHz", lw=3)
plt.plot(B/1e-6, tesc, label="10 EeV escape", lw=3)
plt.xlabel(r"$B (\mu G)$")
plt.ylabel(r"$\tau(Myr)$")
plt.legend()
plt.loglog()
plt.show()
plt.clf()



