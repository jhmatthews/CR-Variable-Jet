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

betas = [2,2.3,2.7]
sigmas = [0.5,1.3,2.2,3.0]
flux_scales = np.logspace(43,45,num=10)
flux_scale = []

my_lcr = np.zeros(1)
my_ls = np.zeros(1)
my_length = np.zeros(1)
my_width = np.zeros(1)
my_power = np.zeros(1)

for i_sigma, SIGMA in enumerate(sigmas):
	for i_flux, flux_scale in enumerate(flux_scales):
		for i_beta, BETA in enumerate(betas):
			logflux = np.log10(flux_scale)
			fname = "array_saves/jetstore_{:.1f}q{:.1f}sig{:.1f}.npy.pkl".format(BETA, logflux,  SIGMA)
			try:
				with open(fname, "rb") as pickle_file:
					j = pickle.load(pickle_file)


				units = [unit.kpc, unit.kpc, unit.c, 1e-6, unit.myr, unit.myr, 1e40, 1e32]
				variables = ["length", "width", "v_j", "B", "tesc", "tsync", "lcr", "lsync"]
				plt.figure()
				for i,v in enumerate (variables):
					y = getattr(j, v) / units[i]
					print (v)
					plt.plot(j.time/unit.myr, y, label=v)

				print (y)

				#pressure = j.jet.equation_of_state(0.1 * j.energy/j.volume)
				# plt.plot(j.time/unit.myr, j.volume * (pressure ** 1.8) / 1e7 / 1e9 / 1e23, label="sync")

				#plt.plot(j.jet.lc.time*unit.kyr/unit.myr, j.jet.lc.flux)
				plt.legend()
				# plt.plot(j.time/unit.myr, j.B/1e-6)
				plt.semilogy()
				plt.semilogx()
				plt.savefig("time_{:.1f}q{:.1f}sig{:.1f}.png".format(BETA, logflux,  SIGMA), dpi=200)
				plt.clf()

				my_lcr = np.concatenate((my_lcr, getattr(j, "lcr")))
				my_ls = np.concatenate((my_ls, 1e9 * getattr(j, "lsync")))
				my_length = np.concatenate((my_length, getattr(j, "length") / units[0]))
				my_width = np.concatenate((my_width, getattr(j, "width") / units[1]))
				power = getattr(j, "v_j")
				my_power = np.concatenate((my_power, power))

			except FileNotFoundError:
				print ("Error for {}".format(fname))
			except AttributeError:
				print ("AttributeError for {}".format(fname))

plt.close("all")