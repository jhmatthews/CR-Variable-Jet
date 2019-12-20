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

def max_energy(power, v_over_c=0.1, eta=0.1):
    return 1e19*np.sqrt( (power/1e44) * (v_over_c/0.1))

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

energies = np.logspace(6,20.5,num=3000)

betas = [2,2.3,2.7]
sigmas = [0.5,1.3,2.2,3.0]
flux_scales = np.logspace(43,45,num=10)
flux_scale = []

my_lcr = np.zeros(1)
my_ls = np.zeros(1)
my_length = np.zeros(1)
my_width = np.zeros(1)
my_power = np.zeros(1)
my_ls = []
my_length = []

fig_all = plt.figure()

for i_sigma, SIGMA in enumerate(sigmas):
	for i_flux, flux_scale in enumerate(flux_scales):
		for i_beta, BETA in enumerate(betas):
			#print ("test")
			logflux = np.log10(flux_scale)
			fname = "array_saves/jetstore_{:.1f}q{:.1f}sig{:.1f}.npy.pkl".format(BETA, logflux,  SIGMA)
			with open(fname, "rb") as pickle_file:
				j = pickle.load(pickle_file)


			time = j.time / unit.myr
			escape_energy = np.log10((1e-6 / j.B) * (j.tesc/unit.myr) * 1e19)
			jet_power = 1e-3 * unit.mprot * j.v_j * j.v_j * (1.0 * unit.kpc)**2 
			Emax = np.log10(max_energy(jet_power, v_over_c=j.v_j))
			print (escape_energy)

			escaping = np.load("array_saves/escaping_beta{:.1f}q{:.1f}sig{:.1f}.npy".format(BETA, logflux, SIGMA))
			select = (energies > 1e18)
			escaping = np.sum(escaping[1:], axis=0)
			#time = np.arange(len(escaping[:,0]))
			escaping[(escaping == 0)] = 1e-99
			plt.pcolormesh(time, np.log10(energies), np.log10(energies * energies * escaping[1:,:]).T, vmin=50)
			plt.ylim(17,20.5)
			plt.colorbar()

			plt.plot(time, escape_energy, c="C3", ls="--")
			plt.plot(time, Emax, c="C3", alpha=0.5)
			plt.savefig("esc_mesh_beta{:.1f}q{:.1f}sig{:.1f}.png".format(BETA, logflux, SIGMA), dpi=200)

			plt.clf()
			escaping = np.load("array_saves/ncr_beta{:.1f}q{:.1f}sig{:.1f}.npy".format(BETA, logflux, SIGMA))
			select = (energies > 1e18)
			escaping = np.sum(escaping[1:], axis=0)
			#time = np.arange(len(escaping[:,0]))
			escaping[(escaping == 0)] = 1e-99
			plt.pcolormesh(time, np.log10(energies), np.log10(energies * energies * escaping[1:,:]).T, vmin=65)
			plt.ylim(17,20.5)
			plt.colorbar()
			plt.plot(time, escape_energy, c="C3", ls="--")
			plt.plot(time, Emax, c="C3", alpha=0.5)
			plt.savefig("ncr_mesh_beta{:.1f}q{:.1f}sig{:.1f}.png".format(BETA, logflux, SIGMA), dpi=200)



