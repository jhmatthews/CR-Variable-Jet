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
import msynchro
from jm_util import *

def max_energy(power, v_over_c=0.1, eta=0.1):
    return 1e19*np.sqrt( (power/1e44) * (v_over_c/0.1))

def alpha_plot(j, plot_name):
	fig = plt.figure( figsize=(6,4) )
	ax3 = fig.add_subplot(111)
	alpha = np.log10(j.lsync/j.l144) / np.log10(1.44e8/1/4e9)
	ax3.plot(j.time/unit.myr, alpha, label=r"$\alpha$, 144Mhz to 1.4GHz")
	ax3.plot(j.time/unit.myr, alpha, label=r"$\alpha$, 144Mhz to 1.4GHz")
	ax3.set_ylim(0,1)
	#ax3.plot(j.time/unit.myr, np.log10(j.lcr), label=r"$\log(L_{\mathrm{uhecr}})$")
	#ax3.plot(j.time/unit.myr, np.log10(j.lsync*1.4e9), label=r"$\log(L_{1400})$")
	#ax3.plot(j.time/unit.myr, np.log10(j.l144*1.44e8), label=r"$\log(L_{144})$")
	#print (j.v_j)

	try:
		power = j.power
	except AttributeError:
		power = 0.5 * 1e-4 * unit.mprot * (j.v_j**3) * unit.kpc * unit.kpc * 0.5 * 0.5 

	ax3.plot(j.time/unit.myr, power/np.max(power), label=r"$Q_j/Q_{\mathrm{max}}$")
	ax3.legend()
	ax3.set_xlabel("t (Myr)", fontsize=16)
	ax3.set_title("Luminosities")
	plt.subplots_adjust(left=0.05, right=0.96)
	
	fig.savefig("alpha_plots/alphaplot_{}.png".format(plot_name), dpi=200)
	fig.clf()

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

energies = np.logspace(6,21,num=3000)

betas = [2,2.3,2.7]
sigmas = np.linspace(0,2,num=5)
flux_scales = np.logspace(43,45,num=10)
flux_scales = [1e44]
sigmas = [1.5]
betas = [2]
flux_scales = [1e44]
sigmas = [1.5]
betas = [2]
seeds = [12,38,100,200]
seeds = [38]

flux_scale = []


my_lcr = np.zeros(1)
my_ls = np.zeros(1)
my_length = np.zeros(1)
my_width = np.zeros(1)
my_power = np.zeros(1)
my_ls = []
my_length = []


#times = 
frequencies = np.logspace(8,12,400)
nspectra = 30
spectra = np.zeros( (nspectra, len(frequencies)) )
ispec = np.random.randint(1500, size=nspectra)

energies = np.logspace(7,14,3000)
select = (energies < 1e14)


for i_sigma, SIGMA in enumerate(sigmas):
	for i_flux, flux_scale in enumerate(flux_scales):
		for i_beta, BETA in enumerate(betas):

			for seed in seeds:
				logflux = np.log10(flux_scale)
				fname = "array_saves/jetstore_beta{:.1f}q{:.1f}sig{:.1f}seed{:d}.npy.pkl".format(BETA, logflux,  SIGMA, seed)
				with open(fname, "rb") as pickle_file:
					j = pickle.load(pickle_file)

				# make the mesh plot with the remaining CRs 
				ncr = load_one_npz("array_saves/ncr_beta{:.1f}q{:.1f}sig{:.1f}seed{:d}.npz".format(BETA, logflux,  SIGMA, seed))
				#ncr = np.load("array_saves/ncr_beta{:.1f}q{:.1f}sig{:.1f}seed{:d}.npy".format(BETA, logflux,  SIGMA, seed))
			
				for i in range(nspectra):
					Bfield = j.B[i]
					spectra[i,:] = msynchro.Ptot(frequencies, energies[select], ncr[0,i,:][select], Bfield)
					print (i)

				#alpha = k.a


#normalised
#for NUSE in range(100):
NUSE = 25
plt.figure(figsize=(10,5))

plt.subplot(121)
lognu = np.log10(frequencies)
for i in range(nspectra):
	normalised = spectra[i]
	normalised = spectra[i] * np.sqrt(frequencies)
	time = j.time[i]
	if i == NUSE:
		plt.plot(lognu, normalised/np.max(normalised), alpha=1, lw=4, zorder=3, c="C3")
		print (time/unit.myr)
	else:
		plt.plot(lognu, normalised/np.max(normalised), alpha=0.4, c="k", zorder=1)


indices = [0,-0.5,-1]
colors = ["C1", "C2", "C4"]
for i, n in enumerate(indices):
	spec_nu_1 = frequencies ** n
	plt.plot(lognu, spec_nu_1/np.max(spec_nu_1), ls="--", label=r"$F_\nu \propto \nu^{{{:.1f}}}$".format(n-0.5), color=colors[i])
plt.semilogy()
plt.xlabel(r"$\log(\nu)$ (Hz)", fontsize=18)
plt.ylabel(r"$\nu^{1/2}~F_\nu$ (Normalised, Arb.)", fontsize=18)
plt.xlim(8,11.99)
plt.ylim(1e-3,1.5)
plt.vlines([np.log10(1.4e9),np.log10(1.44e8)], 1e-3, 1.5, label="Spectral index points", color="C0", ls="-.")
plt.legend(fontsize=16)

alpha = np.log10(j.lsync/j.l144) / np.log10(1.44e8/1.4e9)
plt.subplot(122)
plt.plot(j.time/unit.myr, alpha, label=r"$\alpha$", lw=3)
jj = ispec[NUSE]
plt.scatter(j.time[jj]/unit.myr, alpha[jj], c="C3", zorder=4)
plt.plot(j.time/unit.myr, j.power/np.max(j.power), label=r"$Q_j/Q_j|_{\mathrm{max}}^{-1}$", lw=2)
plt.legend(fontsize=16)
plt.xlabel("$t$ (Myr)", fontsize=18)
plt.subplots_adjust(right=0.98, top=0.98,bottom=0.13, left=0.1, hspace=0.1, wspace=0.1)
plt.savefig("sync_spectra_{}.png".format(NUSE), dpi=200)


