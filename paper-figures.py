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
import sys

def cooling_time(nu, B):
	'''
	returns time in Myr. B in Gauss
	'''
	tau = 40.97942498255395 * ((B/1e-5) ** -1.5) * np.sqrt(1e9/nu)
	return tau


def escape_time(rigidity, B, R):
	'''
	returns time in Myr. B in Gauss
	'''
	tau = 9.05 * (R/100.0)**2 * (rigidity / 10.0) * (B/1e-5)
	return tau

def Hillas(beta, B, R):
	'''
	Hillas energy in EeV 
	'''
	rigidity = 0.9 * (B / 1e-6) * R * beta
	return (rigidity)

def cooling_plot():
	plt.figure(figsize=(7,5))
	nu = 1e9
	B = np.logspace(-7,-4,1000)
	tau_sync = cooling_time(nu, B)
	plt.plot(B/1e-6, tau_sync, label=r"$\tau_{\mathrm{sync}}^e$", c="C0", ls="--")

	rigidity = 10.0
	R = 100.0
	tau_esc1 = escape_time(rigidity, B, R)
	plt.plot(B/1e-6, tau_esc1, c="C1", ls="-", label=r"$\tau_{\mathrm{esc}}^{\mathrm{cr}},~E/Z=10\mathrm{EV}$")

	rigidity = Hillas(0.1, B, R)
	tau_esc2 = escape_time(rigidity, B, R)
	plt.plot(B/1e-6, tau_esc2, c="C1", ls=":", label=r"$\tau_{\mathrm{esc}}^{\mathrm{cr}}$, Lobe Hillas criterion")


	plt.xlim(0.5,100)
	plt.ylim(0.1,999)
	plt.xlabel(r"$B~(\mu{\rm G})$", fontsize=18)
	plt.ylabel(r"$\tau~({\rm Myr})$", fontsize=18)
	plt.loglog()
	plt.legend(fontsize=16)
	plt.subplots_adjust(right=0.96,top=0.98,bottom=0.13,left=0.11)
	plt.savefig("timescales.png", dpi=200)


def max_energy(power, v_over_c=0.1, eta=0.1):
    return 1e19*np.sqrt( (power/1e44) * (v_over_c/0.1))


def three_panel_cr(time, energies, ncr, escaping, folder):

	# first, let's make integrated spectra
	ncr_tot = np.sum(ncr, axis=1) # integrated over time 
	escape_tot = np.sum(escaping, axis=1) # integrated over time 
	ncr_tot_all = energies * energies * np.sum(ncr_tot[1:], axis=0)
	escape_tot_all = energies * energies * np.sum(escape_tot[1:], axis=0)



	fig = plt.figure( figsize=(11,5))
	ax1 = fig.add_subplot(131)
	ax2 = fig.add_subplot(132)
	ax3 = fig.add_subplot(133)
	ncr_max = np.max(ncr_tot_all[energies>=1e17])
	escape_max = np.max(escape_tot_all[energies>=1e17])
	ion_labels = ["Protons", "He", "N", "Fe"]
	logE = np.log10(energies)
	for i in range(1,5):
		ax1.plot(logE, energies * energies * ncr_tot[i]/ncr_max, label=ion_labels[i-1])
		ax2.plot(logE, energies * energies * escape_tot[i]/escape_max, label=ion_labels[i-1])	

	ax1.plot(logE, ncr_tot_all/ncr_max, c="k", lw=4, label="Total", alpha=0.7)
	ax2.plot(logE, escape_tot_all/escape_max, c="k", lw=4, label="Total", alpha=0.7)
	ax1.legend()
	ax2.legend()
	ax1.set_title("Inside lobe, time averaged")
	ax2.set_title("Escaping, time averaged")

	times = np.random.randint(0,len(time),size=30)
	ncr_time_sum = np.sum(ncr[1:], axis=0)
	escape_time_sum = np.sum(escaping[1:], axis=0)
	for i,t in enumerate(times):
		print (energies.shape, ncr_time_sum.shape, ncr_time_sum[t].shape)
		ncr_plot = energies * energies * ncr_time_sum[t]
		if i == 0:
			labels = ("Inside Lobe", "Escaping")
		else:
			labels = (None, None)

		ax3.plot(logE, ncr_plot/np.max(ncr_plot[energies>=1e17]), c="C0", alpha=0.5, zorder=2, label=labels[0])
		escaping_plot = energies * energies * escape_time_sum[t]
		ax3.plot(logE, escaping_plot/np.max(escaping_plot[energies>=1e17]), c="C1", alpha=0.5, zorder=1, label=labels[1])

	ax3.legend()
	ax3.set_title("Spectra at random times")
	for ax in [ax1,ax2,ax3]:
		ax.set_xlim(17,20.5)
		ax.set_ylim(1e-3,2)
		ax.set_yscale("log")
	ax2.set_yticklabels([])
	ax3.set_yticklabels([])
	ax1.set_ylabel("$E^2 dN/dE$ (Normalised)", fontsize=18)
	ax2.set_xlabel(r"$\log[E ({\rm eV})]$", fontsize=18)
	plt.subplots_adjust(hspace=0.05,wspace=0.05,right=0.98,top=0.94, left=0.08, bottom=0.13)
	fig.savefig("{}/cr-3panel.png".format(folder), dpi=200)

	return (fig)

def standard_plot(j, folder):

	tmax = np.max(j.time/unit.myr)

	fig = plt.figure( figsize=(11,5) )

	# First make timescales panel 
	ax1 = fig.add_subplot(131)
	ax1.set_title("Timescales (Myr)")
	tsync = np.sqrt(3.0 / THOMPSON / THOMPSON * 2.0 * PI * MELEC * C * E / 1.4e9 / (j.B**3)) 
	ax1.plot(j.time/unit.myr, np.log10(j.tesc/unit.myr), label=r"$\tau_\mathrm{esc}$")
	ax1.plot(j.time/unit.myr, np.log10(j.tsync/unit.myr), label=r"$\tau_\mathrm{sync}$")
	#ax1.plot(j.time/unit.myr, np.log10(j.length/j.v_advance/unit.myr), label=r"$\tau_\mathrm{adv}$")
	ax1.plot(j.time/unit.myr, np.log10(j.length/j.jet.cs_env/unit.myr), label=r"$\tau_\mathrm{cs}$")
	ax1.plot(j.time/unit.myr, np.log10(j.time/unit.myr), label=r"$\tau_\mathrm{age}$")
	ax1.legend()
	ax1.set_title("Timescales (Myr)")
	ax1.set_ylim(0,3)
	ax1.set_xlim(0,tmax)
	#ax1.set_xlabel("t (Myr)")
	ax1.set_yticks([-1,0,1,2,3])
	ax1.set_yticklabels(["0.1", "1", "10", "100","1000"])




	# calculate some important quantities 
	nprot = j.density / unit.mprot
	print (j.density, j.volume, unit.mprot)
	#print (j.B)
	#print (j.v_j/unit.c, j.gmm, j.B/1e-6, j.length/unit.kpc)
	#rho_0 = [universal_profile(r, j.jet.M500_msol)[2] for r in j.length]
	#print (rho_0)
	# eta_star = j.gmm * j.gmm * j.jet.rho_j / rho_0 
	# factor = np.sqrt(eta_star) / (np.sqrt(eta_star) + 1)
	# v_a_theory = factor * j.v_j
	# v_a_theory = np.gradient(j.length, j.time) / unit.c
	# print (v_a_theory)

	# rho_j = j.power / (j.gmm * (j.gmm - 1.) * j.jet.area * unit.c * unit.c * j.v_j)
	# print (np.mean(j.power), np.exp(np.log(1e43)+2.0))
	# print (np.mean(j.v_j/unit.c), np.mean(j.gmm))

	#print (rho_j / unit.mprot, np.median(j.power))
	tf2 = 220.0 * ((j.B / 1e-6)**(-3)) * ((nprot/1e-6))
	#plt.plot(j.time/unit.myr, np.log10(tf2), label=r"$\tau_\mathrm{F2}$")

	# Plot physical quantities 
	ax2 = fig.add_subplot(132)
	U = j.energy / j.volume
	ax2.plot(j.time/unit.myr, np.log10(j.B/1e-6), label=r"$\log[B (\mu$G)]")
	ax2.plot(j.time/unit.myr, np.log10(j.gmm), label=r"$\log(\Gamma_j)$")
	ax2.plot(j.time/unit.myr, np.log10(nprot/1e-6), label=r"$\log[n~({10^{-6}\rm cm}^{-3})]$")
	ax2.plot(j.time/unit.myr, np.log10(0.1*j.pressure/1e-13), label=r"$\log[P~({10^{-13}\rm Pa})]$")

	ax2.set_ylim(-2,3)
	ax2.set_xlim(0,tmax)
	ax2.legend()
	ax2.set_title("Physical Quantities")
	ax2.set_xlabel("t (Myr)", fontsize=16)


	ax3 = fig.add_subplot(133)
	ax3.plot(j.time/unit.myr, np.log10(j.length/unit.kpc), label=r"Length (kpc)")
	ax3.plot(j.time/unit.myr, np.log10(j.width/unit.kpc), label=r"Width (kpc)")
	ax3.plot(j.time/unit.myr, np.log10(j.length/j.width), label=r"Aspect")
	# ax2.plot(j.time/unit.myr, np.log10( 10.0*(j.time/unit.myr)**0.6), label=r"Theory")
	#ax2.set_yscale("log")
	ax3.legend()
	

	ax3.set_title("Dimensions")
	ax3.set_ylim(0,2.5)
	ax3.set_xlim(0,tmax)
	ax3.set_yticks([0,1,2])
	ax3.set_yticklabels(["1", "10", "100"])

	plt.subplots_adjust(left=0.045, hspace=0.12, wspace=0.12, right=0.98, top=0.94, bottom=0.13)
	
	fig.savefig("{}/time-evolution.png".format(folder), dpi=200)
	fig.clf()


def luminosities_plot(j, folder):
	plt.figure(figsize=(9,5))
	plt.plot(j.time/unit.myr, np.log10(j.power), label=r"$\log(Q_{j})$", c="k", alpha=0.8)
	plt.plot(j.time/unit.myr, np.log10(j.lcr), label=r"$\log(L_{\mathrm{cr,60EeV}})$", c="C1")
	plt.plot(j.time/unit.myr, np.log10(j.lcr_8EeV), label=r"$\log(L_{\mathrm{cr,8EeV}})$", c="C3")
	plt.plot(j.time/unit.myr, np.log10(j.lsync*1.4e9), label=r"$\log(\nu L_{1400})$", c="C0")
	plt.plot(j.time/unit.myr, np.log10(j.l144*1.44e8), label=r"$\log(\nu L_{144})$", c="C2")
	plt.xlabel("t (Myr)", fontsize=16)
	plt.legend(loc=2, fontsize=14)
	plt.ylabel(r"$\log$[Power or Luminosity (erg~s$^{-1}$)]", fontsize=16)
	plt.xlim(0,np.max(j.time/unit.myr))
	plt.subplots_adjust(left=0.1, right=0.98, top=0.94, bottom=0.13)
	plt.savefig("{}/luminosity.png".format(folder), dpi=200)




	

plt.rcParams["text.usetex"] = "True"
plt.rcParams['xtick.labelsize']=14
plt.rcParams['ytick.labelsize']=14
plt.rcParams['font.serif']=['cm']
plt.rcParams['font.family']='serif'	
plt.rcParams["text.usetex"] = "True"
plt.rcParams["lines.linewidth"] = 3
plt.rcParams["axes.linewidth"] = 2
plt.rcParams["xtick.major.width"] = 1.5
plt.rcParams["ytick.major.width"] = 1.5


def make_all_plots(sigmas, flux_scales, betas, seeds, nenergies=3000):

	cooling_plot()
	cr_energies = np.logspace(14,21,num=nenergies)
	e_energies = np.logspace(7,14,num=nenergies)
	for i_sigma, SIGMA in enumerate(sigmas):
		for i_flux, flux_scale in enumerate(flux_scales):
			for i_beta, BETA in enumerate(betas):
				for seed in seeds:

					folder = "paper-figures-{}-p{}".format(seed, BETA)
					os.system("mkdir -p {}".format(folder))
					#print ("test")
					logflux = np.log10(flux_scale)
					fname = "array_saves/jetstore_{:.1f}q{:.1f}sig{:.1f}seed{:d}.npy.pkl".format(BETA, logflux,  SIGMA, seed)
					with open(fname, "rb") as pickle_file:
						j = pickle.load(pickle_file)

					# load escaping spectra  
					escaping = np.load("array_saves/escaping_beta{:.1f}q{:.1f}sig{:.1f}seed{:d}.npy".format(BETA, logflux,  SIGMA, seed))
					
					# load particle spectra 
					ncr = np.load("array_saves/ncr_beta{:.1f}q{:.1f}sig{:.1f}seed{:d}.npy".format(BETA, logflux,  SIGMA, seed))


					fname = "beta{:.1f}q{:.1f}sig{:.1f}seed{:d}".format(BETA, logflux,  SIGMA, seed)

					three_panel_cr(j.time / unit.myr, cr_energies, ncr, escaping, folder)

					standard_plot(j, folder)

					luminosities_plot(j, folder)



if __name__ == "__main__":
	power = 1e44
	seed = int(sys.argv[1])
	beta = float(sys.argv[2])
	sigma = 1.5
	make_all_plots([sigma], [power], [beta], [seed])





