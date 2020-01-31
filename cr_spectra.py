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

def max_energy(power, v_over_c=0.1, eta=0.1):
    return 1e19*np.sqrt( (power/1e44) * (v_over_c/0.1))

def dual_mesh_plot(time, energies, ncr, escaping, BETA, logflux, SIGMA, escape_energy, hubble_energy, Emax):


	time2 = time

	plt.figure(figsize=(7,5))
	plt.subplot(211)
	ncr_sum = np.sum(ncr[1:], axis=0)
	ncr_sum[(ncr_sum == 0)] = 1e-99
	plt.pcolormesh(time2, np.log10(energies), np.log10(energies * energies * ncr_sum[1:,:]).T, vmin=64)
	plt.ylim(15.1,20.5)
	cbar = plt.colorbar()
	cbar.set_label(r"$\log (E^2~J)$")
	plt.plot(time, escape_energy, c="C3", ls="--", label=r"$E(\tau_{\mathrm{esc}}=1$Myr), protons")
	plt.plot(time, hubble_energy, c="C3", ls="-.")
	plt.plot(time, Emax, c="C3", alpha=0.5, label=r"$E_{\mathrm{max}}$, protons")
	plt.legend()
	plt.text(10,16,"CRs Inside Lobe", c="w", fontsize=14)
	plt.gca().set_xticklabels([])
	plt.ylabel(r"$\log [E$(eV)]", fontsize=14)

	escape_sum = np.sum(escaping[1:], axis=0)
	escape_sum[(escape_sum == 0)] = 1e-99
	plt.subplot(212)
	plt.pcolormesh(time2, np.log10(energies), np.log10(energies * energies * escape_sum[1:,:]).T, vmin=50)
	plt.ylim(15.1,20.5)
	cbar = plt.colorbar()
	cbar.set_label(r"$\log (E^2~J)$")
	plt.plot(time, escape_energy, c="C3", ls="--")
	plt.plot(time, hubble_energy, c="C3", ls="-.")
	plt.plot(time, Emax, c="C3", alpha=0.5)
	plt.text(10,16,"Escaping CRs", c="w", fontsize=14)


	plt.xlabel("$t$ (Myr)", fontsize=16)
	plt.ylabel(r"$\log [E$(eV)]", fontsize=14)
	plt.subplots_adjust(hspace=0.05,wspace=0.05,right=1, top=0.98, left=0.1)
	plt.savefig("cr_spectra/dual_mesh_beta{:.1f}q{:.1f}sig{:.1f}.png".format(BETA, logflux, SIGMA), dpi=200)
	plt.clf()
	return (ncr_sum, escape_sum)

def mesh_plot(time, energies, ncr, BETA, logflux, SIGMA, escape_energy, hubble_energy, Emax, prefix, vmin=50):

	time2 = np.linspace(0,np.max(time), num=len(ncr[0,:]))
	time2 = time
	ncr_sum = np.sum(ncr[1:], axis=0)
	ncr_sum[(ncr_sum == 0)] = 1e-99
	plt.pcolormesh(time2, np.log10(energies), np.log10(energies * energies * ncr_sum[1:,:]).T, vmin=vmin)
	plt.ylim(15.1,20.5)
	plt.colorbar()

	plt.plot(time, escape_energy, c="C3", ls="--")
	plt.plot(time, hubble_energy, c="C3", ls="-.")
	plt.plot(time, Emax, c="C3", alpha=0.5)
	plt.xlabel("$t$ (Myr)", fontsize=16)
	plt.ylabel(r"$\log [E$(eV)]", fontsize=16)
	plt.savefig("cr_spectra/{}_mesh_beta{:.1f}q{:.1f}sig{:.1f}.png".format(prefix, BETA, logflux, SIGMA), dpi=200)
	plt.clf()
	return (ncr_sum)

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
sigmas = np.linspace(0.5,2,num=4)
flux_scales = np.logspace(43,45,num=10)
flux_scales = [1e43]
sigmas = [2.0]
betas = [2]

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



			escape_energy = np.log10((1e-6 / j.B) * (j.tesc/unit.myr) * 1e19)
			hubble_energy = np.log10((1e-6 / j.B) * (0.1*1.38e10/unit.myr) * 1e19)
			print (hubble_energy)
			jet_power = 0.5 * 1e-4 * unit.mprot * j.v_j * j.v_j * j.v_j  * (0.5 * unit.kpc)**2 
			print (j.v_j)
			Emax = np.log10(max_energy(jet_power, v_over_c=(j.v_j/unit.c)))
			print (Emax)

			escaping = np.load("array_saves/escaping_beta{:.1f}q{:.1f}sig{:.1f}.npy".format(BETA, logflux, SIGMA))
			
			# make the mesh plot with the escaping CRs 
			#escaping_sum = mesh_plot(j.time / unit.myr, energies, escaping, BETA, logflux, SIGMA, escape_energy, hubble_energy, Emax, "esc", vmin=50)
		
			# make the mesh plot with the remaining CRs 
			ncr = np.load("array_saves/ncr_beta{:.1f}q{:.1f}sig{:.1f}.npy".format(BETA, logflux, SIGMA))
			#ncr_sum = mesh_plot(j.time / unit.myr, energies, ncr, BETA, logflux, SIGMA, escape_energy, hubble_energy, Emax, "ncr", vmin=64)

			ncr_sum, escaping_sum = dual_mesh_plot(j.time / unit.myr, energies, ncr, escaping, BETA, logflux, SIGMA, escape_energy, hubble_energy, Emax)
			print (j.time.shape, ncr.shape)

			# now make combined spectra plots 
			NTIMES = 4
			print (len(escaping[0,:,:]))
			times = np.random.randint(low=10, high=len(escaping[0,:,:]), size=NTIMES)
			#times = [150,200,250,300]
			plt.figure(figsize=(8,4))
			iarg = np.argmin(np.fabs(energies - 1e17))

			sums = (ncr_sum, escaping_sum)
			spectra = (ncr, escaping)

			iplot = 0
			for icr in range(2):
				ncr_use = spectra[icr]
				sum_use = sums[icr]

				for i in times:
					plt.subplot(2,4,iplot+1)
					iplot+=1

					#maximum = sum_use[i,:].T * 1e34
					#E0 = np.max( maximum )
					Ecubed = energies ** 2
					maximum = sum_use[i,:].T * Ecubed
					E0 = np.max( maximum )

					J = Ecubed * sum_use[i,:].T

					plt.plot(np.log10(energies), np.log10(J/E0), label=i, c="k", lw=4, alpha=0.7)
					for n in range(4):
						J = Ecubed * ncr_use[n+1,i,:].T
						#E0 = J[iarg]
						#E0 = 1.0
						# if icr == 0:
							# plt.title("t")
						plt.plot(np.log10(energies), np.log10(J/E0), label=i, alpha=0.7, c="C"+str(n))
						if iplot > 1 and iplot != 5: 
							plt.gca().set_yticklabels([])
						if iplot < 5: 
							plt.gca().set_xticklabels([])
							plt.title("$t={:.1f}$Myr".format(j.time[i]/unit.myr))


					plt.ylim(-6,0.5)
					plt.xlim(15.1,20.49)

			plt.text(0.45,0.03, r"$\log [E$(eV)]", fontsize=16, transform=plt.gcf().transFigure)
			plt.text(0.03,0.4, r"$\log [E^2~J$(Normalised)]", fontsize=14, transform=plt.gcf().transFigure, rotation=90)
			plt.text(0.91,0.55, r"Inside Lobe", fontsize=14, transform=plt.gcf().transFigure, rotation=270)
			plt.text(0.91,0.25, r"Escaping", fontsize=14, transform=plt.gcf().transFigure, rotation=270)
			plt.subplots_adjust(hspace=0, wspace=0, bottom=0.15)
			plt.savefig("cr_spectra/ncr_beta{:.1f}q{:.1f}sig{:.1f}.png".format(BETA, logflux, SIGMA), dpi=200)
			plt.clf()




