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
import simulation as sim
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from jm_util import *


LINESTYLES = [(0,(3,1,1,1)), "-", (0,(5,1)), (0,(1,0.5))]

def cooling_time(nu, B):
	'''
	returns time in Myr. B in Gauss
	'''
	tau = 40.97942498255395 * ((B/1e-5) ** -1.5) * np.sqrt(1e9/nu)
	return tau


def escape_time(rigidity, B, R, eta_bohm=1.0):
        '''
        returns time in Myr. B in Gauss
        '''
        tau = 9.05 * (R/100.0)**2 * (rigidity / 10.0) * (B/1e-5) * (1.0/eta_bohm)
        return tau

def Hillas(beta, B, R):
	'''
	Hillas energy in EeV 
	'''
	rigidity = 0.9 * (B / 1e-6) * R * beta
	return (rigidity)


def max_energy(power, v_over_c=0.1, eta=0.1):
    return 1e19*np.sqrt( (power/1e44) * (v_over_c/0.1))

def lc_plot(j, folder):

	lc_color="k"
	alpha = 0.8

	fig = plt.figure(figsize=(8.75,5.25))
	ax = fig.add_subplot(111)
	ax.set_ylabel(r"$\log[Q_j~(\mathrm{erg~s}^{-1})]$", fontsize=18)
	ax.set_xlabel(r"$t~(\mathrm{Myr})$", fontsize=20)
	#ax.plot(j.time/unit.myr, np.log10(j.v_j/unit.c), c=lc_color, lw=1.5, alpha=alpha)
	ax.plot(j.time/unit.myr, np.log10(j.power), c=lc_color, lw=2, alpha=alpha)
	#print (j.v_j/unit.c)
	
	#ax.legend(loc=4)
	#ax.set_yscale("log")
	ax.set_xlim(0,np.max(j.time/unit.myr))
	ax.set_ylim(43,47.8)
	plt.xticks(fontsize=18)
	plt.yticks(fontsize=18)

	ax2 = ax.twinx()
	print (j.jet.eps_b)
	logE1 = 19 + (0.5 * np.log10(1e43/1e44)) + np.log10(0.3) - (0.5 * np.log10(j.jet.eps_b / 0.1))
	logE2 = 19 + (0.5 * np.log10(1e48/1e44)) + np.log10(0.3) - (0.5 * np.log10(j.jet.eps_b/ 0.1))
	ax2.set_ylim(logE1, logE2)
	ax2.set_ylabel(r"$\log[\beta_j^2~E_{\rm max}/Z~(\mathrm{eV})]$",fontsize=18)
	ax2.grid(ls=":", lw=1.5)
	plt.yticks(fontsize=18)

	#plt.subplots_adjust(right=0.98, left=0.1, top=0.98)

	# A,v_bend,a_low,a_high,c = 1, 100, 1, 20, 0
	# ax = plt.gca()
	# axins = inset_axes(ax, width="50%", height="70%",bbox_to_anchor=(.08, .65, .4, .4),bbox_transform=ax.transAxes, loc=3)
	# freq = np.logspace(-6,0,num=10000)
	# axins.plot(freq * 1000.0, BendingPL(freq, 1000.0,v_bend,a_low,a_high,c), c=lc_color, alpha=alpha, lw=3)
	# axins.set_xlabel(r"$\nu~(\mathrm{Myr}^{-1})$",fontsize=14)
	# #axins.set_ylabel(r"$\mathrm{Power~(Arb.)}$")
	# axins.set_ylim(1e-10,1e10)
	# axins.set_xlim(1e-2,10)
	# plt.loglog()
	# plt.title(r"$\mathrm{PSD}$", fontsize=14)

	axins = inset_axes(ax, width="50%", height="70%",bbox_to_anchor=(.1, .6, .4, .4),bbox_transform=ax.transAxes, loc=3)
	#axins = inset_axes(ax, width="50%", height="70%",bbox_to_anchor=(.75, .65, .4, .4),bbox_transform=ax.transAxes, loc=3)
	freq = np.logspace(-6,0,num=10000)
	#bins = np.arange(-2,0,0.1)
	bins = np.arange(43.1,47.2,0.2)
	axins.hist(np.log10(j.power), bins=bins, color=lc_color, alpha=alpha, density=True)
	axins.set_xlim(43,47)
	axins.set_xticks([43,45,47])
	#axins.set_xlabel(r"$\log[Q_j~(\mathrm{erg~s}^{-1})]$",fontsize=14)
	#axins.set_ylabel(r"$\mathrm{Power~(Arb.)}$")
	#ax3 = axins.twinx()
	plt.title(r"Histogram of $\log Q_j$", fontsize=14)
	#plt.title(r"$\mathrm{Histogram}$", fontsize=14)
	plt.subplots_adjust(top=0.98, right=0.9, left=0.08, bottom=0.13)

	
	fig.savefig("{}/lc.png".format(folder), dpi=200)
	fig.clf()


def lnA_plot(energies, ncr2, escaping2, folder):

	#print (ncr2)
	ncr = ncr2[1:,:,:]
	#print (ncr, ncr2)
	escaping = escaping2[1:,:,:]
	#print (ncr.shape, escaping.shape)
	ncr_tot = np.sum(ncr2, axis=1) # integrated over time 
	escape_tot = np.sum(escaping2, axis=1) # integrated over time 
	elems = sim.get_elem_dict(beta = 2)

	# # lnAa = np.ones_like(ncr) * np.log(elems["a"])
	loga = np.log(elems["a"])
	lnA = np.zeros_like(ncr)
	lnA_esc = np.zeros_like(escaping)
	for i in range(len(loga)):
		lnA[i,:,:] = loga[i] * ncr[i,:,:]
		lnA_esc[i,:,:] = loga[i] * escaping[i,:,:]


	lnA = np.sum(lnA, axis=0) / np.sum(ncr, axis=0) # integrated over A 
	lnA_esc = np.nansum(lnA_esc, axis=0) / np.nansum(escaping, axis=0) # integrated over A


	lnA_array = np.zeros_like(energies)
	lnA_array_esc = np.zeros_like(energies)
	ln_arrays = [lnA_array, lnA_array_esc]
	cr_spectra = [ncr_tot[1:],escape_tot[1:]]
	for j,n in enumerate(cr_spectra):
		for i in range(len(energies)):
			ncr_sum = np.sum(n[:,i])
			#print (ncr_sum.shape, n.shape, ncr_tot.shape)
			A_mean = np.sum(np.log(elems["a"]) * n[:,i], axis=0) / ncr_sum
			#print (ncr_sum, A_mean, n[:,i].shape)
			ln_arrays[j][i] = A_mean

			#std = np.std(ncr[])

	fig = plt.figure( figsize=(8.75,5.25))
	logE = np.log10(energies)
	std = np.nanstd(lnA, axis=0)
	mean = np.nanmean(lnA, axis=0)
	plt.plot(logE, mean, label="Inside lobe, time averaged", lw=4)
	plt.fill_between(logE, mean-std, mean+std, alpha=0.3)

	std = np.nanstd(lnA_esc, axis=0)
	mean = np.nanmean(lnA_esc, axis=0)
	plt.plot(logE, mean, ls="--", label="Escaping, time averaged", lw=4)
	plt.fill_between(logE, mean-std, mean+std, alpha=0.3)

	#plt.plot(logE, ln_arrays[0], label="Inside lobe, time averaged", lw=1)
	#plt.plot(logE, ln_arrays[1], ls="--", label="Escaping, time averaged", lw=1)
	plt.legend(fontsize=16, loc=4, frameon=False)
	plt.xlim(18,20.5)
	A = np.array([4,14,28,56])
	plt.hlines([np.log(A)], 18,20.5, lw=1.5, ls="-.")
	label = ["He", "N", "Si", "Fe"]
	for i, a in enumerate(np.log(A)):
		plt.text(18.25,a-0.2,label[i], fontsize=14)

	plt.xticks(fontsize=18)
	plt.yticks(fontsize=18)
	plt.xlabel(r"$\log[E ({\rm eV})]$", fontsize=20)
	plt.ylabel(r"$\langle \ln A \rangle$", fontsize=20)
	plt.subplots_adjust(top=0.98,bottom=0.13,right=0.97,left=0.09)
	plt.savefig("{}/lnA.png".format(folder), dpi=200)


def three_panel_cr_hist(time, energies, ncr, escaping, folder, e_power=2):

	# first, let's make integrated spectra
	ncr_tot = np.sum(ncr, axis=1) # integrated over time 
	escape_tot = np.sum(escaping, axis=1) # integrated over time 
	elems = sim.get_elem_dict(beta = 2)


	e_units = energies ** e_power
	ncr_tot_all = e_units * np.sum(ncr_tot[1:], axis=0)
	escape_tot_all = e_units * np.sum(escape_tot[1:], axis=0)

	fig = plt.figure( figsize=(11,5))
	ax1 = fig.add_subplot(131)
	ax2 = fig.add_subplot(132)
	ax3 = fig.add_subplot(133)
	ncr_max = np.max(ncr_tot_all[energies>=1e16])
	escape_max = np.max(escape_tot_all[energies>=1e16])
	#ion_labels = elems["species"]
	ion_labels = ["Protons", "He", "CNO", "Fe", "Others"]
	logE = np.log10(energies)
	linestyles = [(0,(3,1,1,1)), "-", (0,(5,1)), (0,(1,0.5)), ":"]
	# linestyles=["-","--",":","-."]
	for i in range(1,len(ion_labels)+1):
		ax1.plot(logE, e_units * ncr_tot[i]/ncr_max, label=ion_labels[i-1], ls=linestyles[i-1])
		ax2.plot(logE, e_units * escape_tot[i]/escape_max, label=ion_labels[i-1], ls=linestyles[i-1])	

	ax1.plot(logE, ncr_tot_all/ncr_max, c="k", lw=4, label="Total", alpha=0.7)
	ax2.plot(logE, escape_tot_all/escape_max, c="k", lw=4, label="Total", alpha=0.7)
	ax1.legend(fontsize=14)
	ax2.legend(fontsize=14)
	ax1.set_title("Inside lobe, time averaged")
	ax2.set_title("Escaping, time averaged")

	times = np.random.randint(0,len(time),size=30)
	ncr_time_sum = np.sum(ncr[1:], axis=0)
	escape_time_sum = np.sum(escaping[1:], axis=0)

	ncr_plot = e_units * ncr_time_sum

	Nbins = 100
	ncr_bins = np.logspace(-3,0,Nbins)
	e_bins = np.logspace(16,20,Nbins)
	dn = 3 / Nbins 
	de = 4 / Nbins
	counts = np.zeros((Nbins+1, Nbins+1))

	for i,t in enumerate(times):
		print (i)
		ncr_plot = e_units * ncr_time_sum[t]
		ncr_plot = ncr_plot/np.max(ncr_plot[energies>=1e16])

		for j, n in enumerate(ncr_plot):
			e = energies[j]

			if e > 1e16 and e<1e20:
				k_n = int( (np.log10(n)+3)/dn)
				k_e = int( (np.log10(e)-16)/de)
				#print (k_n, k_e, e, n)
				counts[k_n, k_e] += 1



	ax3.pcolormesh(counts)

	# for i,t in enumerate(times):
	# 	#print (energies.shape, ncr_time_sum.shape, ncr_time_sum[t].shape)
	# 	ncr_plot = e_units * ncr_time_sum[t]
	# 	if i == 0:
	# 		labels = ("Inside Lobe", "Escaping")
	# 	else:
	# 		labels = (None, None)

	# 	ax3.plot(logE, ncr_plot/np.max(ncr_plot[energies>=1e16]), c="C0", alpha=0.5, zorder=2, label=labels[0])
	# 	escaping_plot = e_units * escape_time_sum[t]
	# 	ax3.plot(logE, escaping_plot/np.max(escaping_plot[energies>=1e16]), c="C1", alpha=0.5, zorder=1, label=labels[1])

	ax3.legend(fontsize=14)
	# ax3.set_title("Spectra at random times")
	# for ax in [ax1,ax2,ax3]:
	# 	ax.set_xlim(16,19.9)
	# 	ax.set_ylim(1e-3,2)
	# 	ax.set_yscale("log")
	ax2.set_yticklabels([])
	ax3.set_yticklabels([])
	ax1.set_ylabel("$E^{:d} dN/dE$ (Normalised)".format(e_power), fontsize=18)
	ax2.set_xlabel(r"$\log[E ({\rm eV})]$", fontsize=18)
	plt.subplots_adjust(hspace=0.05,wspace=0.05,right=0.98,top=0.94, left=0.08, bottom=0.13)
 
	fig.savefig("{}/cr-3panel-E{:d}.png".format(folder, e_power), dpi=200)

	return (fig)


def three_panel_cr(time, energies, ncr, escaping, folder, e_power=2):

	# first, let's make integrated spectra
	ncr_tot = np.sum(ncr, axis=1) # integrated over time 
	escape_tot = np.sum(escaping, axis=1) # integrated over time 
	elems = sim.get_elem_dict(beta = 2)


	e_units = energies ** e_power
	ncr_tot_all = e_units * np.sum(ncr_tot[1:], axis=0)
	escape_tot_all = e_units * np.sum(escape_tot[1:], axis=0)

	fig = plt.figure( figsize=(11,5))
	ax1 = fig.add_subplot(131)
	ax2 = fig.add_subplot(132)
	ax3 = fig.add_subplot(133)
	ncr_max = np.max(ncr_tot_all[energies>=1e16])
	escape_max = np.max(escape_tot_all[energies>=1e16])
	#ion_labels = elems["species"]
	ion_labels = ["Protons", "He", "CNO", "Fe", "Others"]
	logE = np.log10(energies)
	linestyles = ["-", (0,(5,1)), (0,(3,1,1,1)), (0,(1,0.5)), ":"]
	# linestyles=["-","--",":","-."]
	for i in range(1,len(ion_labels)+1):
		ax1.plot(logE, e_units * ncr_tot[i]/ncr_max, label=ion_labels[i-1], ls=linestyles[i-1])
		ax2.plot(logE, e_units * escape_tot[i]/escape_max, label=ion_labels[i-1], ls=linestyles[i-1])	

	ax1.plot(logE, ncr_tot_all/ncr_max, c="k", lw=4, label="Total", alpha=0.7)
	ax2.plot(logE, escape_tot_all/escape_max, c="k", lw=4, label="Total", alpha=0.7)
	ax1.legend(fontsize=14)
	ax2.legend(fontsize=14)
	ax1.set_title("Inside lobe, time averaged")
	ax2.set_title("Escaping, time averaged")

	times = np.random.randint(0,len(time),size=30)
	ncr_time_sum = np.sum(ncr[1:], axis=0)
	escape_time_sum = np.sum(escaping[1:], axis=0)
	for i,t in enumerate(times):
		#print (energies.shape, ncr_time_sum.shape, ncr_time_sum[t].shape)
		ncr_plot = e_units * ncr_time_sum[t]
		if i == 0:
			labels = ("Inside Lobe", "Escaping")
		else:
			labels = (None, None)

		ax3.plot(logE, ncr_plot/np.max(ncr_plot[energies>=1e16]), c="C0", alpha=0.5, zorder=2, label=labels[0])
		escaping_plot = e_units * escape_time_sum[t]
		ax3.plot(logE, escaping_plot/np.max(escaping_plot[energies>=1e16]), c="C1", alpha=0.5, zorder=1, label=labels[1])

	ax3.legend(fontsize=14)
	ax3.set_title("Spectra at random times")
	for ax in [ax1,ax2,ax3]:
		ax.set_xlim(16,20.99)
		ax.set_ylim(1e-3,2)
		ax.set_yscale("log")
		ax.grid(ls=":")
	ax2.set_yticklabels([])
	ax3.set_yticklabels([])
	ax1.set_ylabel("$E^{:d}~n(E)$ (Normalised)".format(e_power), fontsize=18)
	ax2.set_xlabel(r"$\log[E ({\rm eV})]$", fontsize=18)
	plt.subplots_adjust(hspace=0.05,wspace=0.05,right=0.98,top=0.94, left=0.08, bottom=0.13)
 
	fig.savefig("{}/cr-3panel-E{:d}.png".format(folder, e_power), dpi=200)

	return (fig)

def standard_plot(j, folder):

	tmax = np.max(j.time/unit.myr)

	colors = ["C0", "C1", "C2", "C3", "C4"]

	fig = plt.figure( figsize=(11,5) )

	# First make timescales panel 
	ax1 = fig.add_subplot(131)
	ax1.set_title("Timescales (Myr)")
	#tsync_ghz = np.sqrt(3.0 / THOMPSON / THOMPSON * 2.0 * PI * MELEC * C * E / 1.4e9 / (j.B**3)) 
	tsync_ghz = 41.0 * (j.B /1e-5)**(-1.5) * (5**-0.5)
	ax1.plot(j.time/unit.myr, j.tesc/unit.myr, label=r"$\tau_\mathrm{esc}^{\rm cr}~(10~{\rm EeV}~p)$", ls=LINESTYLES[0], c=colors[0])
	ax1.plot(j.time/unit.myr, j.tsync/unit.myr, label=r"$\tau_\mathrm{sync}^e~(1~{\rm GeV~}e^-)$", ls=LINESTYLES[1], c=colors[1])
	ax1.plot(j.time/unit.myr, tsync_ghz, label=r"$\tau_\mathrm{sync}^e~(\nu_c=5~{\rm GHz})$", ls=LINESTYLES[1], c="C4")

	#ax1.plot(j.time/unit.myr, np.log10(j.length/j.v_advance/unit.myr), label=r"$\tau_\mathrm{adv}$")
	ax1.plot(j.time/unit.myr, j.length/j.jet.cs_env/unit.myr, label=r"$\tau_\mathrm{cs}$", ls=LINESTYLES[2], c=colors[2])
	ax1.plot(j.time/unit.myr, j.time/unit.myr, label=r"$t$", ls=LINESTYLES[3], c=colors[3])
	ax1.legend(frameon=False, fontsize=12)
	ax1.set_title("Timescales (Myr)")
	ax1.set_yscale("log")
	ax1.set_ylim(1,800)
	ax1.set_xlim(0,tmax)
	#ax1.set_yticks([-1,0,1,2,3])
	ax1.set_yticks([1,10,100])
	ax1.set_yticklabels(["1", "10", "100"])
	#ax1.set_yticklabels(["0.1", "1", "10", "100","1000"])






	# calculate some important quantities 
	nprot = j.density / unit.mprot
	#print (j.density, j.volume, unit.mprot)
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
	beta = j.v_j / unit.c
	ax2.plot(j.time/unit.myr, j.B/1e-6, label=r"$B~(\mu$G)", ls=LINESTYLES[0], c=colors[0])
	ax2.plot(j.time/unit.myr, beta * j.gmm, label=r"$\beta_j \Gamma_j$", ls=LINESTYLES[1], c=colors[1])
	ax2.plot(j.time/unit.myr, nprot/1e-7, label=r"$n~({10^{-7}~\rm cm}^{-3})$", ls=LINESTYLES[2], c=colors[2])

	kt = j.pressure / nprot / unit.ev / 1e7
	P = j.pressure / unit.ev 
	#ax2.plot(j.time/unit.myr, np.log10(P), label=r"$\log[P~({10^{-13}\rm Pa})]$")
	ax2.plot(j.time/unit.myr, P, label=r"$P~({\rm eV~cm}^{-3})$", ls=LINESTYLES[3], c=colors[3])
	print (np.max(kt), np.min(P))

	ax2.set_yscale("log")
	ax2.set_ylim(0.1,500)
	ax2.set_xlim(0,tmax)
	ax2.set_yticks([0.1,1,10,100])
	ax2.set_yticklabels(["0.1","1", "10", "100"])
	ax2.legend(frameon=False, fontsize=12)
	ax2.set_title("Physical Quantities")
	ax2.set_xlabel("t (Myr)", fontsize=16)


	ax3 = fig.add_subplot(133)
	ax3.plot(j.time/unit.myr, j.length/unit.kpc, label=r"Length (kpc)", ls=LINESTYLES[0], c=colors[0])
	ax3.plot(j.time/unit.myr, j.width/unit.kpc, label=r"Width (kpc)", ls=LINESTYLES[1], c=colors[1])
	ax3.plot(j.time/unit.myr, j.length/j.width, label=r"Aspect", ls=LINESTYLES[2], c=colors[2])
	# ax2.plot(j.time/unit.myr, np.log10( 10.0*(j.time/unit.myr)**0.6), label=r"Theory")
	ax3.set_yscale("log")
	ax3.legend(frameon=False)
	

	ax3.set_title("Dimensions")
	ax3.set_ylim(0.1,300)
	ax3.set_xlim(0,tmax)
	ax3.set_yticks([0.1,1,10,100])
	ax3.set_yticklabels(["0.1","1", "10", "100"])

	plt.subplots_adjust(left=0.045, hspace=0.15, wspace=0.15, right=0.98, top=0.94, bottom=0.13)
	
	fig.savefig("{}/time-evolution.png".format(folder), dpi=200)
	fig.clf()


def luminosities_plot_simpler(j, folder):
	#plt.figure(figsize=(10,7))
	#plt.subplot(211)
	plt.figure(figsize=(8.75,5.25))


	plt.plot(j.time/unit.myr, np.log10(j.power), label=r"$\log(Q_{j})$", c="k", alpha=0.8, lw=2)
	#plt.plot(j.time/unit.myr, np.log10(j.lcr), label=r"$\log(L_{\mathrm{cr,60EeV}})$", c="C1")
	plt.plot(j.time/unit.myr, np.log10(j.lcr_8EeV), label=r"$\log(L_{\mathrm{cr,8EeV}})$", c="C1", ls="-")
	plt.plot(j.time/unit.myr, np.log10(j.lsync*1.4e9), label=r"$\log(\nu L_{1400})$", c="C0", ls=LINESTYLES[2])
	plt.plot(j.time/unit.myr, np.log10(j.l144*1.44e8), label=r"$\log(\nu L_{144})$", c="C2", ls=LINESTYLES[2])
	plt.ylim(42.1,47.99)
	plt.ylabel(r"$\log$~[Luminosity~(erg~s$^{-1}$)]", fontsize=18)
	plt.xlim(0,np.max(j.time/unit.myr))
	#plt.gca().set_xticklabels([])
	plt.legend(fontsize=18, loc=1, frameon=False)
	#plt.text(37, 39.5, "Logarithmic", fontsize=14)


	plt.xlabel(r"$t~({\rm Myr})$", fontsize=20)
	#plt.ylabel(r"Power or Luminosity (erg~s$^{-1}$)]", fontsize=16)
	#plt.xlim(0,np.max(j.time/unit.myr))
	plt.xticks(fontsize=18)
	plt.yticks(fontsize=18)
	plt.subplots_adjust(left=0.08, right=0.97, top=0.98, bottom=0.13, hspace=0, wspace=0)
	plt.savefig("{}/luminosity2.png".format(folder), dpi=200)


def luminosities_plot(j, folder):
	plt.figure(figsize=(10,7))
	plt.subplot(211)


	plt.plot(j.time/unit.myr, np.log10(j.power), label=r"$\log(Q_{j})$", c="k", alpha=0.8)
	#plt.plot(j.time/unit.myr, np.log10(j.lcr), label=r"$\log(L_{\mathrm{cr,60EeV}})$", c="C1")
	plt.plot(j.time/unit.myr, np.log10(j.lcr_8EeV), label=r"$\log(L_{\mathrm{cr,8EeV}})$", c="C1", ls="--")
	plt.plot(j.time/unit.myr, np.log10(j.lsync*1.4e9), label=r"$\log(\nu L_{1400})$", c="C0", ls="-.")
	plt.plot(j.time/unit.myr, np.log10(j.l144*1.44e8), label=r"$\log(\nu L_{144})$", c="C2", ls="-.")
	plt.ylim(41.1,47)
	plt.ylabel(r"$\log$[Luminosity(erg~s$^{-1}$)]", fontsize=12)
	plt.xlim(0,np.max(j.time/unit.myr))
	plt.gca().set_xticklabels([])
	plt.legend(fontsize=14, loc=1)
	plt.text(37, 39.5, "Logarithmic", fontsize=14)


	plt.subplot(212)
	plt.plot(j.time/unit.myr, j.power/np.max(j.power), label=r"$Q_{j}$", c="k", alpha=0.8)
	#plt.plot(j.time/unit.myr, np.log10(j.lcr), label=r"$\log(L_{\mathrm{cr,60EeV}})$", c="C1")
	plt.plot(j.time/unit.myr, j.lcr_8EeV/np.max(j.lcr_8EeV), label=r"$L_{\mathrm{cr,8EeV}}$", c="C1", ls="--")
	plt.plot(j.time/unit.myr, j.lsync/np.max(j.lsync), label=r"$L_{1400})$", c="C0", ls="-.")
	plt.plot(j.time/unit.myr, j.l144/np.max(j.l144), label=r"$L_{144}$", c="C2", ls="-.")
	#plt.text()
	plt.xlabel("t (Myr)", fontsize=16)
	plt.ylabel(r"Luminosity (normalised to max.)", fontsize=12)

	plt.text(10, 0.8, "Linear, normalised", fontsize=14)
	plt.ylim(0,1)
	plt.legend(fontsize=14)
	#plt.ylabel(r"Power or Luminosity (erg~s$^{-1}$)]", fontsize=16)
	plt.xlim(0,np.max(j.time/unit.myr))
	plt.subplots_adjust(left=0.1, right=0.98, top=0.94, bottom=0.13, hspace=0, wspace=0)
	plt.savefig("{}/luminosity.png".format(folder), dpi=200)


def reshape_for_cno(ncr):
	elems = sim.get_elem_dict(beta = 2)
	shape_new = list(ncr.shape)
	shape_new[0] = 6
	ncr2 = np.zeros(shape_new)

	icount = 0
	ncr2[0:3,:,:] = ncr[0:3,:,:]


	for i, s in enumerate(elems["species"]):
		j = i + 1
		#print (i, s, j, shape_new)
		if s in "CNO":
			ncr2[3,:,:] += ncr[j,:,:]
		elif s == "Fe":
			ncr2[4,:,:] = ncr[j,:,:]
		elif s not in "HHe":
			ncr2[5,:,:] += ncr[j,:,:]
			icount += 1

	return (ncr2)
	

def create_html_summary(folder, savename, advance):
	import xhtml, os

	string = xhtml.begin(r'Plots for {}, mean advance {:.4f}c'.format(savename, advance))

	widths= [800,800,800,800]
	fnames = ["time-evolution", "luminosity", "cr-3panel-E2", "lnA"]
	titles = ["Time evolution", "Luminosities over time", "Cosmic ray spectra", "Mass composition"]

	#stri

	for i,f in enumerate(fnames):
		string += xhtml.h2(titles[i])
		os.system("mkdir -p html/img/{}".format(folder))
		os.system("cp {}/{}.png html/img/{}/".format(folder, f, folder))
		string += xhtml.image('file:img/{}/{}.png'.format(folder, f), width=widths[i])
		string += xhtml.hline()
	string += xhtml.end()

	#print(string)

	g=open('html/{}.html'.format(savename),'w')
	g.write(string)


def make_plots(savename=None, nenergies=3000):

	#cooling_plot()
	cr_energies = np.logspace(14,21,num=nenergies)
	e_energies = np.logspace(7,14,num=nenergies)
					
	#print ("test")

	folder = "paper-figures-{}".format(savename)

	os.system("mkdir -p {}".format(folder))

	#set_plot_defaults()

	try:
		fname = "array_saves/jetstore_{}.pkl".format(savename)
		with open(fname, "rb") as pickle_file:
			j = pickle.load(pickle_file)

		print (savename)

		# load escaping spectra  
		# npz files store a list of files so need convert to numpy array
		f_escaping = np.load("array_saves/escaping_{}.npz".format(savename))
		escaping = f_escaping[f_escaping.files[0]]
		f_escaping.close()
		# load particle spectra 
		f_ncr = np.load("array_saves/ncr_{}.npz".format(savename))
		ncr = f_ncr[f_ncr.files[0]]
		f_ncr.close()
		advance = j.length[-1]/j.time[-1]/unit.c
		print (ncr.shape, j.time.shape, np.max(j.time/unit.myr))

		lnA_plot(cr_energies, ncr, escaping, folder)

		#print (ncr.shape)
		ncr2 = reshape_for_cno(ncr)
		escaping2 = reshape_for_cno(escaping)
		fname = savename

		three_panel_cr(j.time / unit.myr, cr_energies, ncr2, escaping2, folder, e_power=2)
		#three_panel_cr(j.time / unit.myr, cr_energies, ncr2, escaping2, folder, e_power=3)

		lc_plot(j, folder)

		standard_plot(j, folder)

		luminosities_plot(j, folder)
		luminosities_plot_simpler(j, folder)

		create_html_summary(folder, savename, advance)

	except FileNotFoundError:
		print ("FileNotFoundError", savename)



# if __name__ == "__main__":
# 	sigma, power, beta, seed = 1.5, 1e44, 2, 38 
# 	psd_beta = None
# 	if len(sys.argv) == 2:
# 		savename = sys.argv[1]
# 	else:
# 		savename = None
# 		power = 10.0**float(sys.argv[4])
# 		seed = int(sys.argv[1])
# 		beta = float(sys.argv[2])
# 		sigma = float(sys.argv[3])
# 		if len(sys.argv)>5:
# 			psd_beta = int(sys.argv[5])

# 	make_all_plots(savename=savename)

if __name__ == "__main__":
	import jm_util as util
	import matplotlib.style as style
	from cycler import cycler
	#style.use("default")
	#style.use("fivethirtyeight"
	custom_cycler = (cycler(color=['#3e8a9f', '#fa7e2f', '#45ab53', '#e50a28', '#946999', '#f69a90', '#808080']))

	custom_cycler = (cycler(color=['#1f77b4', '#fa7e2f', '#45ab53', '#e50a28', '#946999', '#f69a90', '#808080']))
	#util.set_color_cycler(custom_cycler)
	util.set_mod_defaults()
	plt.rcParams['axes.prop_cycle'] = custom_cycler
	#savenames = [f for f in os.listdir("array_saves") if "dim_ref_p" in f]
	#savenames = ["dim_ref_p1e+45_geo4_etah0.3.npz"]
	savenames = ["dim_ref_p1e+45_geo4_etah0.3.npz"]
	#savenames = ["dim_ref_p1e+45_geo4_etah0.3_eta0.0001_steady.npz"]

	for savename in savenames:
		#create_and_save_spectra(sigma, power, beta, seed, nspectra=n)
		make_plots(savename=savename[4:-4])





