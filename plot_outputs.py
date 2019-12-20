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

plt.rcParams["text.usetex"] = "True"
plt.rcParams['xtick.labelsize']=14
plt.rcParams['ytick.labelsize']=14
plt.rcParams['font.serif']=['cm']
plt.rcParams['font.family']='serif'	
plt.rcParams["text.usetex"] = "True"
plt.rcParams["lines.linewidth"] = 3
plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams["xtick.major.width"] = 1.5
plt.rcParams["ytick.major.width"] = 1.5

def standard_plot(j, plot_name):

	plt.figure( figsize=(10,4) )
	plt.subplot(131)
	plt.plot(j.time/unit.myr, np.log10(j.tesc/unit.myr), label=r"$\tau_\mathrm{esc}$")
	plt.plot(j.time/unit.myr, np.log10(j.tsync/unit.myr), label=r"$\tau_\mathrm{sync}$")
	plt.plot(j.time/unit.myr, np.log10(j.length/j.v_j/unit.myr), label=r"$\tau_\mathrm{dyn}$")

	nprot = j.density / unit.mprot / j.volume
	#print (j.B)
	tf2 = 220.0 * ((j.B / 1e-6)**(-3)) * ((nprot/1e-6))
	#plt.plot(j.time/unit.myr, np.log10(tf2), label=r"$\tau_\mathrm{F2}$")
	plt.legend()
	plt.title("Timescales")
	plt.ylim(-1,2)


	plt.subplot(132)
	plt.plot(j.time/unit.myr, np.log10(j.length/unit.kpc), label=r"Length")
	plt.plot(j.time/unit.myr, np.log10(j.width/unit.kpc), label=r"Width")
	plt.plot(j.time/unit.myr, np.log10(j.length/j.width), label=r"Aspect")
	plt.legend()
	plt.title("Dimensions")
	plt.ylim(0,2)

	plt.subplot(133)
	plt.plot(j.time/unit.myr, np.log10(j.lcr), label=r"$L_{\mathrm{uhecr}}$")
	#plt.plot(j.time/unit.myr, np.log10(j.lsync), label=r"Width")
	plt.legend()
	plt.title("Luminosities")
	plt.subplots_adjust(left=0.05, right=0.96)
	
	plt.savefig("stdplot_{}.png".format(plot_name), dpi=200)




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
			try:
				with open(fname, "rb") as pickle_file:
					j = pickle.load(pickle_file)


				units = [unit.kpc, unit.kpc, unit.c, 1e-6, unit.myr, unit.myr, 1e40, 1e32]
				variables = ["length", "width", "v_j", "B", "tesc", "tsync", "lcr", "lsync"]

				plot_name = "{:.1f}q{:.1f}sig{:.1f}".format(BETA, logflux,  SIGMA)
				standard_plot(j, plot_name)


				#my_lcr = np.concatenate((my_lcr, getattr(j, "lcr")))
				#print (j.sync)
				my_ls.append(j.lcr)
				my_length.append(j.length)
				print ("here")
				#print ("MYLS:", my_ls[i])
				#my_ls = np.concatenate((my_ls, getattr(j, "lsync")))
				#my_length = np.concatenate((my_length, getattr(j, "length") / units[0]))
				#my_width = np.concatenate((my_width, getattr(j, "width") / units[1]))
				#power = getattr(j, "v_j")
				#my_power = np.concatenate((my_power, power))

				#fig_all.plot(getattr(j, "lsync"), getattr(j, "length") / units[0])

			except FileNotFoundError:
				print ("Error for {}".format(fname))
			except AttributeError:
				print ("AttributeError for {}".format(fname))

#fig_all.loglog()
#fig_all.savefig("p-d.png", dpi=200)

# plt.close("all")
lsflat = np.log10(np.array(my_ls).flatten())
lslength = np.log10(np.array(my_length).flatten()/unit.kpc)

plt.figure()
plt.hexbin(lslength, lsflat, gridsize=20, extent=(-1, 3, 38,44))
#for i in range(len(my_ls)):
#	plt.plot(my_ls[i], my_length[i], alpha=0.5)
# # #plt.scatter(my_ls, my_length)
#plt.loglog()
plt.savefig("p-d.png", dpi=200)
