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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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

	fig = plt.figure( figsize=(10,4) )
	ax1 = fig.add_subplot(131)
	ax1.plot(j.time/unit.myr, np.log10(j.tesc/unit.myr), label=r"$\tau_\mathrm{esc}$")
	ax1.plot(j.time/unit.myr, np.log10(j.tsync/unit.myr), label=r"$\tau_\mathrm{sync}$")
	ax1.plot(j.time/unit.myr, np.log10(j.length/j.v_j/unit.myr), label=r"$\tau_\mathrm{dyn}$")
	ax1.plot(j.time/unit.myr, np.log10(j.time/unit.myr), label=r"$\tau_\mathrm{age}$")

	nprot = j.density / unit.mprot / j.volume
	#print (j.B)
	tf2 = 220.0 * ((j.B / 1e-6)**(-3)) * ((nprot/1e-6))
	#plt.plot(j.time/unit.myr, np.log10(tf2), label=r"$\tau_\mathrm{F2}$")
	ax1.legend()
	ax1.set_title("Timescales (Myr)")
	ax1.set_ylim(-1,2)
	ax1.set_xlabel("t (Myr)")
	ax1.set_yticks([-1,0,1,2])
	ax1.set_yticklabels(["0.1", "1", "10", "100"])


	ax2 = fig.add_subplot(132)
	ax2.plot(j.time/unit.myr, np.log10(j.length/unit.kpc), label=r"Length (kpc)")
	ax2.plot(j.time/unit.myr, np.log10(j.width/unit.kpc), label=r"Width (kpc)")
	ax2.plot(j.time/unit.myr, np.log10(j.length/j.width), label=r"Aspect")
	ax2.legend()
	ax2.set_xlabel("t (Myr)")
	ax2.set_title("Dimensions")
	ax2.set_ylim(0,2)
	ax2.set_yticks([0,1,2])
	ax2.set_yticklabels(["1", "10", "100"])

	ax3 = fig.add_subplot(133)
	alpha = np.log10(j.lsync/j.l144) / np.log10(1.44e8/1/4e9)
	#ax3.plot(j.time/unit.myr, alpha, label="alpha")
	#ax3.set_ylim(0,1)
	ax3.plot(j.time/unit.myr, np.log10(j.lcr), label=r"$\log(L_{\mathrm{uhecr}})$")
	ax3.plot(j.time/unit.myr, np.log10(j.lsync*1.4e9), label=r"$\log(L_{1400})$")
	ax3.plot(j.time/unit.myr, np.log10(j.l144*1.44e8), label=r"$\log(L_{144})$")
	print (j.v_j)
	power = 0.5 * 1e-4 * unit.mprot * (j.v_j**3) * unit.kpc * unit.kpc * 0.5 * 0.5 
	ax3.plot(j.time/unit.myr, np.log10(power), label=r"$\log(Q_j)$")
	ax3.legend()
	ax3.set_xlabel("t (Myr)")
	ax3.set_title("Luminosities")
	plt.subplots_adjust(left=0.05, right=0.96)
	
	fig.savefig("std_plots/stdplot_{}.png".format(plot_name), dpi=200)
	fig.clf()

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
	print (j.v_j)
	power = 0.5 * 1e-4 * unit.mprot * (j.v_j**3) * unit.kpc * unit.kpc * 0.5 * 0.5 
	ax3.plot(j.time/unit.myr, power/np.max(power), label=r"$Q_j/Q_{\mathrm{max}}$")
	ax3.legend()
	ax3.set_xlabel("t (Myr)")
	ax3.set_title("Luminosities")
	plt.subplots_adjust(left=0.05, right=0.96)
	
	fig.savefig("alpha_plots/alphaplot_{}.png".format(plot_name), dpi=200)
	fig.clf()


def lc_plot(j, plot_name):
	# fig = plt.figure( figsize=(7,4) )
	# ax1 = fig.add_subplot(111)
	# ax1.plot(j.time/unit.myr, np.log10(j.tesc/unit.myr), label=r"$\tau_\mathrm{esc}$")
	# #ax1.plot(j.time/unit.myr, np.log10(j.tsync/unit.myr), label=r"$\tau_\mathrm{sync}$")
	# #ax1.plot(j.time/unit.myr, np.log10(j.length/j.v_j/unit.myr), label=r"$\tau_\mathrm{dyn}$")

	# nprot = j.density / unit.mprot / j.volume
	# #print (j.B)
	# tf2 = 220.0 * ((j.B / 1e-6)**(-3)) * ((nprot/1e-6))
	# #plt.plot(j.time/unit.myr, np.log10(tf2), label=r"$\tau_\mathrm{F2}$")
	# ax1.legend()
	# ax1.set_ylabel("Q_j (Myr)", fontsize=16)
	# ax1.set_xlabel("t (Myr)", fontsize=16)
	# ax1.set_ylim(-1,2)

	lc_color="k"
	alpha = 0.7

	fig = plt.figure(figsize=(10,6))
	ax = fig.add_subplot(111)
	ax.set_ylabel("$\log(v_j/c)$", fontsize=18)
	ax.set_xlabel("$t~(\mathrm{Myr})$", fontsize=18)
	ax.plot(j.time/unit.myr, np.log10(j.v_j/unit.c), c=lc_color, lw=1.5, alpha=alpha)
	print (j.v_j/unit.c)

	#ax.legend(loc=4)
	#ax.set_yscale("log")
	#plt.xlim(-100,0)
	#plt.ylim(1e42,1e47)
	#plt.subplots_adjust(right=0.98, left=0.1, top=0.98)

	A,v_bend,a_low,a_high,c = 1, 1e-3, 1, 20, 0
	ax = plt.gca()
	axins = inset_axes(ax, width="50%", height="70%",bbox_to_anchor=(.08, .65, .4, .4),bbox_transform=ax.transAxes, loc=3)
	freq = np.logspace(-6,0,num=10000)
	axins.plot(freq * 1000.0, BendingPL(freq, 1000.0,v_bend,a_low,a_high,c), c=lc_color, alpha=alpha, lw=3)
	axins.set_xlabel(r"$\nu~(\mathrm{Myr}^{-1})$")
	axins.set_ylabel(r"$\mathrm{Power~(Arb.)}$")
	plt.loglog()
	plt.title("$\mathrm{PSD}$")

	axins = inset_axes(ax, width="50%", height="70%",bbox_to_anchor=(.65, .65, .4, .4),bbox_transform=ax.transAxes, loc=3)
	freq = np.logspace(-6,0,num=10000)
	bins = np.arange(-2,0,0.1)
	axins.hist(np.log10(j.v_j), bins=bins, color=lc_color, alpha=alpha, density=True)
	#axins.set_xlim(42,45)
	axins.set_xlabel(r"$v_j$")
	#axins.set_ylabel(r"$\mathrm{Power~(Arb.)}$")
	plt.title("$\mathrm{PDF}$")

	
	fig.savefig("lc_plots/lc_{}.png".format(plot_name), dpi=200)
	fig.clf()



betas = [2,2.3,2.7]
sigmas = [0,0.5,1.3,2.2,3.0]
flux_scales = np.logspace(43,45,num=10)
flux_scale = []
sigmas = np.linspace(0,2,num=5)
betas = [2,2.3,2.7]
flux_scales = np.logspace(43,45,num=10)

my_lcr = np.zeros(1)
my_ls = np.zeros(1)
my_length = np.zeros(1)
my_width = np.zeros(1)
my_power = np.zeros(1)
my_ls = []
my_length = []

fig_all = plt.figure()
ax1 = fig_all.add_subplot(111)

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
				lc_plot(j, plot_name)


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

				ax1.plot(np.log10(j.length/unit.kpc), np.log10(j.l144/1e7), alpha=0.5, c="k")

				#fig_all.plot(getattr(j, "lsync"), getattr(j, "length") / units[0])

			except FileNotFoundError:
				print ("Error for {}".format(fname))
			#except AttributeError:
			#	print ("AttributeError for {}".format(fname))

#fig_all.loglog()
ax1.set_xlim(-0.5,2)
fig_all.savefig("p-d.png", dpi=200)
#gi

# plt.close("all")
lsflat = np.log10(np.array(my_ls).flatten())
lslength = np.log10(np.array(my_length).flatten()/unit.kpc)

# plt.figure()
# plt.hexbin(lslength, lsflat, gridsize=20, extent=(-1, 3, 38,44))
# #for i in range(len(my_ls)):
# #	plt.plot(my_ls[i], my_length[i], alpha=0.5)
# # # #plt.scatter(my_ls, my_length)
# #plt.loglog()
# plt.savefig("p-d.png", dpi=200)
