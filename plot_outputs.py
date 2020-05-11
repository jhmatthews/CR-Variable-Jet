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
from simulation import unit, universal_profile
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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

def standard_plot(j, plot_name):

	fig = plt.figure( figsize=(10,4) )
	ax1 = fig.add_subplot(131)
	ax1.plot(j.time/unit.myr, np.log10(j.tesc/unit.myr), label=r"$\tau_\mathrm{esc}$")
	ax1.plot(j.time/unit.myr, np.log10(j.tsync/unit.myr), label=r"$\tau_\mathrm{sync}$")
	#ax1.plot(j.time/unit.myr, np.log10(j.length/j.v_advance/unit.myr), label=r"$\tau_\mathrm{adv}$")
	ax1.plot(j.time/unit.myr, np.log10(j.length/j.jet.cs_env/unit.myr), label=r"$\tau_\mathrm{cs}$")
	ax1.plot(j.time/unit.myr, np.log10(j.time/unit.myr), label=r"$\tau_\mathrm{age}$")

	nprot = j.density / unit.mprot / j.volume
	#print (j.B)
	print (j.v_j/unit.c, j.gmm, j.B/1e-6, j.length/unit.kpc)

	rho_j = j.power / (j.gmm * (j.gmm - 1.) * j.jet.area * unit.c * unit.c * j.v_j)

	#print (rho_j / unit.mprot, np.median(j.power))
	tf2 = 220.0 * ((j.B / 1e-6)**(-3)) * ((nprot/1e-6))
	#plt.plot(j.time/unit.myr, np.log10(tf2), label=r"$\tau_\mathrm{F2}$")
	ax1.legend()
	ax1.set_title("Timescales (Myr)")
	ax1.set_ylim(-1,2)
	ax1.set_xlabel("t (Myr)")
	ax1.set_yticks([-1,0,1,2,3])
	ax1.set_yticklabels(["0.1", "1", "10", "100","1000"])


	ax2 = fig.add_subplot(132)
	ax2.plot(j.time/unit.myr, np.log10(j.length/unit.kpc), label=r"Length (kpc)")
	ax2.plot(j.time/unit.myr, np.log10(j.width/unit.kpc), label=r"Width (kpc)")
	ax2.plot(j.time/unit.myr, np.log10(j.length/j.width), label=r"Aspect")
	ax2.plot(j.time/unit.myr, np.log10( 10.0*(j.time/unit.myr)**0.6), label=r"Theory")
	#ax2.set_yscale("log")
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
	#print (j.v_j)
	power1 = 0.5 * 1e-4 * unit.mprot * (j.v_j**3) * unit.kpc * unit.kpc * 0.5 * 0.5 
	try:
		power2 = j.power
	except AttributeError:
		power2 = power1
	#power = 0.5 * 1e-4 * unit.mprot * (j.v_j**3) * unit.kpc * unit.kpc * 0.5 * 0.5 
	ax3.plot(j.time/unit.myr, np.log10(power2), label=r"$\log(Q_j)$")
	ax3.plot(j.time/unit.myr, np.log10(power1), label=r"$\log(Q_j)$")
	ax3.legend()
	ax3.set_xlabel("t (Myr)")
	ax3.set_title(r"Luminosities (erg~s${-1}$)")
	plt.subplots_adjust(left=0.05, right=0.96)
	
	fig.savefig("test_std_plots/stdplot_{}.png".format(plot_name), dpi=200)
	fig.clf()

def standard_plot2(j, plot_name):

	fig = plt.figure( figsize=(7,6) )
	ax1 = fig.add_subplot(221)

	tsync = np.sqrt(3.0 / THOMPSON / THOMPSON * 2.0 * PI * MELEC * C * E / 1.4e9 / (j.B**3)) 
	ax1.plot(j.time/unit.myr, np.log10(j.tesc/unit.myr), label=r"$\tau_\mathrm{esc}$")
	ax1.plot(j.time/unit.myr, np.log10(j.tsync/unit.myr), label=r"$\tau_\mathrm{sync}$")
	#ax1.plot(j.time/unit.myr, np.log10(j.length/j.v_advance/unit.myr), label=r"$\tau_\mathrm{adv}$")
	ax1.plot(j.time/unit.myr, np.log10(j.length/j.jet.cs_env/unit.myr), label=r"$\tau_\mathrm{cs}$")
	ax1.plot(j.time/unit.myr, np.log10(j.time/unit.myr), label=r"$\tau_\mathrm{age}$")

	nprot = j.density / unit.mprot / j.volume
	#print (j.B)
	#print (j.v_j/unit.c, j.gmm, j.B/1e-6, j.length/unit.kpc)
	rho_0 = [universal_profile(r, j.jet.M500_msol)[2] for r in j.length]
	#print (rho_0)
	eta_star = j.gmm * j.gmm * j.jet.rho_j / rho_0 
	factor = np.sqrt(eta_star) / (np.sqrt(eta_star) + 1)
	v_a_theory = factor * j.v_j
	v_a_theory = np.gradient(j.length, j.time) / unit.c
	print (v_a_theory)

	rho_j = j.power / (j.gmm * (j.gmm - 1.) * j.jet.area * unit.c * unit.c * j.v_j)
	print (np.mean(j.power), np.exp(np.log(1e43)+2.0))
	print (np.mean(j.v_j/unit.c), np.mean(j.gmm))

	#print (rho_j / unit.mprot, np.median(j.power))
	tf2 = 220.0 * ((j.B / 1e-6)**(-3)) * ((nprot/1e-6))
	#plt.plot(j.time/unit.myr, np.log10(tf2), label=r"$\tau_\mathrm{F2}$")
	ax1.legend()
	ax1.set_title("Timescales (Myr)")
	ax1.set_ylim(-1,3)
	#ax1.set_xlabel("t (Myr)")
	ax1.set_yticks([-1,0,1,2,3])
	ax1.set_yticklabels(["0.1", "1", "10", "100","1000"])

	ax2 = fig.add_subplot(222)
	U = j.energy / j.volume
	#ax2.plot(j.time/unit.myr, np.log10(nprot), label=r"$\log[n_c~(cm^{-3})]$")
	ax2.plot(j.time/unit.myr, np.log10(j.B/1e-6), label=r"$\log[B (\mu$G)]")
	ax2.plot(j.time/unit.myr, np.log10(j.gmm), label=r"$\log(\Gamma_j)$")
	ax2.plot(j.time/unit.myr, np.log10(nprot), label=r"$\log[n ({\rm cm}^{-3})]$")
	#ax2.plot(j.time/unit.myr, np.log10(j.volume / (10.0*unit.kpc)**3), label=r"$V$")
	#ax2.plot(j.time/unit.myr, np.log10(j.energy / 1e56), label=r"$E/1e56$")
	#ax2.plot(j.time/unit.myr, np.log10(j.energy / j.volume / 1e-11), label=r"$U$")
	#ax2.plot(j.time/unit.myr, np.log10(v_a_theory), label=r"$v1$", ls="--")
	#ax2.plot(j.time/unit.myr, np.log10(j.v_advance / unit.c), label=r"$v2$", alpha=0.6)
	print ("MEAN:", np.mean(j.v_advance / unit.c))
	#print (j.v_advance * 

	#print (np.m)
	#ax2.plot(j.time/unit.myr, np.log10(U), label=r"$U$")
	#ax2.set_yscale("log")
	ax2.legend()
	#ax2.set_xlabel("t (Myr)")
	ax2.set_title("Physical Quantities")
	#ax2.set_ylim(-0.1,3)
	#ax2.set_yticks([0,1,2])
	#ax2.set_yticklabels(["1", "10", "100"])


	ax2 = fig.add_subplot(223)
	ax2.plot(j.time/unit.myr, np.log10(j.length/unit.kpc), label=r"Length (kpc)")
	ax2.plot(j.time/unit.myr, np.log10(j.width/unit.kpc), label=r"Width (kpc)")
	ax2.plot(j.time/unit.myr, np.log10(j.length/j.width), label=r"Aspect")
	# ax2.plot(j.time/unit.myr, np.log10( 10.0*(j.time/unit.myr)**0.6), label=r"Theory")
	#ax2.set_yscale("log")
	ax2.legend()
	ax2.set_xlabel("t (Myr)", fontsize=16)

	ax2.set_title("Dimensions")
	ax2.set_ylim(0,2.5)
	ax2.set_yticks([0,1,2])
	ax2.set_yticklabels(["1", "10", "100"])

	ax3 = fig.add_subplot(224)
	alpha = np.log10(j.lsync/j.l144) / np.log10(1.44e8/1/4e9)
	#ax3.plot(j.time/unit.myr, alpha, label="alpha")
	#ax3.set_ylim(0,1)
	ax3.plot(j.time/unit.myr, np.log10(j.lcr), label=r"$\log(L_{\mathrm{uhecr}})$")
	ax3.plot(j.time/unit.myr, np.log10(j.lsync*1.4e9), label=r"$\log(L_{1400})$")
	ax3.plot(j.time/unit.myr, np.log10(j.l144*1.44e8), label=r"$\log(L_{144})$")
	#print (j.v_j)
	power1 = 0.5 * 1e-4 * unit.mprot * (j.v_j**3) * unit.kpc * unit.kpc * 0.5 * 0.5 
	try:
		power2 = j.power
	except AttributeError:
		power2 = power1
	#power = 0.5 * 1e-4 * unit.mprot * (j.v_j**3) * unit.kpc * unit.kpc * 0.5 * 0.5 
	ax3.plot(j.time/unit.myr, np.log10(power2), label=r"$\log(Q_j)$")
	cumulative = np.cumsum(power2)
	mean = cumulative / np.arange(1,len(power2)+1)
	#ax3.plot(j.time/unit.myr, np.log10(mean), label="mean Qj")


	#ax3.plot(j.time/unit.myr, np.log10(power1), label=r"$\log(Q_j)$")
	ax3.legend()
	ax3.set_xlabel("t (Myr)", fontsize=16)
	ax3.set_title("Luminosities")
	ax3.set_ylim(40,46)
	plt.subplots_adjust(left=0.08, right=0.96, wspace=0.1, bottom=0.1, top=0.96)
	
	fig.savefig("test_std_plots/stdplot_{}.png".format(plot_name), dpi=200)
	fig.clf()

def alpha_plot(j, plot_name):
	fig = plt.figure( figsize=(6,4) )
	ax3 = fig.add_subplot(111)
	alpha = np.log10(j.lsync/j.l144) / np.log10(1.44e8/1.4e9)
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
	ax.set_ylabel(r"$\log[Q_j~(\mathrm{erg~s}^{-1})]$", fontsize=18)
	ax.set_xlabel(r"$t~(\mathrm{Myr})$", fontsize=18)
	#ax.plot(j.time/unit.myr, np.log10(j.v_j/unit.c), c=lc_color, lw=1.5, alpha=alpha)
	ax.plot(j.time/unit.myr, np.log10(j.power), c=lc_color, lw=2, alpha=alpha)
	#print (j.v_j/unit.c)

	#ax.legend(loc=4)
	#ax.set_yscale("log")
	ax.set_xlim(0,np.max(j.time/unit.myr))
	ax.set_ylim(42,48)
	#plt.subplots_adjust(right=0.98, left=0.1, top=0.98)

	A,v_bend,a_low,a_high,c = 1, 1e-3, 1, 20, 0
	ax = plt.gca()
	axins = inset_axes(ax, width="50%", height="70%",bbox_to_anchor=(.08, .65, .4, .4),bbox_transform=ax.transAxes, loc=3)
	freq = np.logspace(-6,0,num=10000)
	axins.plot(freq * 1000.0, BendingPL(freq, 1000.0,v_bend,a_low,a_high,c), c=lc_color, alpha=alpha, lw=3)
	axins.set_xlabel(r"$\nu~(\mathrm{Myr}^{-1})$",fontsize=14)
	#axins.set_ylabel(r"$\mathrm{Power~(Arb.)}$")
	axins.set_ylim(1e-10,1e10)
	axins.set_xlim(1e-2,10)
	plt.loglog()
	plt.title(r"$\mathrm{PSD}$")

	axins = inset_axes(ax, width="50%", height="70%",bbox_to_anchor=(.75, .65, .4, .4),bbox_transform=ax.transAxes, loc=3)
	freq = np.logspace(-6,0,num=10000)
	#bins = np.arange(-2,0,0.1)
	bins = np.arange(42,47,0.25)
	axins.hist(np.log10(j.power), bins=bins, color=lc_color, alpha=alpha, density=True)
	#axins.set_xlim(42,45)
	axins.set_xticks([42,44,46])
	axins.set_xlabel(r"$\log[Q_j~(\mathrm{erg~s}^{-1})]$",fontsize=14)
	#axins.set_ylabel(r"$\mathrm{Power~(Arb.)}$")
	plt.title(r"$\mathrm{PDF}$")
	plt.subplots_adjust(top=0.98, right=0.98)

	
	fig.savefig("lc_plots/lc_{}.png".format(plot_name), dpi=200)
	fig.clf()



betas = [2,2.3,2.7]
sigmas = [0,0.5,1.3,2.2,3.0]
flux_scales = np.logspace(43,45,num=10)
flux_scale = []
sigmas = np.linspace(0,2,num=5)
betas = [2,2.3,2.7]
flux_scales = np.logspace(43,45,num=11)
flux_scales = [1e44]
sigmas = [1.5]
betas = [2]
seeds = [12,38,100,200]
seeds = [38]

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

			for seed in seeds:
				#print ("test")
				logflux = np.log10(flux_scale)
				fname = "array_saves/jetstore_{:.1f}q{:.1f}sig{:.1f}seed{:d}.npy.pkl".format(BETA, logflux,  SIGMA, seed)
				try:
					with open(fname, "rb") as pickle_file:
						j = pickle.load(pickle_file)

					print (fname)


					units = [unit.kpc, unit.kpc, unit.c, 1e-6, unit.myr, unit.myr, 1e40, 1e32]
					variables = ["length", "width", "v_j", "B", "tesc", "tsync", "lcr", "lsync"]

					plot_name = "{:.1f}q{:.1f}sig{:.1f}seed{:d}".format(BETA, logflux,  SIGMA, seed)
					standard_plot2(j, plot_name)
					lc_plot(j, plot_name)
					alpha_plot(j, plot_name)

					#my_lcr = np.concatenate((my_lcr, getattr(j, "lcr")))
					#print (j.sync)
					my_ls.append(j.lcr)
					my_length.append(j.length)
					#print ("here")
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
