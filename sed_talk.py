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
import os, sys
# import time
import pickle
from simulation import unit
import msynchro
from jm_util import *
sys.path.append('/Users/matthews/cr/synchrotron_tests/GAMERA/lib')
import gappa as gp
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import gridspec

set_cr_defaults()

# def create_and_save_spectra(SIGMA, flux_scale, BETA, seed, nspectra=200):

# 	frequencies = np.logspace(8,12,400)
# 	spectra = np.zeros( (nspectra, len(frequencies)) )
# 	ispec = np.random.randint(1500, size=nspectra)

# 	energies = np.logspace(7,14,3000)
# 	select = (energies < 1e14)

# 	logflux = np.log10(flux_scale)
# 	fname = "array_saves/jetstore_beta{:.1f}q{:.1f}sig{:.1f}seed{:d}.npy.pkl".format(BETA, logflux,  SIGMA, seed)
# 	with open(fname, "rb") as pickle_file:
# 		j = pickle.load(pickle_file)

# 	# make the mesh plot with the remaining CRs 
# 	ncr = np.load("array_saves/ncr_beta{:.1f}q{:.1f}sig{:.1f}seed{:d}.npy".format(BETA, logflux,  SIGMA, seed))

# 	spectra = np.zeros( (nspectra, len(frequencies)) )
# 	for i in range(nspectra):
# 		Bfield = j.B[i]
# 		spectra[i,:] = msynchro.Ptot(frequencies, energies[select], ncr[0,i,:][select], Bfield)
# 		print (i)

# 	np.save("array_saves/syncspectra_beta{:.1f}q{:.1f}sig{:.1f}seed{:d}.npy".format(BETA, logflux,  SIGMA, seed), spectra)
# 	return (0)

def get_gamera_spectrum(nu, energies, ne, nprot, B, density):
	e = energies * gp.eV_to_erg # energy axis
	ne = ne / gp.eV_to_erg
	particles = list(zip(e,ne)) # input needs to be 2D-array
	protons = list(zip(e,nprot/gp.eV_to_erg)) # input needs to be 2D-array
	fr = gp.Radiation() # create the object

	#ambient_density = 1 # 1/cm^3, necessary for Bremsstrahlung and hadronic emission
	# radiation field parameters, necessary for Inverse-Compton radiation. 
	temp = 2.7 # Temperature in K
	edens = 0.25 * gp.eV_to_erg # energy density in erg / cm^-3
	distance = 1e3 # in pc

	#fr.SetAmbientDensity(ambient_density)
	fr.SetBField(B)
	fr.AddThermalTargetPhotons(temp,edens)
	fr.SetDistance(distance)
	fr.SetElectrons(particles)
	fr.SetProtons(protons)  
	fr.SetAmbientDensity(density)

	ee = unit.h * nu  # has to be in erg!
	fr.CalculateDifferentialPhotonSpectrum(ee)
	#sed = fr.GetICSpectrum()  # sum of all components
	#ee,f = np.array(sed).T
	#fnorm = ee * f * 4.0 * np.pi * (distance * unit.pc) **2

	sed = fr.GetTotalSpectrum()  # sum of all components
	ee,f = np.array(sed).T
	fnorm = ee * f * 4.0 * np.pi * (distance * unit.pc) **2

	sed2 = fr.GetPPSpectrum()  # sum of all components
	ee2,f2 = np.array(sed2).T
	fnorm2 = ee2 * f2 * 4.0 * np.pi * (distance * unit.pc) **2

	sed3 = fr.GetICSpectrum()  # sum of all components
	ee3,f3 = np.array(sed3).T
	fnorm3 = ee3 * f3 * 4.0 * np.pi * (distance * unit.pc) **2



	return(ee / unit.h, fnorm * unit.h, fnorm2 * unit.h, fnorm3 * unit.h)



def make_plots(savename="", nspectra=200, load=False):

	nu0 = 8
	nu1 = np.log10(1e12 / 4.13620e-15)
	##frequencies = np.log
	frequencies = np.logspace(nu0,nu1,300)
	energies = np.logspace(7,14,3000)
	#spectra = np.load("array_saves/syncspectra_beta{:.1f}q{:.1f}sig{:.1f}seed{:d}.npy".format(BETA, logflux,  SIGMA, seed))
	#folder = "paper-figures-{}-p{}-s{}_q{}".format(seed, BETA, sigma, logflux)
	
	fname = "array_saves/jetstore_{}.pkl".format(savename)
	with open(fname, "rb") as pickle_file:
		j = pickle.load(pickle_file)

	ncr = load_one_npz("array_saves/ncr_{}.npz".format(savename))

	nspectra = len(j.B)

	NSPEC = 9
	tuse = np.linspace(10,90,NSPEC)
	NUSE = [np.argmin(np.fabs(j.time/unit.myr-t)) for t in tuse]

	cmap = cm.get_cmap('viridis')
	colors = cmap(np.linspace(0,1,num=NUSE[-1]+1))
	#NUSE = np.random.randint(nspectra, size=NSPEC)
	#NUSE = np.linspace(50,nspectra-1,NSPEC).astype(int)
	fig = plt.figure(figsize=(8,8))
	gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3]) 
	#ax1 = fig.add_subplot(211)
	ax1 = fig.add_subplot(gs[1])
	ax2 = fig.add_subplot(gs[0])
	ax2.plot(j.time/unit.myr, np.log10(j.power), c="k", alpha=0.8)

	if load:
		sed_store = np.load("array_saves/sed_store_{}.npy".format(savename))
		ne_store = np.load("array_saves/ne_store_{}.npy".format(savename))
	else:
		sed_store = np.zeros((NSPEC,len(frequencies)))
		ne_store = np.zeros((NSPEC,len(ncr[0,0,:])))

	axins2 = inset_axes(ax1,width="100%",  height="100%", loc='lower left', bbox_to_anchor=(.65, .65, .3, .3), bbox_transform=ax1.transAxes)


	for n, i in enumerate(NUSE):

		
		#ax1.loglog(energies, energies *energies*ncr[0,i,:], c="C"+str(n))
		# for i in NUSE:
		#ne = ncr[]
		# i = NUSE
		print (j.B[i])
		#ncr[0,i,:][energies>1e12] = 0
		t = j.time[i] / unit.myr
		#plt.subplot(212)
		if load:
			nu = frequencies
			nufnu = sed_store[n,:]
		else:
			nu, f, fp, fic = get_gamera_spectrum(frequencies, energies, ncr[0,i,:], ncr[1,i,:], j.B[i], j.density[i]/unit.mprot)
			nufnu = nu * f
			ne_store[n] = ncr[0,i,:]
			sed_store[n] = nu * f
		#ax2.loglog(nu, nu*f, alpha=0.5, c="C"+str(n))
		# plt.loglog(nu * 4.13620e-15, nu*fp, ls="-.")
		ax1.loglog(nu, nufnu, ls="-", c=colors[i], alpha=0.85)

		n_to_plot = energies * energies * ne_store[n]
		print (np.max(n_to_plot))
		axins2.loglog(energies/1e6, n_to_plot / np.max(n_to_plot), c=colors[i], alpha=0.85)
		#ax2.scatter(j.time[i]/unit.myr, np.log10(j.power[i]), c=colors[i], zorder=3)
		ax2.vlines([j.time[i]/unit.myr],43,47, color=colors[i], ls="--")

		P = msynchro.Ptot(frequencies, energies, ncr[0,i,:], j.B[i])



		#plt.loglog(nu * 4.13620e-15, nu*P, ls="--", c="C"+str(n))
		#ne_store[n] = ncr[0,i,:]
		#sed_store[n] = nu * f

	# work out what the proxy frequencies and energies are and mark them
	nu1 = 2.057e10 # 20 GHz 
	nu2 = 0.44 * 1e6 / HEV 
	ax1.vlines([nu1,nu2], 1e38,1e45, ls="-.", lw=1.5)
	Eproxy = np.sqrt(H * MELEC * C * C * nu1 * 4e13 / 1e-5) / EV2ERGS

	print (BOLTZMANN * 2.7 / EV2ERGS, Eproxy/1e6)

	#if load == False:
	#	np.save("array_saves/ne_store_{}.npy".format(savename))
	#	np.save("array_saves/sed_store_{}.npy".format(savename))
	t = j.time/unit.myr
	# print (np.max(t), t[-1])
	norm = matplotlib.colors.Normalize(vmin=t[0], vmax=t[-1])
	mappable1 = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap )

	#ax1 = plt.gca()
	axins1 = inset_axes(ax1,width="60%",  height="15%", loc='lower left', bbox_to_anchor=(.1, .25, .4, .4), bbox_transform=ax1.transAxes)

	E = np.logspace(-7,0)
	plt.plot(E, 1e44 * E**0.5)
	axins2.vlines([Eproxy/1e6], 1e-4,10, ls="-.", lw=1.5)


	cbar = plt.colorbar(mappable=mappable1, cax = axins1, shrink=1, orientation="horizontal")
	cbar.set_label("$t$~(Myr)", fontsize=12)
	#ax1.set_ylabel("$E^2 dN/dE$")
	ax1.set_xlabel(r"$\nu$ (Hz)", fontsize=18)
	ax1.set_ylabel(r"$\nu F_\nu$ (erg~s$^{-1}$)", fontsize=18)
	axins2.set_xlabel("$E$ (MeV)")
	axins2.set_ylabel("$E^2 n(E)$")
	#ax1.set_xlabel("$E$ (eV)")
	ax1.set_ylim(1e38,1e45)
	ax1.set_xlim(1e8,1e25)
	ax2.set_xlabel("$t$~(Myr)", fontsize=14)
	ax2.set_ylabel(r"$\log[Q_j~(\mathrm{erg~s}^{-1})]$", fontsize=16)
	ax2.set_xlim(0,np.max(j.time/unit.myr))
	ax2.set_ylim(43,47)
	axins2.set_xlim(10,1e6)
	axins2.set_ylim(1e-4,10)
	plt.subplots_adjust(top=0.98, bottom=0.1, right=0.98, hspace=0.2, wspace=0.2)

	xray_kev = 1000.0 / 4.13620e-15
	ax1.plot([2*xray_kev, 20*xray_kev], [1e39,1e39], alpha=0.3, lw=15)
	ax1.text(5*xray_kev, 3e38, "X-ray\n2-20 keV", c="C0", horizontalalignment='center')
	ax1.plot([2e7/4.13620e-15, 3e11/4.13620e-15], [1e39,1e39], alpha=0.3, lw=15)
	ax1.text(9e8/4.13620e-15, 3e38, "Fermi LAT\n20MeV-300 GeV", c="C1", horizontalalignment='center')
	ax1.plot([1.44e8,5e9], [1e39,1e39], alpha=0.3, lw=15)
	ax1.text(1e9, 3e38, "Radio\n0.1-5 GHz", c="C2", horizontalalignment='center')
	ax1.text(1.4e9, 3e44, "Broadband SED", fontsize=14)
	axins2.text(15,1e-3, "$e^-$ spectrum", fontsize=12)

	plt.savefig("paper-figures-{}/sed.png".format(savename), dpi=300)

if __name__ == "__main__":

	savenames = [f for f in os.listdir("array_saves") if "dim_ref_p1e+45_geo4_etah0.3" in f]

	for savename in savenames[1:]:
		#create_and_save_spectra(sigma, power, beta, seed, nspectra=n)
		make_plots(savename[4:-4], nspectra=1500, load=False)
