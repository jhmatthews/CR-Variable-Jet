import scipy.stats as st
import matplotlib.pyplot as plt 
from constants import *
from scipy.interpolate import interp1d
import naima
import astropy.units as u
import subroutines as sub
import os

def power_threshold(rigidity, v_over_c=0.1, eta=0.1):
    power = (0.1 / eta) * (rigidity / 1e19)**2 * (0.1/v_over_c) * 1e44
    return power 

def max_energy(power, v_over_c=0.1, eta=0.1):
    return 1e19*np.sqrt( (power/1e44) * (v_over_c/0.1))


def run_jet_simulation(energies, flux_scale, BETA, lc, tau_loss, 
                       frac_elem=[1.0,0.1,1e-4,3.16e-05], R0=1e9,
	                   plot_all = False, sigma=1.5):
    '''
    Run a jet simulation with given flux_scale and spectral index BETA.

    Parameters:
    	energies			array-like
    						array of CR energies in eV to consider 
    	flux_scale 			float
    						median of jet power PDF 
    	BETA 				float
    						injection spectral index
    	lc 					DELCgen.Lightcurve object
    						containing light curve and time bins, amplitude should be one
    	R0					float
    				 		Injection rigidity
    	frac_elem			array-like
    						Injection abundances
    	plot_all			Bool
    						make plots or not

    Returns:
    	ncr 				5 x len(flux) x len(energies) array
    						CR spectra inside lobe
    	escaping 			5 x len(flux) x len(energies) array
    						Escaping CR spectra
    	lcr 				CR luminosity above 60 EeV
    '''
    # elemental "abundances" - really injection fractions 
    frac_elem = np.array(frac_elem)
    elems = ["H", "He", "N", "Fe"]
            
    # normalise just in case not already! 
    frac_elem /= np.sum(frac_elem)

    # charges 
    z_elem = np.array([1,2,7,26])

    # array to store the current CR spectrum for each ion
    ncr = np.zeros( (len(z_elem),len(energies)) )

    # set up flux array
    flux = lc.flux * flux_scale 

    # arrays to store the stored and escaping CRs for every time bin and every ion
    # these get returned by the function
    ncr_time = np.zeros( (len(z_elem),len(flux),len(energies)) )
    escaping_time = np.zeros( (len(z_elem),len(flux),len(energies)) )

    # Zbar is the average charge 
    Zbar = np.sum(frac_elem * z_elem)

    lgamma = np.zeros_like(flux)
    
    for i in range(len(flux)):

        # get the maximum rigidity from the power requirement
        Rmax = max_energy(flux[i], v_over_c=0.5)

        # normalisation of cosmic ray distribution depends on BETA 
        if BETA == 2:
            dynamic = 1.0/(np.log(Rmax/R0))
        else:
            dynamic = (2 - BETA) / ((Rmax/R0)**(2-BETA)-1.0)
            
        # put 10% of jet energy into CRs, store in eV units 
        Q0 = 0.1 * flux[i] / EV2ERGS

        # get the time step - IMPROVE 
        delta_t = lc.tbin * 1000.0 * YR
        
        if plot_all:
            # this plots the lightcurve 
            fig, ax2, ax3 = sub.init_lc_plot(lc.time[:i+1] / 1000.0, flux[:i+1])
        
        lcr = 0.0

        for j, frac in enumerate(frac_elem):

            rigidities = energies / z_elem[j]

            # escape time is 1e7 YR for 10EV rigidity 
            escape_time = (1e19 / rigidities) * 1e7 * YR 

            # need to get things in the right units
            nu_i = frac / z_elem[j] / R0**2 * dynamic / Zbar

            # add_on is the injection term 
            add_on = nu_i * Q0 * ((rigidities / R0)**-BETA)

            # beyond the cutoff we set injected term to 0
            add_on[(rigidities>=Rmax)] = 0.0
            add_on[(rigidities<=R0)] = 0.0

            # number of escaping CRs 
            escaping = ncr[j] / escape_time

            # loss rate 
            loss =  ncr[j] / (tau_loss[elems[j]].total_interpol *  1e6 * YR)

            # this is dn/dt for each energy 
            change = (add_on - escaping - loss)*delta_t

            # change the ncr distribution
            ncr[j] += change
            ncr[j][(ncr[j]<0)] = 0

            #if j == 0:
            ncr_time[j,i,:] = ncr[j]
            escaping_time[j,i,:] = escaping

            # cutoff for UHECRs (60 EeV)
            select = (energies > 6e19)

            # store the UHECR luminosity 
            lcr = np.fabs(np.trapz(energies[select] * EV2ERGS, energies[select] * escaping[select]))
            
            if j == 0:
                #lcrtot = np.trapz(energies * EV2ERGS, energies * escaping)
                all_lobe = np.trapz(energies * EV2ERGS, energies * ncr[j])

            if plot_all:
                sub.plot_spectra(fig, ax2, ax3, rigidities * z_elem[j], ncr[j], escaping*delta_t)

        if plot_all:
            os.system("mkdir spectra/beta{:.1f}q{:.1f}sig{:.1f}".format(BETA, np.log10(flux_scale), sigma))
            plt.savefig("spectra/beta{:.1f}q{:.1f}sig{:.1f}/spectra{:03d}.png".format(BETA, np.log10(flux_scale), sigma, i))
            plt.close("all")

    # save arrays to file 
    np.save("array_saves/escaping_beta{:.1f}q{:.1f}sig{:.1f}.npy".format(BETA, np.log10(flux_scale), sigma), escaping_time)
    np.save("array_saves/ncr_beta{:.1f}q{:.1f}sig{:.1f}.npy".format(BETA, np.log10(flux_scale), sigma), ncr_time)

    return (ncr_time, escaping_time, lcr)