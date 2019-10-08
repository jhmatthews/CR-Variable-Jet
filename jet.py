from DELCgen import *
import scipy.stats as st
import matplotlib.pyplot as plt 
from constants import *
from scipy.interpolate import interp1d
#from scipy.integrate import qua
import naima
import astropy.units as u

# File Route
route = "./"
datfile = "NGC4051.dat"

class Losses:
	def __init__(self, elem_name):
		self.energies, self.total, self.pp, self.pd, self.ep = np.loadtxt("tau_{}.dat".format(elem_name), unpack=True)

	def interpol(self, energies):
		interp_func = interp1d(self.energies, self.total)
		self.total_interpol = interp_func(energies)

def get_gamma_ray(energies, ncr, density=1e-4, int_range=(1e12,1e13)):
	select = (energies > int_range[0]) * (energies < int_range[1])
	ntot = np.trapz(energies[select], energies[select]*ncr[select])

	return (ntot * density * 0.5 * 4.5e-26)

def get_gamma_ray_sed(energies, ncr, Epmin = 1e9, Epmax = 1e20):

	interp_func = interp1d(energies * u.eV, ncr)
	def PD(my_energy, Emin):
		try: 
			f = interp_func(energies)
		except ValueError:
			f = 0.0
		return (f * u.Unit('1/eV'))

	ECPL = PD
	spectrum_energy = energies * u.eV
	nepd = len(spectrum_energy) / np.log10(spectrum_energy[-1]/spectrum_energy[0])
	gamma = naima.models.PionDecay(ECPL, nh=1e-4*u.Unit('1/cm3'), Epmin = Epmin * u.eV, Epmax=Epmax * u.eV, nEpd=nepd)
	sed = gamma.sed(spectrum_energy, distance=3.7*u.Mpc)
	return sed /sed.unit

def get_synchrotron_spectrum(energies, ncr, B=100.0):
	interp_func = interp1d(energies * u.eV, ncr)
	def PD(my_energy, Emin):
		try: 
			f = interp_func(energies)
		except ValueError:
			f = 0.0
		return (f * u.Unit('1/eV'))

	ECPL = PD
	spectrum_energy = energies * u.eV
	nepd = len(spectrum_energy) / np.log10(spectrum_energy[-1]/spectrum_energy[0])
	SYN = naima.models.Synchrotron(ECPL, B=B*u.uG, Epmin = Epmin * u.eV, Epmax=Epmax * u.eV, nEpd=nepd)
	sed = SYN.sed(spectrum_energy, distance=3.7*u.Mpc)
	return sed /sed.unit


def get_lgamma(energies, ncr):
	sed = get_gamma_ray_sed(energies, ncr)
	iarg = np.argmin(energies - 1e12)
	return (sed[iarg])

def power_threshold(rigidity, v_over_c=0.1, eta=0.1):
	power = (0.1 / eta) * (rigidity / 1e19)**2 * (0.1/v_over_c) * 1e44
	return power 

def max_energy(power, v_over_c=0.1, eta=0.1):
	return 1e19*np.sqrt( (power/1e44) * (v_over_c/0.1))


def get_lc(lognorm_params, PSD_params, tbin, Age):
	# Simulation params
	# let's do everything in units of kyr
	# run for 100 Myr (1e5 kyr) in bins of 0.1 Myr
	#lognorm_params = (1.5,0,np.exp(1.5))
	RedNoiseL,RandomSeed,aliasTbin = 100,12,100
	N = Age / tbin

	lc = Simulate_DE_Lightcurve(BendingPL, PSD_params,st.lognorm,lognorm_params,
	                                RedNoiseL=RedNoiseL,aliasTbin=aliasTbin,randomSeed=RandomSeed,LClength=Age, tbin=tbin)

	return (lc)


if __name__ == "main":
	# set up light curve 
	pl_index = 1
	# bend the power law at 1 Myr (1e-3 kyr^-1) and steeply decay it beyond that with index 10. 
	# A,v_bend,a_low,a_high,c = 1, 1e-3, pl_index, 10, 1
	PSD_params = (1, 1e-3, pl_index, 10, 1)
	lognorm_params = (1.5,0,np.exp(1.5))
	tbin = 100 # 100 kyr 
	# Age of 1000 means 100 Myr
	# lognorm, PSD, tbin and Age
	lc = get_lc(lognorm_params, PSD_params, tbin,1000)

	# get losses
	elems = ["H", "He", "N", "Fe"]
	tau_loss = dict()
	energies = np.logspace(6,20,num=3000)
	for i in range(len(elems)):
		tau_loss[elems[i]] = Losses(elems[i])
		tau_loss[elems[i]].interpol(energies)


#Comparison_Plots([lc,lc])
#lc.Plot_Lightcurve()
lc.Save_Lightcurve('lightcurve.dat')

# time is in kyr, so convert to Myr
times = np.arange(0,1000.0*tbin,tbin) 

# flux_scale = 1e43
# powers = power_threshold(energies, v_over_c=0.5)
# fraction = np.zeros_like(energies)
# for_cdf = powers / flux_scale

# for i, energy in enumerate(energies):
# 	fraction[i] = np.sum(flux > powers[i]) / (1.0*N)


flux_scales = np.logspace(43,45,num=20)
betas = np.arange(2,3,0.1)

lgammas = np.zeros((len(flux_scales), len(betas)))
lcrs = np.zeros((len(flux_scales), len(betas)))

for i_flux, flux_scale in enumerate(flux_scales):
	#flux_scale = 1e43
	flux = lc.flux * flux_scale 
	#flux = lc.flux

#plt.clf()

	for i_beta, BETA in enumerate(betas):


		frac_elem = np.array([1.0,0.1,1e-4,3.16e-05])
		#frac_elem = np.array([1.0,0,0,0])
		frac_elem /= np.sum(frac_elem)
		z_elem = np.array([1,2,7,26])
		ncr = np.zeros( (len(z_elem),len(energies)) )
		ncr_time = np.zeros( (4,len(flux),len(energies)) )
		escaping_time = np.zeros( (4,len(flux),len(energies)) )

		# do you want to plot everything? 
		plot_all = False

		R0 = 1e9
		Zbar = np.sum(frac_elem * z_elem)

		#print (np.mean(flux), 3e59 / np.mean(flux) / 1e6 / YR)

		lgamma = np.zeros_like(flux)

		for i in range(len(flux)):

			# get the maximum rigidity
			Rmax = max_energy(flux[i], v_over_c=0.5)
			#print (i, Rmax)

			# normalisation of cosmic ray distribution
			if BETA == 2:
				dynamic = 1.0/(np.log(Rmax/R0))
			else:
				dynamic = (2 - BETA) / ((Rmax/R0)**(2-BETA)-1.0)
			Q0 = 0.1 * flux[i] / EV2ERGS

			# get the time step 
			delta_t = tbin * 1000.0 * YR

			if plot_all:
				plt.clf()
				plt.figure()
				plt.subplot(211)
				plt.plot(lc.time[:i+1] / 1000.0, flux[:i+1])
				plt.semilogy()
				plt.xlabel("t (Myr)")
				plt.ylabel("$Q_j$ (erg/s)")

			lcr = 0.0

			for j, frac in enumerate(frac_elem):

				rigidities = energies / z_elem[j]

				# escape time is 1e7 YR?
				escape_time = (1e19 / rigidities) * 1e7 * YR 

				# need to get things in the right units
				nu_i = frac / z_elem[j] / R0**2 * dynamic / Zbar

				# add on is the injection term 
				add_on = nu_i * Q0 * ((rigidities / R0)**-BETA)

				# beyond the cutoff we set injected term to 0
				add_on[(rigidities>=Rmax)] = 0.0
				add_on[(rigidities<=R0)] = 0.0

				#if j == 0:
				#	print ('test:', np.trapz(energies* EV2ERGS, energies * add_on * delta_t), Q0)

				# number of escaping CRs 
				escaping = ncr[j] / escape_time

				# loss rate 
				loss =  ncr[j] / (tau_loss[elems[j]].total_interpol *  1e6 * YR)

				change = (add_on - escaping - loss)*delta_t

				# change the ncr distribution
				ncr[j] += change
				ncr[j][(ncr[j]<0)] = 0

				#if j == 0:
				ncr_time[j,i,:] = ncr[j]
				escaping_time[j,i,:] = escaping

				select = (energies > 6e19)

				#if j == 0:
				lcr += np.trapz(energies[select] * EV2ERGS, energies[select] * escaping[select])
				
				if j == 0:
					#lcrtot = np.trapz(energies * EV2ERGS, energies * escaping)
					all_lobe = np.trapz(energies * EV2ERGS, energies * ncr[j])
					#print (lcr, lcrtot, all_lobe)

				#if j == 0:
				#	lgamma[i] = get_lgamma(energies, ncr[j])
				if plot_all:
					plt.subplot(223)
					energy = rigidities * z_elem[j]
					plt.plot(energy, energy*energy*ncr[j])
					plt.loglog()
					plt.xlim(1e17,1e21)
					plt.ylim(1e70,1e75)

					plt.subplot(224)
					plt.plot(energy, energy*energy*escaping*delta_t)

					# plt.plot(energies, 1e15*((energies / 1e9)**-2), c="k", ls="--")
					plt.xlim(1e17,1e21)
					plt.ylim(1e67,1e72)
					#plt.plot(energies, 1e43*((energies / 1e9)**-3), c="k", ls=":")
					plt.loglog()

			if plot_all:
				plt.savefig("spectra{:03d}.png".format(i))	
				plt.close("all")

		select = (energies > 1e10) * (energies < 2e10)
		my_lgamma = np.trapz(energies * EV2ERGS, energies*ncr[0])
		distance = 3.7 * PARSEC * 1e6
		lgammas[i_flux,i_beta] = my_lgamma * 1e-4 * 4.5e-26 * 0.5 * C / 4.0 / PI / distance / distance
		lcrs[i_flux,i_beta] = lcr
		print ("BETA {:.1f} median lum {:8.4e} mean lum {:8.4e} UHECR LUM: {:8.4e} {:8.4e}".format(BETA, flux_scale, np.mean(flux), lcr, lgammas[i_flux,i_beta]))

print (np.mean(flux), 3e59 / np.mean(flux) / 1e6 / YR)
import naima
import astropy.units as u
#ECPL
ncr = ncr[0]
#print (energies)
spectrum_energy = np.logspace(6,20,num=3000) * u.eV
interp_func = interp1d(energies * u.eV, ncr)
def PD(my_energy, Emin=1e9):
	#if my_energy/u.eV > Emin:
	try: 
		f = interp_func(spectrum_energy)
	except ValueError:
		f = 0.0
	
	#if (my_energy < Emin):
	#	f = 0.0;
	return (f * u.Unit('1/eV'))

EE, ff = np.loadtxt("gamma-rays-south.dat", unpack=True)
ECPL = PD
nepd = len(spectrum_energy) / np.log10(spectrum_energy[-1]/spectrum_energy[0])
gamma = naima.models.PionDecay(ECPL, nh=1e-4*u.Unit('1/cm3'), Epmin = 1e6 * u.eV, Epmax=10.0**20 * u.eV, nEpd=nepd)
sed = gamma.sed(spectrum_energy, distance=3.7*u.Mpc)
plt.loglog(spectrum_energy, sed)
plt.loglog(1e9*EE, ff, "o")
plt.xlim(1e7,1e15)
plt.ylim(1e-17,1e-9)
plt.savefig("gamma.png")



# #import sys
# #sys.exit()
# plt.figure(figsize=(6,10))
# times = np.arange(0,1000.0*tbin,tbin) 
# #ev_gamma_ray = np.trapz()
# plt.clf()
# plt.subplot(511) 
# plt.plot(times/1000.0, flux)
# plt.xlim(0,100)
# plt.semilogy()

# #ncr_time.T *= energies * energies
# for i in range(len(frac_elem)):
# 	plt.gca().set_xticklabels([])
# 	plt.subplot(5,1,i+2)
# 	logn = np.log10(ncr_time[i])
# 	to_plot =  (logn+np.log10(energies)+np.log10(energies)).T
# 	to_plot[np.isinf(to_plot)] = -999
# 	logfrac = np.log10(frac_elem[i])
# 	lims = (np.max(to_plot)-4,np.max(to_plot))
# 	plt.pcolormesh(times/1000.0, np.log10(energies), to_plot, vmin=lims[0], vmax=lims[1])  
# 	plt.xlim(0,100)
# 	plt.ylim(18,21)
# 	plt.colorbar(orientation="horizontal")

# plt.subplot(211)
# plt.plot(lc.time[:i+1] / 1000.0, flux[:i+1])
# plt.semilogy()
# plt.xlabel("t (Myr)")
# plt.ylabel("Q_j (erg/s)")

# plt.subplot(234)
# #plt.plot(energies, fraction*100.0)
# powers2 = np.arange(40,47,0.1)
# for_pdf = powers2 / flux_scale
# plt.hist(np.log10(flux), log=True, density=True, bins=powers2)
# plt.plot(np.log10(powers2), st.lognorm.pdf(for_pdf, lognorm_params[0], lognorm_params[1], lognorm_params[2]), c="r")
# #plt.semilogx()
# plt.xlabel("$E_{max}$ (eV)")
# plt.ylabel("PDF")
# plt.xlim(40,47)

# plt.subplot(235)
# plt.plot(energies, fraction*100.0)
# plt.plot(energies, 1e4*st.lognorm.sf(for_cdf, lognorm_params[0], lognorm_params[1], lognorm_params[2]))
# plt.loglog()
# plt.xlabel("$E_{max}$ (eV)")
# plt.ylabel("Survival Function")

# plt.subplot(236)
# plt.xlabel("E (eV)")
# plt.ylabel("n(E)")

# #lt.plot(energies, ncr)
# plt.plot(energies, ncr/escape_time)
# plt.plot(energies, 1e15*((energies / 1e9)**-2), c="k", ls="--")
# plt.xlim(1e15,1e21)
# #plt.plot(energies, 1e43*((energies / 1e9)**-3), c="k", ls=":")
# plt.loglog()
# plt.show()	






#tklc = Simulate_TK_Lightcurve(BendingPL, (1.0,300,2.1,2.3,0.1),
#				st.lognorm,(0.3, 0.0, 7.4),
#                                  tbin = 1, LClength = 1000, mean = 1, std = 0.1)
#--------- Commands ---------------                            

#lc.Plot_Lightcurve()

#lc.Plot_Periodogram()

# # Save lightcurve and Periodogram as text files
# delc.Save_Lightcurve('lightcurve.dat')
# delc.Save_Periodogram('periodogram.dat')