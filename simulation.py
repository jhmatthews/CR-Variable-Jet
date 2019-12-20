import scipy.stats as st
import matplotlib.pyplot as plt 
from constants import *
from scipy.interpolate import interp1d
import naima
import astropy.units as u
import subroutines as sub
import os
import pickle
import synchrotron
from scipy.signal import savgol_filter

def power_threshold(rigidity, v_over_c=0.1, eta=0.1):
    power = (0.1 / eta) * (rigidity / 1e19)**2 * (0.1/v_over_c) * 1e44
    return power 

def max_energy(power, v_over_c=0.1, eta=0.1):
    return 1e19*np.sqrt( (power/1e44) * (v_over_c/0.1))


def run_jet_simulation(energies, flux_scale, BETA, lc, tau_loss, 
                       frac_elem=[1.0,0.1,1e-4,3.16e-05], R0=1e9,
	                   plot_all = False, sigma=1.5, NMAX=1000, NRES=20):
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
    #elems = ["H", "He", "N", "Fe"]
    species = ["e", "H", "He", "N", "Fe"]
            
    # normalise just in case not already! 
    frac_elem /= np.sum(frac_elem)

    # charges 
    z_elem = np.array([1,2,7,26])

    # array to store the current CR spectrum for each ion
    ncr = np.zeros( (len(species),len(energies)) )

    # set up flux array
    flux = lc.flux * flux_scale 

    # arrays to store the stored and escaping CRs for every time bin and every ion
    # these get returned by the function
    NWRITE = (NMAX // NRES) + 1
    ncr_time = np.zeros( (len(species), NWRITE,len(energies)) )
    escaping_time = np.zeros( (len(species), NWRITE,len(energies)) )

    # Zbar is the average charge 
    Zbar = np.sum(frac_elem * z_elem)

    lgamma = np.zeros_like(flux)

    # initialise the jet 
    jet = JetClass(lc)
    jet.init_atmosphere(1.0, 50.0, 0.5)
    jet.init_jet(power_norm = flux_scale, rho_j = 1e-4)

    # this is a class to store variables over time 
    jet_store = JetStore()

    dimensions = np.zeros( (2, NWRITE, len(jet.z)) )

    tmax = np.max(lc.time)
    i = 0
    
    print (len(flux))

    while jet.length < (200.0 * unit.kpc) and i < NMAX:

        if (i % NRES) == 0:
            write = True
            iwrite = (i // NRES)
        else:
            write = False

        #print (i, jet.length)

        

        jet.UpdateSolution()
        jet.lcr = 0.0
        jet.sync = 0.0

        #print (i, jet.length, NMAX, len(jet_store.time))
        # power = jet.power

        # get the maximum rigidity from the power requirement
        Rmax = max_energy(jet.power, v_over_c=jet.v_j)
            
        # put 10% of jet energy into CRs, store in eV units 
        Q0 = 0.1 * jet.power / EV2ERGS

        # get the time step - IMPROVE 
        #delta_t = lc.tbin * 1000.0 * YR
        delta_t = jet.dt
        
        if plot_all:
            # this plots the lightcurve 
            fig, ax2, ax3 = sub.init_lc_plot(lc.time[:i+1] / 1000.0, flux[:i+1])
        
        lcr = 0.0

        for j, part in enumerate(species):

            if part == "e": #electrons 
                if jet.set_aspect:
                    cutoff = 1e14
                else:
                    cutoff = 1e12
                R0=1e6
                meanZ = 1.0
                frac = 1.0
                z = 1.0
                rigidities = energies / z

                cooling_time = (1e9 / rigidities) * jet.tsync_gev
            else:
                cutoff = Rmax
                R0 = 1e9
                meanZ = Zbar
                frac = frac_elem[j-1]
                z = z_elem[j-1]
                rigidities = energies / z

                escape_time = (1e19 / rigidities) * jet.tescape_1e19


            # normalisation of cosmic ray distribution depends on BETA 
            if BETA == 2:
                dynamic = 1.0/(np.log(cutoff/R0))
            else:
                dynamic = (2 - BETA) / ((cutoff/R0)**(2-BETA)-1.0)

            # need to get things in the right units
            nu_i = frac / z / R0**2 * dynamic / meanZ

            # add_on is the injection term 
            add_on = nu_i * Q0 * ((rigidities / R0)**-BETA)

            # beyond the cutoff we set injected term to 0
            add_on[(rigidities>=cutoff)] = 0.0
            add_on[(rigidities<=R0)] = 0.0

            if part == "e":
                cooling = ncr[j] / cooling_time
                loss = 0.0
                escaping = np.zeros_like(ncr[j])
            else:
                # number of escaping CRs 
                escaping = ncr[j] / escape_time

                # loss rate 
                cooling =  ncr[j] / (tau_loss[part].total_interpol *  1e6 * YR)

            # this is dn/dt for each energy 
            change = (add_on - escaping - cooling)*delta_t

            # change the ncr distribution
            ncr[j] += change
            ncr[j][(ncr[j]<0)] = 0

            #if j == 0:
            if write:
                ncr_time[j,iwrite,:] = ncr[j]
                escaping_time[j,iwrite,:] = escaping

            # cutoff for UHECRs (60 EeV)
            select = (energies > 6e19)

            # store the UHECR luminosity 
            if part != "e":
            	jet.lcr += np.fabs(np.trapz(energies[select] * EV2ERGS, energies[select] * escaping[select]))
            elif write: # only compute at certain time steps 
                #jet.lsync = synchrotron.Ptot([1e9], energies, ncr[j], jet.B)
                jet.lsynch = 0.0
            
            if j == 0:
                #lcrtot = np.trapz(energies * EV2ERGS, energies * escaping)
                all_lobe = np.trapz(energies * EV2ERGS, energies * ncr[j])

            if plot_all:
                sub.plot_spectra(fig, ax2, ax3, rigidities * z_elem[j], ncr[j], escaping*delta_t)

        if write: 
            jet_store.Update(jet)
            dimensions[0,iwrite,:] = jet.z
            dimensions[1,iwrite,:] = jet.width

        i += 1

        if plot_all:
            os.system("mkdir spectra/beta{:.1f}q{:.1f}sig{:.1f}".format(BETA, np.log10(flux_scale), sigma))
            plt.savefig("spectra/beta{:.1f}q{:.1f}sig{:.1f}/spectra{:03d}.png".format(BETA, np.log10(flux_scale), sigma, i))
            plt.close("all")

    # save arrays to file 
    np.save("array_saves/escaping_beta{:.1f}q{:.1f}sig{:.1f}.npy".format(BETA, np.log10(flux_scale), sigma), escaping_time)
    np.save("array_saves/ncr_beta{:.1f}q{:.1f}sig{:.1f}.npy".format(BETA, np.log10(flux_scale), sigma), ncr_time)
    np.save("array_saves/dim_{:.1f}q{:.1f}sig{:.1f}.npy".format(BETA, np.log10(flux_scale), sigma), dimensions)

    fname_store = "array_saves/jetstore_{:.1f}q{:.1f}sig{:.1f}.npy".format(BETA, np.log10(flux_scale), sigma)
    jet_store.dump_to_file(fname_store)


    return (ncr_time, escaping_time, lcr)

class units:
    def __init__(self):
        self.kpc = 3.086e21
        self.pc = 3.086e18
        self.c = 2.997925e10
        self.yr = 3.1556925e7
        self.myr = 3.1556925e13
        self.kyr = 3.1556925e10
        self.radian = 57.29577951308232
        self.msol = 1.989e33
        self.mprot = 1.672661e-24
        self.ev = 1.602192e-12
        # self.thomson = 
        # self.


unit = units()

class JetStore:
    def __init__(self):
        self.B = np.array([])
        self.energy = np.array([])
        self.volume = np.array([])
        self.R_escape = np.array([])
        self.density = np.array([])
        self.v_j= np.array([])
        self.v_advance = np.array([]) 
        self.time = np.array([])
        self.length = np.array([])
        self.width = np.array([])
        self.tsync = np.array([])
        self.tesc = np.array([])
        self.lcr = np.array([])
        self.lsync= np.array([])

    def Update(self, jet):
        self.B = np.append(self.B, jet.B)
        self.energy = np.append(self.energy, jet.energy)
        self.volume = np.append(self.volume, jet.volume)
        self.R_escape = np.append(self.R_escape, jet.R_escape)
        self.density = np.append(self.density, jet.density)
        self.v_j = np.append(self.v_j, jet.v_j)
        self.v_advance = np.append(self.v_advance, jet.v_advance)
        self.time = np.append(self.time, jet.time)
        self.length= np.append(self.length, jet.length)
        self.width= np.append(self.width, jet.half_width)
        self.tsync = np.append(self.tsync, jet.tsync_gev)
        self.tesc = np.append(self.tesc, jet.tescape_1e19)
        self.lcr = np.append(self.lcr, jet.lcr)
        self.lsync = np.append(self.lsync, jet.lsync)
        self.jet = jet

    def dump_to_file(self, fname):
        FileObj = open("{}.pkl".format(fname), "wb")
        pickle.dump(self, FileObj)


# def lsynch_VDW(B, nu, ne):
# 	nu0 = 3.0 * E * B / (4.0 * PI * MELEC * C)
# 	x = 4.0 / 9.0 * (E * E / MELEC / C / C)**2 * (nu**-0.5) * (nu0**-1.5)
# 	x *= (B * B / 8.0 / PI) * 2.0 * C 


class JetClass:
    def __init__(self, lightcurve, nz=20000, zmax = 200):
        '''
        Initialise the jet class instance. set up the number of cells in
        the z direction, give it a lightcurve class, and set the time step

        IMPROVE TO USE KWARGS and defaults
        '''
        #self.dt = dt * unit.yr # should already be in years?
        self.lc = lightcurve 

        # lightcurve is in Kyrs, so convert
        self.interp_func = interp1d(self.lc.time * unit.kyr, self.lc.flux)
        self.z = np.linspace(0,zmax,nz) * unit.kpc 
        self.pressure = np.zeros_like(self.z)
        self.energy = 0.0
        self.density = 0.0
        self.lcr = 0.0
        self.set_aspect = True
        self.length = 0.0
        self.rho_j = 1e-4
        self.power = 1e43
        self.power_norm = 1e43
        self.area = (0.5 * unit.kpc)**2
        self.v_j = (2.0 * self.power / self.rho_j / unit.mprot / self.area)**(1./3.)
        self.v_perp = np.zeros_like(self.z)
        self.mdot = self.rho_j * self.v_j * self.area
        self.dz = zmax * unit.kpc / (nz  - 1)
        self.time = 0.0
        self.gamma = 4./3.
        self.zeta = 0.1
        self.kappa = 0.3
        #self.eta_b = 0.3
        self.efficiency_factor = 0.5
        self.R_escape = 0.0
        self.half_width = 0.0
        self.lsync = 0.0
        self.pressure_uniform = 0.0
        self.hs_pressure = 0.0


    def get_timescales(self):
        self.tsync_gev = 600.0 / 1e9 / unit.ev / (self.B**2)
        gyrofactor = 1.0
        tesc_ref = gyrofactor * 10.0 * unit.myr * (self.R_escape / 100.0 / unit.kpc)**2 
        self.tescape_1e19 = tesc_ref * (self.B / 1e-6)



    def init_jet(self, power_norm=1e43, rho_j=1e-4):
        self.rho_j = rho_j
        self.power_norm = power_norm

    def init_atmosphere(self, rho0, rcore, beta):
        '''
        initialise King profile parameters
        '''
        self.beta = beta
        self.rcore = rcore * unit.kpc 
        self.rho0 = rho0

    def density_function(self, r):
        '''
        get density at radius r according to King profile
        '''
        term = 1.0 + ((r/self.rcore)**2)
        #print (term**(-3.0 * self.beta / 2.0), (-3.0 * self.beta / 2.0), term)
        return self.rho0 * (term**(-3.0 * self.beta / 2.0))

    def pressure_function(self, r):
        '''
        get pressure at radius r 
        '''

        return (0.0)

    def equation_of_state(self, internal_energy):
        self.gamma = 4./3.
        return (self.gamma - 1) * internal_energy

    def get_v_advance(self):
        '''
        get the advance speed. Calls the density function and calculates density contrast
        '''
        rho = self.density_function(self.length)

        # get density contrast
        eta = np.sqrt(self.rho_j / rho) 

        hs_pressure = 0.5 * self.rho_j * unit.mprot * self.v_j * self.v_j
        if self.hs_pressure != 0.0:
            hs_pressure = 0.5 * self.rho_j * unit.mprot * self.v_j * self.v_j
            dp = hs_pressure - self.hs_pressure

            if np.fabs((dp) / (self.hs_pressure)) > 0.1:
                hs_pressure = self.hs_pressure + (np.sign(dp) * 0.1 * self.hs_pressure)

        self.hs_pressure = hs_pressure
        self.v_advance = eta * self.v_j

    def get_v_perp(self):
        '''
        get sideways expansion speed
        '''
        r = np.sqrt((self.z**2) + (self.width**2))
        rho = self.density_function(r)
        P_a = self.pressure_function(r)
        theta = np.arctan(self.z/self.width)

        select = (self.z <= self.length)
        #print (self.z, self.length, self.dz)
        phi = np.zeros_like(self.z[select])

        # find gradient of tangent and normal to the curve of the lobe
        grad_tan = np.gradient(self.z[select], self.width[select])
        grad_normal = -1.0 / grad_tan

        # get angle of shock normal 
        phi = np.arctan(grad_normal)

        # find cases where width = 0
        phi[(self.width[select]==0.0)] = 0

        #print (grad_normal, phi, self.length)
        # for i,z in enumerate(self.z[select]):
        #     if i == (len(self.z[select])-1):
        #         phi[i] = 0
        #     else:
        #         dx = np.fabs(self.width[i] - self.width[i+1])
        #         dz = np.fabs(self.z[i+1] - self.z[i])
        #         #if i == 0:
        #         phi[i] = np.arctan(dz / dx)
        #         #else:
        #         #    phi[i] = 0.5 * (np.arctan(dz / dx)+phi[i-1])

        #print (phi)
        
        #pressure = (self.z / self.length) * 

        self.v_perp[select] = np.sqrt( (self.pressure[select] - P_a) / (rho[select] * unit.mprot)) * np.cos(phi)

        # ensure points beyond jet head cannot expand sideways 
        self.v_perp[~select] = 0.0
        #print ("Pressure:", self.v_perp, self.pressure, self.energy, self.volume, rho)

        if np.isnan(self.v_perp).any():
            print ("vperp exit")
            print (self.pressure[select], rho[select], phi)
            import sys
            sys.exit()

        return (self.v_perp) 

    def initial_width(self, z, L):
        w = L / np.pi * np.arccos(np.sqrt(z/L))
        w[(z>L)] = 0.0

        return w

    def set_power(self, time):
        self.power = self.interp_func(time) * self.power_norm
        self.v_j = (2.0 * self.power / self.rho_j / unit.mprot / self.area)**(1./3.)
        self.mdot = self.rho_j * self.v_j * self.area

    def centre_of_mass(self):
        '''
        get the centre of mass
        '''
        numerator = np.trapz(self.z * self.width * self.width, x=self.z)
        denominator = np.trapz(self.width * self.width, x=self.z)
        return (numerator/denominator)

    def mean_escape_distance(self):
        '''
        get the mean escape distance by working out the distance to 
        every point on the lobe edge from the centre of mass  
        '''
        zcm = self.centre_of_mass()

        # distance to one lobe
        dist1 = np.sqrt(self.width**2 + (self.z-zcm)**2)

        # distance to second lobe 
        dist2 = np.sqrt(self.width**2 + (self.z+zcm)**2)

        return (0.5 * (np.mean(dist1) + np.mean(dist2)))

    def set_lobe_pressure(self):
        P0 = self.pressure_uniform
        select_lobe = (self.z <= self.length)
        self.pressure[select_lobe] = P0
        R0 = 0.75 * self.length

        select_lobe_head = (self.z > R0) * select_lobe
        self.pressure[select_lobe_head] += (self.z[select_lobe_head] - R0) / (self.length - R0) * self.hs_pressure 
        if np.sum(select_lobe_head) > 5:
            self.pressure[select_lobe_head] = savgol_filter(self.pressure[select_lobe_head], 5, 3)
        normalise = np.trapz(np.pi * (self.width[select_lobe]**2) * self.pressure[select_lobe], x = self.z[select_lobe])
        #print (normalise)
        if self.energy == 0.0:
            print ("Error energy is zero!")
        normalise /= (self.gamma - 1.0) * self.energy 
        #print (normalise)
        #normalise = 1.0
        self.pressure /= normalise

        # if self.length > (10.0 * unit.kpc):
        #     import matplotlib.pyplot as plt 
        #     plt.plot(self.z[select_lobe], self.pressure[select_lobe], marker="o")
        #     print (P0, self.hs_pressure )
        #     import sys
        #     sys.exit()


    def UpdateSolution(self):
        '''
        update the jet solution
        '''
        # get velocities 
        if self.length > (0.1 * unit.kpc):
            self.set_power(self.time)
        self.get_v_advance()

        #print (self.v_advance/C)
        
        if self.set_aspect == False: 
            self.get_v_perp()
            self.dt = self.dz / self.v_advance 
        else:
            self.dt = self.dz / self.v_advance   

        # evolve width 
        self.length += self.v_advance * self.dt
        self.energy += self.efficiency_factor * self.power * self.dt 
        self.density += self.mdot * self.dt

        if self.set_aspect:
            #self.length = 0.0
            if (self.length == 0.0): print ("ERRROR", self.dz, self.dt)
            self.width = self.initial_width(self.z, self.length)
            self.set_aspect = False
            self.i = 1
        else:
            self.width += self.v_perp * self.dt
            self.width = savgol_filter(self.width, 5,3)
            self.i += 1

        self.half_width = self.width[np.argmin(np.fabs(self.z-(self.length*0.5)))]

        # find where the lobe ends 
        select_lobe = (self.z <= self.length)

        # get the volume by integration 
        self.volume = np.trapz(np.pi * (self.width[select_lobe]**2), x = self.z[select_lobe])
        #self.volume = 4.0 / 3.0 * np.pi * self.width[0] * self.width[0] * self.length
        #print ("Vol:", self.volume, self.width, self.z)

        self.internal = (self.energy / self.volume)
        self.pressure_uniform = self.equation_of_state(self.internal)
        self.pressure_array = self.set_lobe_pressure()


        self.B = np.sqrt(3.0 * self.pressure_uniform * self.zeta * 8.0 * np.pi / (1.0 + self.zeta + self.kappa))
        self.time += self.dt

        #self.R_escape = self.mean_escape_distance()
        # IMPROVE 
        self.R_escape = self.volume ** (1./3.)

        # get escape and synchrotron timescales
        self.get_timescales()

        # if self.i > 20: 
        #     import sys
        #     sys.exit()
        if np.isnan(self.volume):
            print ("Error: NAN volume exit!")

            import sys
            sys.exit()




