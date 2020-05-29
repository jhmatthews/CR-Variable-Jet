import scipy.stats as st
import matplotlib.pyplot as plt 
from constants import *
from scipy.interpolate import interp1d
import astropy.units as u
import subroutines as sub
import os
import pickle
from scipy.signal import savgol_filter
from DELCgen import *
from scipy.optimize import fsolve
from msynchro.units import unit
import msynchro

def get_elem_dict(fname="abundances.txt", beta=2):
    elem = dict()
    Z, A, name, logF = np.genfromtxt(fname, unpack=True, dtype=str)
    elem["z"] = Z.astype(float)
    elem["a"] = A.astype(float)
    fsol = 10.0 ** logF.astype(float)
    elem["frac"] =  fsol * elem["z"] * elem["z"] * (elem["a"] ** (beta-2.0))
    elem["species"] = name

    return (elem)

class my_powerlaw:
    '''
    container for generating random numbers from a powerlaw with slope < 1. 
    For a power of form x**-n, with n>1, from xmin to xmax, the resulting
    CDF can be shown to take the analytic form

        CDF(x) = (xmin^alpha - x^alpha) / (xmin^alpha - xmax^alpha)

    where alpha = 1-n. Thus, if z is a random number in range (0,1),
    then a random variable rv can be found by inverting this expression,
    i.e. rv = [z * (b^alpha - a^alpha) + a^alpha] ^ (1/alpha).

    This is embedded in a class so an individual function can be passed without
    additional arguments, the function rvs() generates the actual random numbers
    similarly to scipy.stats distribution syntax.
    '''
    def __init__(self, n=1.2, xmin=3.5, xmax=10.0):
        '''
        initialise the powerlaw parameters
        '''
        if n <= 1:
            raise ValueError("n must be > 1!")
        self.n = n
        self.alpha = 1.0 - self.n
        self.xmin = xmin 
        self.xmax = xmax 

    def rvs(self, size=None):
        '''
        generate (size) random variables. if size is None, 
        generate a single float.
        '''

        # note that np.random.random generates a single float if size==None
        z = np.random.random(size=size)

        term1 = z * (self.xmax ** self.alpha)
        term2 = (1. - z) * (self.xmin ** self.alpha)
        rv = (term1 + term2) ** (1.0 / self.alpha)
        return (rv)

def cooling_rate(gammas, B, B_CMB=3.24e-6):
    '''
    Get the electron cooling rate from synchrotron and inverse Compton.
    returns CGS units. 

    Parameters:
        gammas  array-like
                Lorentz factors of electrons 

        B       float 
                magnetic field in Gauss 

    Returns:
        Cooling rate dE/dt in erg/s
    '''
    Utot = (B**2 + B_CMB**2) / 8.0 / np.pi

    x = 4.0 / 3.0 * unit.thomson * unit.c * Utot * gammas * gammas

    return (x)



def universal_profile(r, M500, H0=70.0):
    """
    Universal pressure profile of Arnaud et al. 2010, A&A, 517, 92
    http://adsabs.harvard.edu/abs/2010A%26A...517A..92A.

    Keyword arguments:
    r -- radii at which to calculate (in cm)
    M500 -- enclosed mass within R500, (in solar masses)
    H0 -- hubble constant in km/s/Mpc (not yet used)

    Returns:
    pressure -- array -- pressure in CGS, same shape as r
    temperature -- float -- temperature in K
    density -- array -- density in CGS, same shape as r
    R500 -- float -- radius at which density is 500x critical
    """

    # best fitting parameters from Arnaud et al. eq 12
    P0 = 8.403 
    c500 = 1.177
    gamma = 0.3081
    alpha = 1.0510
    beta = 5.4905
    h70 = H0/70.0       # could be used later 


    # R500 from M500 - radius at which density is "critical"
    R500 = 1104.0 * unit.kpc * ((M500/3.84e14) ** (1./3.))

    # get P(x), equation (11)
    x = r / R500
    px = P0 / ((c500 * x)**gamma)
    exponent = (beta - gamma) / alpha
    px /= (1 + ((c500 * x)**alpha)) ** exponent

    # P500 from equation (5)
    P500 = 1.65e-3 * ((M500 / 3.0e14)**(2.0/3.0))
    P500 *= 1000.0 * unit.ev

    # from Arnaud's equation (7)
    ap = 0.12
    xterm = (x/0.5)**3
    apx = 0.1 - ((ap+0.1) * (xterm / (1 + xterm)))


    # finally get pressure 
    pressure = P500 * px * ((M500 / 3e14)**(ap+apx))

    # temperature is a single value from T-M500 relation
    temperature = (M500 / 3.84e14) ** (1.0/1.71)    # temperature in 5 keV
    temperature *= 5000.0 * unit.ev / unit.kb       # temperature in Kelvin

    # density from ideal gas 
    density = pressure / unit.kb / temperature

    return (pressure, temperature, density, R500)

# def get_frac_elem(method):
#     if method == "wykes":
#         return ()
def get_lc(lognorm_params, PSD_params, tbin, Age, RandomSeed=12):
    # Get a DELCgen lightcurve object for a given set of PSD and PDF params 
    # we  do everything in units of kyr
    # run for Age  Myr (1e5 kyr) in bins of tbin Myr
    # lognorm_params = (1.5,0,np.exp(1.5))
    RedNoiseL,aliasTbin = 100,10
    N = Age / tbin

    # this is a hack to still use an lc class but have constant jet power
    if lognorm_params[0] == 0.0:
        lognorm_params = (1.0, lognorm_params[1], lognorm_params[2])
        lc = Simulate_DE_Lightcurve(BendingPL, PSD_params,st.lognorm,lognorm_params,
                                    RedNoiseL=RedNoiseL,aliasTbin=aliasTbin,randomSeed=RandomSeed,LClength=Age, tbin=tbin)

        lc.flux = np.ones_like(lc.flux)

    else:
        lc = Simulate_DE_Lightcurve(BendingPL, PSD_params,st.lognorm,lognorm_params,
                                    RedNoiseL=RedNoiseL,aliasTbin=aliasTbin,randomSeed=RandomSeed,LClength=Age, tbin=tbin)

    return (lc)



def power_threshold(rigidity, v_over_c=0.1, eta=0.1):
    power = (0.1 / eta) * (rigidity / 1e19)**2 * (0.1/v_over_c) * 1e44
    return power 

def max_energy(power, v_over_c = 0.1, eta_H = 0.3):
    return 1e19  * np.sqrt((power/1e44) * (v_over_c/0.1)) * eta_H

def get_source_term(energies, jet, part, BETA, Q0, Rmax, frac_elem, z_elem, j):

    Eichmann = False
    # Zbar is the average charge 
    Zbar = np.sum(frac_elem * z_elem)
    Rmax = Rmax/100.0

    if part == "e": #electrons 
        if jet.set_aspect:
            Rcutoff = 1e12
        else:
            Rcutoff = 1e14

        R0 = 1e6
        meanZ = frac = z = 1.0
        rigidities = energies / z
        Qnorm = Q0 * (1.0 - jet.kappa)
    else:
        cutoff = Rmax
        R0 = 1e9
        meanZ = Zbar
        frac = frac_elem[j-1]
        z = z_elem[j-1]

        Q0 = jet.kappa * Q0

    rigidities = energies / z
    E0 = R0 * z
    Emax = Rmax * z


    # normalisation of cosmic ray distribution depends on BETA 
    if BETA == 2:
        dynamic = 1.0/(np.log(Rmax/R0))
    else:
        dynamic = (2 - BETA) / ((Rmax/R0)**(2-BETA)-1.0)

    exp_term = np.exp(-(rigidities/Rmax)**3)

    if Eichmann:
        # need to get things in the right units
        nu_i = frac / z / R0**2 * dynamic / meanZ

        # add_on is the injection term 
        add_on = nu_i * Q0 * ((rigidities / R0)**-BETA) *  exp_term



    else:
        nu_i = frac * dynamic 
        add_on = nu_i * Q0 * (energies**-BETA) * exp_term
    
    # beyond the cutoff we set injected term to 0
    #add_on[(rigidities>=Rmax)] = 0.0
    add_on[(rigidities<=R0)] = 0.0
    #print (Rmax)

    return (add_on)


def run_jet_simulation(energy_params, flux_scale, BETA, lc, tau_loss, 
                       elem = None, R0=1e9,
	                   plot_all = False, sigma=1.5, NMAX=1000, NRES=20, tau_on=100.0, 
                       seed=0, save_arrays=True, savename=None):
    '''
    Run a jet simulation with given flux_scale and spectral index BETA.

    Parameters:
    	energy_params		tuple
    						tuple containing function to generate array,
                            min energy, max energy, and number of energy bins.
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
        tau_on              Float
                            Length of outburst in Myr

    Returns:
    	ncr 				5 x len(flux) x len(energies) array
    						CR spectra inside lobe
    	escaping 			5 x len(flux) x len(energies) array
    						Escaping CR spectra
    	lcr 				CR luminosity above 60 EeV
    '''

    # create energy bins for ions and electrons
    elec_e_edges = np.logspace(7, 14, energy_params[2] + 1)
    elec_energies = 0.5 * (elec_e_edges[1:] + elec_e_edges[:-1])
    prot_e_edges = np.logspace(14, 21, energy_params[2] + 1)
    prot_energies = 0.5 * (prot_e_edges[1:] + prot_e_edges[:-1])
    # number of energy bins
    N_energies = energy_params[2]

    # elemental "abundances" - really injection fractions 
    # if elem is None we set to default
    # otherwise use dictionary supplied 
    if elem == None:
        frac_elem = [1.0,0.1,1e-4,3.16e-05]
        species = ["e", "H", "He", "N", "Fe"]    
        z_elem = np.array([1,2,7,26])
    else:
        frac_elem = np.array(elem["frac"])
        species = np.concatenate( (np.array(["e"]), np.array(elem["species"])) )
        z_elem = np.array(elem["z"])
    frac_elem /= np.sum(frac_elem)      # make sure normalised 

    # array to store the current CR spectrum for each ion
    ncr = np.zeros( (len(species), N_energies) )

    # set up flux array
    flux = lc.flux * flux_scale 

    # arrays to store the stored and escaping CRs for every time bin and every ion
    # these get returned by the function
    NWRITE = (NMAX // NRES)
    ncr_time = np.zeros( (len(species), NWRITE, N_energies) )
    escaping_time = np.zeros( (len(species), NWRITE, N_energies) )

    # max time is in Myrs 
    TMAX = tau_on * unit.myr
    LMAX = 300.0 

    # initialise the jet 
    jet = JetClass(lc, env="UP", zmax = LMAX, nz = int (LMAX * 100))
    # only need this if it's a king profile
    # jet.init_atmosphere(1.0, 50.0, 0.5)
    jet.init_jet(power_norm = flux_scale, eta = 1e-5, f_edd_norm = 1e-4)

    # this is a class to store variables over time 
    jet_store = JetStore(jet.rho_j, prot_e_edges, elec_e_edges)

    if save_arrays:
        dimensions = np.zeros( (2, NWRITE, len(jet.z)) )

    i = 0
    failure = False

    while jet.length < (LMAX * unit.kpc) and i < NMAX and jet.time < TMAX and failure == False:

        if (i % NRES) == 0:
            write = True
            iwrite = (i // NRES)
        else:
            write = False

        # advance the jet solution, but check for errors and raise exceptions if necessary 
        try:
            ireturn = jet.UpdateSolution()
        except ValueError:
            ireturn = -1
            print ("ValueError from jet UpdateSolution: time {}".format(jet.time))
        if ireturn < 0:
            failure = True
            break

        # zero CR luminosities before looping over ion species 
        jet.lcr = 0.0
        jet.lcr_8EeV = 0.0

        # get the maximum rigidity from the power requirement
        Rmax = max_energy(jet.epsilon * jet.power, v_over_c=jet.v_j/unit.c, eta_H = 1)
            
        # put a percentage of jet energy into CRs, store in eV units 
        Q0 = jet.epsilon * jet.f_work * jet.power / EV2ERGS

        # get the time step - IMPROVE 
        delta_t = jet.dt


        for j, part in enumerate(species):

            ncr_old = ncr[j]
        
            if part == "e":
                energies = elec_energies
                energy_edges = elec_e_edges 
                tau_total = 0.0
                escaping = np.zeros_like(ncr[j])

                # get the Lorentz factors of electrons 
                gammas = (energy_edges * unit.ev) / unit.melec_csq
                gammas_cen = (energies * unit.ev) / unit.melec_csq 
                energy_loss_rate = cooling_rate(gammas, jet.B) / unit.ev
                loss_rate_cen = cooling_rate(gammas_cen, jet.B) / unit.ev
                tau_total = 0.0
            else:
                # number of escaping CRs
                energies = prot_energies
                energy_edges = prot_e_edges 
                rigidities = energies / z_elem[j-1]
                escape_time = (1e19 / rigidities) * jet.tescape_1e19
                escape_recip = 1.0 / escape_time
                cooling_recip = 1.0 / (tau_loss[part].total_interpol * unit.myr)
                tau_total = 1.0 / (cooling_recip + escape_recip)
                loss_rate_cen = 0.0
                energy_loss_rate = 0.0
                escaping = escape_recip * ncr_old

            #print (energy_loss_rate, jet.cool_adiabatic / unit.ev)
            energy_loss_rate += jet.cool_adiabatic * energy_edges

            #print ("Adiab:", jet.cool_adiabatic / unit.ev, energy_loss_rate)
            # this could be improved but at least it's in a subroutine
            source_term = get_source_term (energies, jet, part, BETA, Q0, Rmax, frac_elem, z_elem, j)
            
            # calculate the subcycle time step and evolve the particle distribution 
            dt_particle = msynchro.evolve.get_dt(energies, loss_rate_cen, tau_total, source_term, ncr[j])

            if dt_particle == 0.0 or np.isnan(dt_particle) or i == 0:
                n_subcycles = int(1)
            else:
                n_subcycles = int(delta_t // dt_particle) + 1
            dts = np.ones(n_subcycles) * (delta_t/n_subcycles)
            #print (n_subcycles, dt_particle, delta_t)
            for dt in dts:
                ncr[j] = msynchro.evolve.particle_evolve(energy_edges, energy_loss_rate, tau_total, source_term, ncr[j], dt)

            # check for zeros 
            ncr[j][(ncr[j]<0)] = 0


            # write the distribution to the global array if this is an integer 
            # multiple of iwrite 
            if write:
                ncr_time[j,iwrite,:] = ncr[j]
                escaping_time[j,iwrite,:] = escaping

            # cutoff for UHECRs (60 EeV)
            select_60eV = (energies > 6e19)
            select_8eV = (energies > 8e18)

            # store the UHECR luminosity or the synchrotron luminosity
            if part != "e":
                jet.lcr += np.fabs(np.trapz(energies[select_60eV] * unit.ev, energies[select_60eV] * escaping[select_60eV]))
                jet.lcr_8EeV += np.fabs(np.trapz(energies[select_8eV] * unit.ev, energies[select_8eV] * escaping[select_8eV]))

            elif write: # only compute synchrotron luminosities at certain time steps
                select = (energies <= 1e14) 
                jet.lsync = 0.0
                jet.l144 = 0.0
                jet.lsync = msynchro.Ptot([1.4e9], energies[select], ncr[j][select], jet.B)
                jet.l144 = msynchro.Ptot([1.44e8], energies[select], ncr[j][select], jet.B)

        if write: 
            jet_store.Update(jet)
            if save_arrays:
                dimensions[0,iwrite,:] = jet.z
                dimensions[1,iwrite,:] = jet.width

        i += 1

    # print out some info to user 
    if failure: 
        print ("Failure for for Q {} beta {} sigma {}".format(np.log10(flux_scale), BETA, sigma))
    else:
        print ("Finished sim for Q {} beta {} sigma {}".format(np.log10(flux_scale), BETA, sigma))
    
    print ("Final iterators: {}/{}, {}/{}Myr, {}/{}kpc".format(i, NMAX, jet.time/unit.myr, TMAX/unit.myr, jet.length/unit.kpc, LMAX))

    if savename == None:
        savename = "beta{:.1f}q{:.1f}sig{:.1f}seed{:d}".format(BETA, np.log10(flux_scale), sigma, seed)
    if save_arrays:
        # save arrays to file 
        np.save("array_saves/escaping_{}.npy".format(savename), escaping_time)
        np.save("array_saves/ncr_{}.npy".format(savename), ncr_time)
        np.save("array_saves/dim_{}.npy".format(savename), dimensions)

    # store the jet class whatever
    fname_store = "array_saves/jetstore_{}.npy".format(savename)
    jet_store.dump_to_file(fname_store)


    return (ncr_time, escaping_time, 0.0)

# class units:
#     def __init__(self):
#         self.kpc = 3.086e21
#         self.pc = 3.086e18
#         self.c = 2.997925e10
#         self.yr = 3.1556925e7
#         self.myr = 3.1556925e13
#         self.kyr = 3.1556925e10
#         self.radian = 57.29577951308232
#         self.msol = 1.989e33
#         self.mprot = 1.672661e-24
#         self.ev = 1.602192e-12
#         self.kb = 1.38062e-16
#         self.h = 6.6262e-27
#         self.g = 6.670e-8

# # class to use for units
# unit = units()

class JetStore:
    def __init__(self, eta, prot_e_edges, elec_e_edges):
        self.B = np.array([])
        self.energy = np.array([])
        self.volume = np.array([])
        self.R_escape = np.array([])
        self.density = np.array([])
        self.pressure = np.array([])
        self.v_j= np.array([])
        self.v_advance = np.array([]) 
        self.time = np.array([])
        self.length = np.array([])
        self.width = np.array([])
        self.tsync = np.array([])
        self.tesc = np.array([])
        self.lcr = np.array([])
        self.lcr_8EeV = np.array([])
        self.lsync = np.array([])
        self.l144 = np.array([])
        self.eta = eta
        self.prot_e_edges = prot_e_edges 
        self.elec_e_edges = elec_e_edges 
        self.power = np.array([])
        self.gmm = np.array([])
        self.P0 = np.array([])

    def Update(self, jet):
        self.B = np.append(self.B, jet.B)
        self.energy = np.append(self.energy, jet.energy)
        self.volume = np.append(self.volume, jet.volume)
        self.R_escape = np.append(self.R_escape, jet.R_escape)
        self.density = np.append(self.density, jet.density)
        self.pressure = np.append(self.pressure, jet.pressure_uniform)
        self.v_j = np.append(self.v_j, jet.v_j)
        self.v_advance = np.append(self.v_advance, jet.v_advance)
        self.time = np.append(self.time, jet.time)
        self.length= np.append(self.length, jet.length)
        self.width= np.append(self.width, jet.half_width)
        self.tsync = np.append(self.tsync, jet.tsync_gev)
        self.tesc = np.append(self.tesc, jet.tescape_1e19)
        self.lcr = np.append(self.lcr, jet.lcr)
        self.lcr_8EeV = np.append(self.lcr_8EeV, jet.lcr_8EeV)
        self.lsync = np.append(self.lsync, jet.lsync)
        self.l144 = np.append(self.l144, jet.l144)
        self.power = np.append(self.power, jet.power)
        self.gmm = np.append(self.gmm, jet.gmm)
        self.jet = jet
        self.P0 = np.append(self.P0, jet.pressure_uniform)


    def dump_to_file(self, fname):
        FileObj = open("{}.pkl".format(fname), "wb")
        pickle.dump(self, FileObj)


# def lsynch_VDW(B, nu, ne):
# 	nu0 = 3.0 * E * B / (4.0 * PI * MELEC * C)
# 	x = 4.0 / 9.0 * (E * E / MELEC / C / C)**2 * (nu**-0.5) * (nu0**-1.5)
# 	x *= (B * B / 8.0 / PI) * 2.0 * C 


class JetClass:
    def __init__(self, lightcurve, nz=20000, zmax=200, env="UP", debug=False):
        '''
        Initialise the jet class instance. set up the number of cells in
        the z direction, give it a lightcurve class, and set the time step

        IMPROVE TO USE KWARGS and defaults?
        '''
        #self.dt = dt * unit.yr # should already be in years?
        self.lc = lightcurve 


        # lightcurve is in Kyrs, so convert
        # also set a bunch of defaults. 
        # IMPROVE: organise this a bit better 
        self.interp_func = interp1d(self.lc.time * unit.kyr, self.lc.flux)
        self.z = np.linspace(0,zmax,nz) * unit.kpc 
        self.pressure = np.zeros_like(self.z)
        self.energy = 0.0
        self.density = 0.0
        self.mass = 0.0
        self.lcr = 0.0
        self.lcr_8EeV = 0.0
        self.set_aspect = True
        self.length = 0.0
        self.rho_j = 1e-4
        self.power = 1e43
        self.power_norm = 1e43
        self.area = np.pi * ((0.5 * unit.kpc)**2)
        self.v_j = (2.0 * self.power / self.rho_j / unit.mprot / self.area)**(1./3.)
        self.v_perp = np.zeros_like(self.z)
        self.mdot = self.rho_j * self.v_j * self.area
        self.dz = zmax * unit.kpc / (nz  - 1)
        self.time = 0.0
        self.gamma = 4./3.
        self.zeta = 1.0
        self.kappa = 0.5
        #self.eta_b = 0.3
        self.R_escape = 0.0
        self.half_width = 0.0
        self.lsync = 0.0
        self.pressure_uniform = 0.0
        self.hs_pressure = 0.0
        self.gmm = 2.0
        self.mu_weight = 1.0
        self.H0 = 70.0
        self.geometry_factor = 1.0
        self.rel_advance = True
        self.debug = debug
        self.epsilon = 0.3
        self.f_work = 0.5    
        self.E_B = 0.0
        self.power_available = self.f_work * self.power 


        # set profiles up 
        if env == "UP":
            self.atmosphere = env
            self.profile = self.UniversalProfile
            self.up_init = False
        elif env == "King":
            self.atmosphere = env
            self.king_init = False     
            self.profile = self.KingProfile
        else:
            raise KeyError("Atmosphere {} not supported. Choose King or UP".format(env))

    def _print(self, *args):
        '''print statement that depends on debug mode'''
        if self.debug:
            print( "Debug:".join(map(str,args)))

    def get_timescales(self):
        self.tsync_gev = 600.0 / 1e9 / unit.ev / (self.B**2)
        gyrofactor = 1.0
        tesc_ref = gyrofactor * 10.0 * unit.myr * (self.R_escape / 100.0 / unit.kpc)**2 
        self.tescape_1e19 = tesc_ref * (self.B / 1e-6)
        # self.tsync_Ghz


    def sound_speed(self, t, gamma=(5./3.), mu=1.0):
        p_over_rho = unit.kb * t / mu / unit.mprot 
        cs = np.sqrt(p_over_rho * gamma)
        return (cs)


    def init_jet(self, **kwargs):

        self.power_norm = kwargs["power_norm"]

        if self.atmosphere == "UP":
            self.f_edd_norm = kwargs["f_edd_norm"]

            # mass in solar masses - get the black hole mass 
            self.mbh = self.power_norm / self.f_edd_norm / 1.26e38 
            #self.M500_msol = ((self.mbh/1e9) / (10.0**0.49))**(1.0/0.77) * 1e14 
            # this is the Bogdan relation 
            exponent = (np.log10(self.mbh/1e9) + 0.22) / 1.07 
            self.M500_msol = (10.0 ** exponent) * 1e13
            #self.M500_msol = 1e14
            self.M500_cgs = self.M500_msol * unit.msol
            # self._print ("M500: {}".format(self.M500_msol))
            print ("M500: {}".format(self.M500_msol))
            self.mbh *= unit.msol

            # get rho0 and temperature of cluster at 1kpc 
            p, temperature, d, R500 = universal_profile(0.1 * unit.kpc, self.M500_msol, H0=self.H0)
            self.rho0 = d
            self.temperature = temperature

            # sound speed of isothermal atmosphere
            self.cs_env = self.sound_speed(self.temperature, gamma=(5./3.), mu=self.mu_weight)


        # rho_j is really number density 
        self.rho_j = kwargs["eta"] * self.rho0
        self.eta_density = kwargs["eta"]



        # calculate the gammas for the time series 
        gmm_for_interp = np.zeros_like(self.lc.flux)
        for i in range(len(self.lc.flux)):
            f = self.lc.flux[i] * self.power_norm

            # this is power over area over rho c**3
            X = f / self.area / self.rho_j / unit.mprot / (unit.c ** 3) 
            func = lambda gmm: np.sqrt(1-(gmm**-2)) * ((gmm**2) - gmm) - X

            # find gamma 
            gmm_for_interp[i] = fsolve(func, 3.0)[0]

        # do a linear interpolation to get gamma
        # second order creates issues because sometimes gamma < 1 due to inflections
        self.gamma_interp_func = interp1d(self.lc.time * unit.kyr, gmm_for_interp, kind="linear")

    def init_atmosphere(self, rho0, rcore, beta):
        '''
        Initialise King (beta) profile parameters
        '''
        self.beta = beta
        self.rcore = rcore * unit.kpc 
        self.rho0 = rho0
        self.king_init = True

        # default to 1 keV temperature - remember this is just a testing mode 
        self.temperature = 1.0 * 1000.0 * unit.ev / unit.kb



    def KingProfile(self, r):
        '''
        get density at radius r according to King profile
        '''
        if self.king_init == False:
            raise Exception("King profile has not been initialised. Need init_atmosphere step.")

        term = 1.0 + ((r/self.rcore)**2)
        density = self.rho0 * (term**(-3.0 * self.beta / 2.0))
        pressure = (density / unit.mprot / self.mu_weight) * unit.kb * self.temperature
        return (density, pressure)

    def UniversalProfile(self, r):
        pressure, temperature, density, R500 = universal_profile(r, self.M500_msol, H0=self.H0)
        #density = self.rho_j 
        # HACK 
        return (density, pressure)


    def pressure_function(self, r):
        '''
        get pressure at radius r 
        '''

        return (0.0)

    def equation_of_state(self, internal_energy, eos_type="Ideal"):
        ee = internal_energy / self.density

        if eos_type == "TM":
            # this is the Taub-Matthews equation of state
            # see eq 4 of Mignone 2007, https://arxiv.org/pdf/0704.1679.pdf
            pressure = (ee + 2) / (ee + 1) * self.density * ee / 3.0
            self.gamma = (pressure / internal_energy) + 1
        elif eos_type == "Ideal":
            self.gamma = 4./3.
            pressure = (self.gamma - 1.0) * internal_energy 
        #print (self.gamma)
        return (pressure)

    def get_v_advance(self):
        '''
        get the advance speed. Calls the density function and calculates density contrast
        '''
        if self.length > 0:
            rho = self.profile(self.length)[0]
        else:
            rho = self.rho0

        # get density contrast
        eta = self.rho_j / rho
        eta_star = self.gmm * self.gmm * eta 
        root_eta = np.sqrt(self.rho_j / rho) 

        if self.rel_advance:
            factor = np.sqrt(eta_star) / (np.sqrt(eta_star) + 1)
        else:
            factor = root_eta 

        hs_pressure = 0.5 * self.rho_j * unit.mprot * self.v_j * self.v_j
        if self.hs_pressure != 0.0:
            hs_pressure = 0.5 * self.rho_j * unit.mprot * self.v_j * self.v_j
            dp = hs_pressure - self.hs_pressure

            if np.fabs((dp) / (self.hs_pressure)) > 0.1:
                hs_pressure = self.hs_pressure + (np.sign(dp) * 0.1 * self.hs_pressure)

        self.hs_pressure = hs_pressure
        self.v_advance = factor * self.v_j / self.geometry_factor

    def get_v_perp(self):
        '''
        get sideways expansion speed
        '''
        r = np.sqrt((self.z**2) + (self.width**2))
        rho, P_a = self.profile(r)

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
        delta_P = self.pressure[select] - P_a[select]

        # check if pressure gradient is negative anywhere 
        # if (delta_P < 0).any():
        #     print ("delta_P is <0!", self.pressure_uniform, np.max(P_a))

        delta_P[(delta_P < 0)] = 0.0
        self.v_perp[select] = np.sqrt( delta_P / (rho[select] * unit.mprot)) * np.cos(phi)


        # ensure points beyond jet head cannot expand sideways 
        self.v_perp[~select] = 0.0
        #print ("Pressure:", self.v_perp, self.pressure, self.energy, self.volume, rho)

        if np.isnan(self.v_perp).any():
            print ("vperp exit: perpendicular velocity is nan.")
            print ("Time and length of jet:", self.time/unit.myr, self.length/unit.kpc)
            print ("Diag:", np.isnan(self.pressure).any(), np.isnan(P_a).any(), np.isnan(rho).any(), np.isnan(np.cos(phi)).any())

            print (self.pressure[select], rho[select], phi)
            return (-1)

        return (0) 

    def initial_width(self, z, L):
        #w = L / 2.0
        w = np.zeros_like(z)
        w[0] = L/2.0
        #w = L / np.pi * np.arccos(np.sqrt(z/L))
        #w[(z>L)] = 0.0

        return w

    def v_from_gamma(self, gmm):

        beta = np.sqrt(1-gmm**-2)
        if (1-gmm**-2 < 0):
            print ("error in gamma!", gmm, gmm**(-2.0)) 

        return (beta * unit.c)
    def gamma_from_v(self, v):
        beta = v/unit.c
        return (1-np.sqrt(1-beta**2))

    def jet_power(self, gmm, rho, area):
        v = self.v_from_gamma(gmm)
        Q = gmm * (gmm - 1.) * area * rho * unit.c * unit.c * v 
        return (Q)



    def set_power(self, time):
        self.gmm = self.gamma_interp_func(time)
        #self.gmm = self.gamma_interp_func(time)
        # IMPROVE CHECK ACTUAL DENSITY USING M200 profile??
        self.power = self.jet_power(self.gmm, self.rho_j * unit.mprot, self.area)
        #self.power = self.interp_func(time) * self.power_norm
        self.v_j = self.v_from_gamma(self.gmm)
        self.mdot = self.gmm * self.rho_j * self.v_j * self.area * unit.mprot
        self.power_available = self.f_work * self.power

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
        #if self.length > (2.0  * self.dz):
        self.set_power(self.time)
        self.get_v_advance()

        #print (self.v_advance/C)
        
        if self.set_aspect == False: 
            ireturn = self.get_v_perp()
            self.dt = self.dz / self.v_advance 
        else:
            ireturn = 0
            self.dt = self.dz / self.v_advance   

        if (ireturn < 0):
            return (ireturn)

        # evolve width 
        self.length += self.v_advance * self.dt
        delta_energy = self.power_available * self.dt 
        self.energy += delta_energy 
        self.mass += self.mdot * self.dt
        self.E_B += (1.0 - self.kappa) * self.zeta * self.epsilon * delta_energy

        # decide whether we need to artifically set the aspect ratio or not
        if self.set_aspect:
            #self.length = 0.0
            if (self.length == 0.0): 
                print ("ERROR, length is zero!", self.dz, self.dt)
            self.width = self.initial_width(self.z, self.length)
            self.set_aspect = False
            self.i = 1
            self.volume = 0.0
        else:
            self.width += self.v_perp * self.dt
            self.width = savgol_filter(self.width, 5,3)
            self.i += 1

        # find the point halfway up the lobe 
        # self.half_width = np.max(self.width) 
        # self.half_width = self.width[np.argmin(np.fabs(self.z-(self.length*0.5)))]

        # find where the lobe ends and select all elements inside it
        select_lobe = (self.z <= self.length)

        if np.sum(select_lobe) == 0: 
            self.half_width = 1.0
        else:
            # find the maximum width, divide by 2, then also multiply by two to get total width
            self.half_width = np.max(self.width[select_lobe]) 


        # get the volume by integration 
        v_old = self.volume
        self.volume = np.trapz(np.pi * (self.width[select_lobe]**2), x = self.z[select_lobe])
        dVdt = (self.volume - v_old) / self.dt
        self.density = self.mass / self.volume

        # work out magnetic field by finding B field internal energy 
        self.B = np.sqrt(self.E_B / self.volume * 8.0 * np.pi)

        # get 1/3 * (1/V) * dV/dt, which is adiabatic cooling rate / E
        # this will be used for both CRs and electrons 
        self.cool_adiabatic = 1/3 * (1.0 / self.volume) * dVdt

        self.internal = (self.energy / self.volume)
        self.pressure_uniform = self.equation_of_state(self.internal)
        self.pressure_array = self.set_lobe_pressure()

        # get magnetic field strength from equation ??
        #self.B 
        #self.B = np.sqrt(3.0 * self.pressure_uniform * self.zeta * 8.0 * np.pi / (1.0 + self.zeta + self.kappa))
        
        # advance time 
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
            print ("Error: Volume is NAN, exiting!")
            print (self.pressure_uniform, self.internal, self.energy, self.B)

            import sys
            sys.exit()

        return (0)




