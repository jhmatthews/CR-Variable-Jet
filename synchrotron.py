from scipy import integrate,special
import numpy as np 
from constants import *

def F(x):
    """ This is F(x) defined in equation 6.31c in R&L.
        
        F(x) = x*int(K_5/3(x)dx) where the integral goes from x to infinity.

        for some reason, special.kv(5/3,1e10) is NaN, not 0 ???
        for now, just clip the function above 1e5 to be 0. 
        
        This function can be evaluated in mathematica using the following command
            F[x_] := N[x*Integrate[BesselK[5/3, y], {y, x, Infinity}]]
        From mathematica, we find that
                  x         F(x)
              ----- ------------
                0.1     0.818186
                  1     0.651423
                 10  0.000192238
                100            0
        Comparing our function to the Mathematica integral, we find
            >>> np.allclose(Synchrotron.F([0.1,1,10,100]), [0.818186, 0.651423, 0.000192238,0], rtol=1e-4, atol=1e-4)
            True
        Note, this function is _F so that the docstring will get executed.
    """
    if x > 1e5: 
        return 0
    return x*integrate.quad(lambda j: special.kv(5./3,j),x,np.inf, limit=200)[0]

def psynch(energies, nu, Bfield):
    sinalpha = 2.0 / 3.0
    gammas = energies * EV2ERGS / MELEC / C / C
    nu_c = 3.0 * gammas * gammas * E * Bfield * sinalpha / MELEC / C / 2.0 
    term1 = np.sqrt(3)
    term2 = E * E * E * Bfield * sinalpha / MELEC / C / C 

    
    #def array_map(x):
    x = nu/nu_c
    term3 = np.array(list(map(F, x)))
    #term3 = np.array([F(nu/nuc) for nuc in nu_c])
    return term1 * term2 * term3 

def Ptot(nus, energies, ncr, Bfield):
    pnu = np.zeros_like(nus)
    for inu, nu in enumerate(nus):
        #print (inu)
        power = psynch(energies, nu, Bfield)
        pnu[inu] = -np.trapz(energies, ncr * power)

    return pnu

# import time 
# nu = np.logspace(7,12,100)
# energies =np.logspace(9,14,100)
# ncr = 1e40 * ((energies/1e9) **-2)

# import matplotlib.pyplot as plt 

# t1 = time.time()
# power = Ptot(nu, energies, ncr, 1e-6)
# print (time.time() - t1)

# print (power)
# plt.plot(nu,power)
# plt.loglog()
# plt.show()
