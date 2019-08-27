import matplotlib.pyplot as plt 
from scipy.interpolate import interp1d
import numpy as np 
def init_lc_plot(t, f):
    plt.clf()
    fig = plt.figure(figsize=(8,6))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(223)
    ax3 = fig.add_subplot(224)
    ax1.plot(t, f)
    ax1.set_yscale("log")
    ax1.set_xlabel("t (Myr)", fontsize=16)
    ax1.set_ylabel("$Q_j$ (erg/s)", fontsize=16)
    return (fig, ax2, ax3)

def plot_spectra(fig, ax2, ax3, energy, ncr, escaping):
    xlims = (1e16,1e21)

    to_plot = energy*energy*ncr
    ax2.plot(energy, to_plot)
    ax2.loglog()
    ax2.set_xlim(xlims)
    my_max = 2.0*np.max(to_plot[(energy<xlims[1]) * (energy>xlims[0])])
    if my_max > ax2.get_ylim[1]:
        ax2.set_ylim(my_max/1e5,my_max)
    ax2.set_ylabel("$E^2 n(E)$", fontsize=16)
    ax2.set_xlabel("$E$", fontsize=16)
    #ax2.set_ylim(1e70,1e75)

    to_plot = energy*energy*escaping
    ax3.plot(energy, to_plot)
    ax3.set_xlim(xlims)
    ax3.set_xlabel("$E$", fontsize=16)
    my_max = 2.0*np.max(to_plot[(energy<xlims[1]) * (energy>xlims[0])])

    if my_max > ax3.get_ylim[1]:
        ax3.set_ylim(my_max/1e5,my_max)
    #ax3.set_ylabel("$E^2 n(E)$", fontsize=16)
    #ax3.set_ylim(1e67,1e72)
    ax3.loglog()

class Losses:
    def __init__(self, elem_name):
        self.energies, self.total, self.pp, self.pd, self.ep = np.loadtxt("tau_{}.dat".format(elem_name), unpack=True)

    def interpol(self, energies):
        interp_func = interp1d(self.energies, self.total)
        self.total_interpol = interp_func(energies)
