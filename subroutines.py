import matplotlib.pyplot as plt 
from scipy.interpolate import interp1d
import numpy as np 
import subprocess
from constants import *

def init_fig(fname="movie", fmt="mp4", fps=6):
    plt.close("all")
    fig = plt.figure(frameon=False, figsize=(8,6), dpi=300)
    canvas_width, canvas_height = fig.canvas.get_width_height()

    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(223)
    ax3 = fig.add_subplot(224)

    # Open an ffmpeg process
    if fmt == "mp4":
        outf = '{}.mp4'.format(fname)
        cmdstring = ('ffmpeg', 
            '-y', '-r', '{:d}'.format(fps), # overwrite, fps
            '-s', '{:d}x{:d}'.format(canvas_width, canvas_height), # size of image string
            '-pix_fmt', 'argb', # format
            '-f', 'rawvideo',  '-i', '-', # tell ffmpeg to expect raw video from the pipe
            '-vcodec', 'mpeg4', outf) # output encoding
    elif fmt == "gif":
        outf = '{}.gif'.format(fname)
        cmdstring = ('ffmpeg', 
            '-y', '-r', '{:d}'.format(fps), # overwrite, fps
            '-s', '{:d}x{:d}'.format(canvas_width, canvas_height), # size of image string
            '-pix_fmt', 'argb', # format
            '-f', 'rawvideo',  '-i', '-', # tell ffmpeg to expect raw video from the pipe
            '-vcodec', 'gif', outf) # output encoding
        
    p = subprocess.Popen(cmdstring, stdin=subprocess.PIPE)
    
    return (p, fig, ax1, ax2, ax3)

def init_lc_plot(t, f):
    plt.clf()
    fig = plt.figure(figsize=(8,6))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(223)
    ax3 = fig.add_subplot(224)
    ax1.plot(t, f)
    ax1.set_yscale("log")
    ax1.set_xlabel("t (Myr)", fontsize=16)
    ax1.set_ylabel("$Q_j$ (erg/s)", fontsize=16)
    return (fig, ax2, ax3)

def plot_spectra(fig, ax2, ax3, energy, ncr, escaping, j, xlims = (1e16,1e21)):
    to_plot = energy*energy*ncr
    ax2.plot(energy, to_plot)
    ax2.loglog()
    ax2.set_xlim(xlims)
    my_max = 2.0*np.max(to_plot[(energy<xlims[1]) * (energy>xlims[0])])
    #if j > 0:
    #    if my_max > ax2.get_ylim()[1]:
    #        ax2.set_ylim(my_max/1e5,my_max)
   # else:
    #    ax2.set_ylim(my_max/1e5,my_max)
    ax2.set_ylabel("$E^2 n(E)$", fontsize=16)
    ax2.set_xlabel("$E$", fontsize=16)
    #ax2.set_ylim(1e70,1e75)

    to_plot = energy*energy*escaping
    ax3.plot(energy, to_plot)
    ax3.set_xlim(xlims)
    ax3.set_xlabel("$E$", fontsize=16)
    my_max = 2.0*np.max(to_plot[(energy<xlims[1]) * (energy>xlims[0])])

    if j > 0:
        if my_max > ax3.get_ylim()[1]:
            ax3.set_ylim(my_max/1e5,my_max)
    else:
        ax3.set_ylim(my_max/1e5,my_max)
    #ax3.set_ylabel("$E^2 n(E)$", fontsize=16)
    #ax3.set_ylim(1e67,1e72)
    ax3.loglog()


def plot_spectra_morph(fig, ax2, ax3, energy, ncr, dimensions, j):
    to_plot = energy*energy*ncr
    select = (ncr > 0)
    #print (ncr)
    ax2.plot(energy, to_plot)
    ax2.loglog()
    #ax2.set_xlim(xlims)
    # my_max = 2.0*np.max(to_plot[(energy<xlims[1]) * (energy>xlims[0])])
    # if j > 0:
    #     if my_max > ax2.get_ylim()[1]:
    #         ax2.set_ylim(my_max/1e5,my_max)
    # else:
    #     ax2.set_ylim(my_max/1e5,my_max)
    ax2.set_ylabel("$E^2 n(E)$", fontsize=16)
    ax2.set_xlabel("$E$", fontsize=16)
    #ax2.set_ylim(1e70,1e75)

    #to_plot = 
    if j == 0:
        ax3.plot(dimensions[1]/1000.0/PARSEC, dimensions[0]/1000.0/PARSEC, c="k")
        ax3.plot(-dimensions[1]/1000.0/PARSEC, dimensions[0]/1000.0/PARSEC, c="k")

        ax3.set_xlim(-100,100)
        ax3.set_ylim(0,200)
        ax3.set_xlabel("$x$ (kpc)", fontsize=16)
        ax3.set_ylabel("$z$ (kpc)", fontsize=16)
    # my_max = 2.0*np.max(to_plot[(energy<xlims[1]) * (energy>xlims[0])])

    # if j > 0:
    #     if my_max > ax3.get_ylim()[1]:
    #         ax3.set_ylim(my_max/1e5,my_max)
    # else:
    #     ax3.set_ylim(my_max/1e5,my_max)
    #ax3.set_ylabel("$E^2 n(E)$", fontsize=16)
    #ax3.set_ylim(1e67,1e72)
    #ax3.loglog()

def plot_lc(ax1, time, flux):
    ax1.plot(time, flux)
    ax1.set_yscale("log")
    ax1.set_xlabel("t (Myr)", fontsize=16)
    ax1.set_ylabel("$v_j/c$", fontsize=16)


class Losses:
    def __init__(self, elem_name):
        self.energies, self.total, self.pp, self.pd, self.ep = np.loadtxt("tau_{}.dat".format(elem_name), unpack=True)

    def interpol(self, energies):
        interp_func = interp1d(self.energies, self.total)
        self.total_interpol = interp_func(energies)
