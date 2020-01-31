from DELCgen import *
import scipy.stats as st
import matplotlib.pyplot as plt 
from constants import *
import sys
from simulation import get_lc, unit
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#plt.xkcd()

# File Route
route = "./"
datfile = "NGC4051.dat"

class Losses:
	def __init__(self, elem_name):
		self.energies, self.total, self.pp, self.pd, self.ep = np.loadtxt("tau_{}.dat".format(elem_name), unpack=True)

	def interpol(self, energies):
		from scipy.interpolate import interp1d
		interp_func = interp1d(self.energies, self.total)
		self.total_interpol = interp_func(energies)

# Bending power law params
MYR = 1e3
pl_index = int(sys.argv[1])
v_bend = 1.0 * MYR
a_low = pl_index
a_high = 20
c = 0
PSD_params = (1, 1.0 * MYR, a_low, a_high, c)
# Simulation params
# let's do everything in units of kyr
# run for 100 Myr (1e5 kyr) in bins of 100 kyr
tbin = 0.01 * MYR   # 100 kyr 
Age = 100.0 * MYR
Length = int(Age / tbin)
lognorm_params = (1.5,0,np.exp(np.log(1)))
N = Age 


lc = get_lc(lognorm_params, PSD_params, tbin, Length)
# RedNoiseL,RandomSeed,aliasTbin = 100,15,10
#lc = Simulate_TK_Lightcurve(BendingPL, PSD_params, RedNoiseL=RedNoiseL, aliasTbin=aliasTbin,randomSeed=RandomSeed,length=Length,tbin=tbin)


flux_scale = 1e43
flux = lc.flux * flux_scale 

lc.time = np.linspace(0,np.max(lc.time),len(lc.flux))
print (lc.flux.shape, lc.time.shape)

print (np.median(lc.flux), np.median(flux), np.mean(flux))


lc_color="k"
alpha = 0.8

plt.rcParams["text.usetex"] = "True"
plt.clf()
plt.figure(figsize=(10,6))
plt.ylabel("$Q_j~(\mathrm{erg~s}^{-1})$", fontsize=18)
plt.xlabel("$t-t_0~(\mathrm{Myr})$", fontsize=18)
plt.plot(lc.time * unit.kyr / unit.myr, flux, c=lc_color, lw=1, alpha=alpha, label="$\mathrm{Variable~(Pink~noise)}$")
# plt.plot(lc.time * unit.kyr / unit.myr, np.flip(flux), c=lc_color, lw=1, alpha=alpha, label="$\mathrm{Variable~(Pink~noise)}$")

#plt.legend(loc=4)
plt.semilogy()
plt.xlim(0,100)
plt.ylim(1e41,1e47)
plt.subplots_adjust(right=0.98, left=0.1, top=0.98)

ax = plt.gca()
axins = inset_axes(ax, width="50%", height="70%",bbox_to_anchor=(.08, .65, .4, .4),bbox_transform=ax.transAxes, loc=3)
freq = np.logspace(0,4,num=10000)
axins.plot(freq / MYR, BendingPL(freq, 1.0 * MYR, v_bend, a_low, a_high, c), c=lc_color, alpha=alpha, lw=3)
#axins.set_xlim(1e-3, 1000)
plt.xlim(1e-3,5)
plt.ylim(1e-7,1e3)
axins.set_xlabel(r"$\nu~(\mathrm{Myr}^{-1})$")
axins.set_ylabel(r"$\mathrm{Power~(Arb.)}$")
plt.loglog()
plt.title("$\mathrm{PSD}$")

axins = inset_axes(ax, width="50%", height="70%",bbox_to_anchor=(.65, .65, .4, .4),bbox_transform=ax.transAxes, loc=3)
bins = np.arange(40,46,0.1)
# bins = np.logspace(42,48,1000)
axins.hist(np.log10(flux[:len(flux)//2]), bins=bins, color=lc_color, alpha=alpha, density=True)
#axins.set_xlim(42#,48)
#axins.set_xlabel(r"$Q_j$")
# axins.set_ylabel(r"$\mathrm{~(Arb.)}$"))
plt.title(r"$\mathrm{PDF}, \log(Q_j)$")
plt.savefig("pdf_psd_lc_schema.png", dpi=300)
