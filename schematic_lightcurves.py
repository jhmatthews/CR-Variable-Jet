from DELCgen import *
import scipy.stats as st
import matplotlib.pyplot as plt 
from constants import *

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
pl_index = 1
A,v_bend,a_low,a_high,c = 1, 1e-3, pl_index, 5, 1
# Probability density function params
kappa,theta,lnmu,lnsig,weight = 5.67, 5.96, 2.14, 0.31,0.82
# Simulation params
# let's do everything in units of kyr
# run for 100 Myr (1e5 kyr) in bins of 100 kyr
lognorm_params = (1.5,0,np.exp(1.5))
RedNoiseL,RandomSeed,aliasTbin, tbin, Age = 100,12,10,10,10000
N = Age / tbin

lc_pink = Simulate_DE_Lightcurve(BendingPL, (A,v_bend,a_low,a_high,c),st.lognorm,lognorm_params,
                                RedNoiseL=RedNoiseL,aliasTbin=aliasTbin,randomSeed=RandomSeed,LClength=Age, tbin=tbin)


#lc_pink.Plot_Periodogram()
#lc_pink.Save_Lightcurve('lightcurve.dat')
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
flux_scale = 1e43
flux = lc_pink.flux * flux_scale 
time = lc_pink.time/1000.0
time_long = np.arange(0,500,tbin/1000.0)
ton = 10.0
toff = 30.0
tpink = 5e59 / np.mean(flux) 
print (tpink / YR / 1e6)

lc_color="k"
alpha = 0.7
lc_steady = np.ones_like(time_long) * flux_scale * 5.0
lc_tophat = np.ones_like(flux) * flux_scale * 1.0
lc_tophat[(time>60.0) * (time<70.0)] = 1e45

plt.rcParams["text.usetex"] = "True"
plt.clf()
plt.figure(figsize=(10,6))
plt.ylabel("$Q_j~(\mathrm{erg~s}^{-1})$", fontsize=18)
plt.xlabel("$t-t_0~(\mathrm{Myr})$", fontsize=18)
plt.plot(time[(time<tpink)]-100, flux[(time<tpink)], c=lc_color, lw=1.5, alpha=alpha, label="$\mathrm{Variable~(Pink~noise)}$")
plt.plot(time_long-100, lc_steady, lw=4, alpha=1, label="$\mathrm{Buoyant~(Steady, low)}$", ls="--", c= "C1")
plt.plot(time-100, lc_tophat, lw=4, alpha=1, label="$\mathrm{Outburst~(Tophat)}$", c= "C0")
plt.legend(loc=4)
plt.semilogy()
plt.xlim(-100,0)
plt.ylim(1e42,1e47)
plt.subplots_adjust(right=0.98, left=0.1, top=0.98)

ax = plt.gca()
axins = inset_axes(ax, width="50%", height="70%",bbox_to_anchor=(.08, .65, .4, .4),bbox_transform=ax.transAxes, loc=3)
freq = np.logspace(-6,0,num=10000)
axins.plot(freq * 1000.0, BendingPL(freq, 1000.0,v_bend,a_low,a_high,c), c=lc_color, alpha=alpha, lw=3)
axins.set_xlabel(r"$\nu~(\mathrm{Myr}^{-1})$")
axins.set_ylabel(r"$\mathrm{Power~(Arb.)}$")
plt.loglog()
plt.title("$\mathrm{PSD}$")

axins = inset_axes(ax, width="50%", height="70%",bbox_to_anchor=(.65, .65, .4, .4),bbox_transform=ax.transAxes, loc=3)
freq = np.logspace(-6,0,num=10000)
bins = np.arange(42,48,0.1)
axins.hist(np.log10(flux), bins=bins, color=lc_color, alpha=alpha, density=True)
axins.set_xlim(42,45)
axins.set_xlabel(r"$Q_j$")
#axins.set_ylabel(r"$\mathrm{Power~(Arb.)}$")
plt.title("$\mathrm{PDF}$")
plt.savefig("lc_shema.png", dpi=300)
