# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from jm_util import *
from constants import *
from simulation import unit
import matplotlib, sys
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


set_plot_defaults()

#suffix = sys.argv[1]
suffix = "ref_p1e+45_geo4_etah0.3"
#suffix = "beta{:.1f}q{:.1f}sig{:.1f}seed{:d}".format(BETA, np.log10(flux_scale),  SIGMA, seed)
fname = "array_saves/dim_{}.npz".format(suffix)
fname_jet = "array_saves/jetstore_{}.pkl".format(suffix)
print (fname)
dimensions = load_one_npz(fname)
jet = load_pickle(fname_jet)

DELTA = 10
print ("FRAMES:", len(dimensions[0,:,0])/DELTA)
NPTS = len(dimensions[0,:,0])
print (NPTS)

#N = 200
tmax = np.max(jet.time/unit.myr)
print (tmax)
imax = (tmax  / 0.1 / 5)
print (imax)
N_use = 180


cmap = cm.get_cmap('viridis')
colors = cmap(np.linspace(0,1,num=N_use))
#colors2 = 

plt.figure(figsize=(10,5))
for N in np.arange(0,N_use,DELTA):

	print (N, N * 0.5)
	x = dimensions[1,N,:]/1000.0/PARSEC
	z = dimensions[0,N,:]/1000.0/PARSEC
	x[:20] = x[20]
	select = (x>0.01)
	xx = x[select]
	zz = z[select]

	#print (xx, zz)

	marker= "o"
	marker=None
	x = zz
	z = xx
	plt.plot(x, z, marker=marker, c=colors[N])
	plt.plot(-x, z, marker=marker, c=colors[N])
	plt.plot(x, -z, marker=marker, c=colors[N])
	plt.plot(-x, -z, marker=marker, c=colors[N])

#plt.show()
print (x)
width = np.max(x)
length = np.max(z)


LIM = 30
ratio = 774.0/434.0
XLIM = 300 
plt.xlim(-XLIM, XLIM)
LIM = XLIM / ratio
plt.ylim(-LIM, LIM)
plt.ylabel("$x$ (kpc)", fontsize=16)
plt.xlabel("$z$ (kpc)", fontsize=16)


t = jet.time/unit.myr
norm = matplotlib.colors.Normalize(vmin=t[0], vmax=t[N*5])
mappable1 = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap )

ax1 = plt.gca()
axins1 = inset_axes(ax1,width="50%",  height="15%", loc='lower left', bbox_to_anchor=(.15, .15, .4, .4), bbox_transform=ax1.transAxes)

cbar = plt.colorbar(mappable=mappable1, cax = axins1, shrink=1, orientation="horizontal")
cbar.set_label("$t$ (Myr)", fontsize=16)

plt.subplots_adjust(left=0.08, right=0.98, top=0.98)

x = np.linspace(-XLIM, XLIM, 1000)
y = np.linspace(-LIM, LIM, 1000)
xx,yy = np.meshgrid(x,y)
r = np.sqrt(xx**2 + yy**2)
_, P = jet.jet.profile(r * unit.kpc)
#plt.scatter(xx,yy)
print (np.nanmax(P))
mappable2 = ax1.pcolormesh(xx,yy,np.log10(P/np.nanmax(P)), cmap="binary", vmax=0, vmin=-1.5)
#plt.colorbar()
axins1 = inset_axes(ax1,width="50%",  height="15%", loc='lower right', bbox_to_anchor=(.45, .15, .4, .4), bbox_transform=ax1.transAxes)
cbar = plt.colorbar(mappable = mappable2, cax = axins1, shrink=1, orientation="horizontal")
cbar.set_label(r"$\log(P/P_0)$", fontsize=16)


# a = jet.length[N]/unit.kpc
# b = jet.width[N]/unit.kpc
b = length
a = width
print (a,b)
x = np.linspace(-a,a,1000)
y = np.sqrt((1 - (x**2/a**2)) * b**2)
print (a,b,y)
ax1.plot(x,y, ls="--", c="C3")
ax1.plot(x,-y, ls="--", c="C3")
ax1.text(-250,60,r"$\frac{{z^2}}{{L^2}} + \frac{{x^2}}{{R^2}}=1$ for $t={:d}$Myr".format(int(t[N*5])), color="C3", fontsize=14)

plt.savefig("shell.png".format(suffix), dpi=300)

plt.figure()

# let's get various Mach numbers
r = np.linspace(0,300,3000)
rho, P = jet.jet.profile(r * unit.kpc)
rhotop, Ptop = jet.jet.profile(jet.length)
rhobase, Pbase = jet.jet.profile(jet.width)

rhotop *= unit.mprot
rhobase *= unit.mprot

cstop = np.sqrt(5./3. * Ptop / rhotop)
csbase = np.sqrt(5./3. * Pbase / rhobase)

#cs_lobe = np.sqrt(5./3. * P / rho)
M_head = jet.v_advance / cstop 
M_lobe_tip = jet.pressure / Ptop
M_base = jet.pressure / Pbase

# zcrit = np.zeros_like(t)
# N = len(t)
plt.plot(t, M_head)
plt.plot(t, M_lobe_tip)
plt.plot(t, M_base)
plt.ylim(0.1,100)
plt.xlim(0,65)
plt.semilogy()


# cs = 
# M_head = 

# for i in range(0,N,10):
# 	x = dimensions[1,i,:]/1000.0/PARSEC
# 	z = dimensions[0,i,:]/1000.0/PARSEC
# 	r = np.sqrt(x**2 + z**2)
# 	_, P = jet.jet.profile(r * unit.kpc)
# 	#plt.plot(r, P, c=colors[i])
# 	#plt.plot(r, np.ones_like(r)*jet.pressure[i], c=colors[i])


# 	zcrit[i] = z[np.argmin(np.fabs(P-jet.pressure[i]))]

#plt.semilogy()
#plt.figure()
# plt.plot(r, P)#
#plt.plot(t, zcrit)
plt.plot(t, jet.pressure)
plt.savefig("mach.png")


