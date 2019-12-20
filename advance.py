from simulation import JetClass, unit
from DELCgen import *
import numpy as np 
import matplotlib.pyplot as plt 

def get_lc(lognorm_params, PSD_params, tbin, Age, randomseed = 12):
    # Simulation params
    # let's do everything in units of kyr
    # run for 100 Myr (1e5 kyr) in bins of 0.1 Myr
    #lognorm_params = (1.5,0,np.exp(1.5))
    RedNoiseL,RandomSeed,aliasTbin = 100,12,100
    N = Age / tbin

    lc = Simulate_DE_Lightcurve(BendingPL, PSD_params,st.lognorm,lognorm_params,
                                    RedNoiseL=RedNoiseL,aliasTbin=aliasTbin,randomSeed=randomseed,LClength=Age, tbin=tbin)

    return (lc)


plt.clf()

# z = np.linspace(0,100.0,10000) * unit.kpc 
# def initial_width(z, L):
#     w = L / np.pi * np.arccos(np.sqrt(z/L))
#     w[(z>L)] = 0.0

#     return w

# plt.plot(initial_width(z, 50.0 * unit.kpc), z)   
# plt.show() 


# set up light curve. pl_index of 1 means red noise.
pl_index = 1

# bend the power law at 1 Myr (1e-3 kyr^-1) and steeply decay it beyond that with index 10. 
# A,v_bend,a_low,a_high,c = 1, 1e-3, pl_index, 10, 1
PSD_params = (1, 1e-3, pl_index, 10, 1)
lognorm_params = (3,0,np.exp(3))
tbin = 100    		# 100 kyr 
age = 10000


lightcurve = get_lc(lognorm_params, PSD_params, tbin, age)

ax1 = plt.subplot(511)
ax2 = plt.subplot(512)
ax3 = plt.subplot(513)
ax4 = plt.subplot(514)
ax5 = plt.subplot(515)

for i in range(1):

	print ("\nJet number ", i)

	lightcurve = get_lc(lognorm_params, PSD_params, tbin, age, randomseed=12)
	jet = JetClass(lightcurve)

	# initialise atmosphere with density, scale length and beta
	jet.init_atmosphere(1.0, 50.0, 0.5)
	jet.init_jet(power_norm = 3e42, rho_j = 1e-3)
	lengths = []
	widths = []
	v = []
	Bs = []
	t = 0
	i = 0
	times = []
	plot = True

	x = np.linspace(-40,40,800)
	y = np.linspace(0,100,1000)
	xx,yy = np.meshgrid(x,y)
	r = np.sqrt((xx**2) + (yy**2))
	rho = jet.density_function(r)

	# time = np.linspace(0,50.0*unit.myr,10000)
	# plt.plot(time / unit.myr, jet.interp_func(time))
	# plt.plot(jet.lc.time * unit.kyr / unit.myr, jet.lc.flux, alpha=0.5, ls="--")
	# plt.semilogy()
	# plt.show()
	iplot = 0
	print ("0--------100kpc Jet progress")
	counter = 10.0 * unit.kpc

	while (jet.length < 100.0 * unit.kpc):
		jet.UpdateSolution()
		times.append(jet.time)
		lengths.append(jet.length)
		halfway = np.argmin(np.fabs(jet.z - (jet.length/2.0))) 
		widths.append(jet.width[halfway])
		Bs.append(jet.B)
		v.append(jet.v_j)
		t += jet.dt

		if jet.length > counter:
			counter += 10.0 * unit.kpc 
			print ("#", end="", flush=True)

		if i % 50 == 0 and plot:
			select = (jet.z <= jet.length)
			phi = np.zeros_like(jet.z[select])
			grad_tan = np.gradient(jet.z[select], jet.width[select])
			grad_normal = -1.0 / grad_tan
			phi = np.arctan(grad_normal)
			#print (grad_normal, jet.length)
			#plt.plot(grad_normal, jet.z[select]/unit.kpc)

			plt.subplot(211)
			plt.plot(np.array(times) / unit.myr, np.array(v)  / unit.c)
			plt.subplot(212)
			plt.plot(jet.width/unit.kpc, jet.z/unit.kpc, c="k")
			plt.plot(-jet.width/unit.kpc, jet.z/unit.kpc, c="k")
			#plt.xlim(-jet.length/unit.kpc,jet.length/unit.kpc)
			#plt.pcolormesh(xx,yy,np.log10(rho))
			plt.xlim(-50,50)
			plt.ylim(0,100)
			#plt.ylim(0,jet.length/unit.kpc * 2.0)
			plt.savefig("dim{:03d}.png".format(iplot))
			iplot +=1 
			plt.clf()


		i += 1


	lengths = np.array(lengths)
	widths = np.array(widths)
	times = np.array(times)
	v = np.array(v)
	Bs = np.array(Bs)

	ax1.plot(times / unit.myr, lengths / unit.kpc)	
	ax2.plot(times / unit.myr, widths / unit.kpc)

	ax3.plot(times / unit.myr, widths/lengths)	
	ax4.plot(times / unit.myr, v/unit.c)	

	ax5.plot(times / unit.myr, Bs)	

ax1.loglog()
ax2.loglog()
ax3.loglog()
ax4.loglog()
ax5.loglog()
plt.show()
# def __init__(self, lightcurve, dt, nz=1000, zmax = 200):