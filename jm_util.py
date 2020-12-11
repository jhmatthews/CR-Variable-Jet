import matplotlib.pyplot as plt 
import numpy as np 
import pickle

def load_one_npz(fname):
	f = np.load(fname)
	x = f[f.files[0]]
	f.close()
	return (x)

def load_pickle(fname):
	with open(fname, "rb") as pickle_file:
		j = pickle.load(pickle_file)
	return (j)

def set_plot_defaults():
	plt.rcParams["text.usetex"] = "True"
	plt.rcParams['xtick.labelsize']=14
	plt.rcParams['ytick.labelsize']=14
	plt.rcParams['font.serif']=['cm']
	plt.rcParams['font.family']='serif'	
	plt.rcParams["text.usetex"] = "True"
	plt.rcParams["lines.linewidth"] = 3
	plt.rcParams["axes.linewidth"] = 2
	plt.rcParams["xtick.major.width"] = 1.5
	plt.rcParams["ytick.major.width"] = 1.5

def set_cr_defaults():
	## FIGURE
	plt.rcParams["text.usetex"] = "True"
	plt.rcParams['figure.figsize']=(8, 8) # MNRAS columnwidth

	## FONT
	plt.rcParams['font.serif']=['cm']
	plt.rcParams['font.family']='serif'	
	# plt.rcParams['font.serif']=['cm']

	# plt.rcParams['mathtext.fontset'] = 'cm'
	# plt.rcParams['mathtext.rm']='serif'
	plt.rcParams['text.latex.preamble']=r'\usepackage{amsmath}'

	#plt.rcParams['font.size']=18
	plt.rcParams['xtick.labelsize']=15
	plt.rcParams['ytick.labelsize']=15
	# plt.rcParams['legend.fontsize']=18
	# plt.rcParams['axes.titlesize']=18
	# plt.rcParams['axes.labelsize']=18
	plt.rcParams["lines.linewidth"] = 2
	plt.rcParams['axes.linewidth']=2
	## TICKS
	#plt.rcParams['xtick.top']='True'
	plt.rcParams['xtick.bottom']='True'
	plt.rcParams['xtick.minor.visible']='True'
	plt.rcParams['xtick.direction']='out'
	# plt.rcParams['ytick.left']='True'
	plt.rcParams['ytick.right']='True'
	plt.rcParams['ytick.minor.visible']='True'
	plt.rcParams['ytick.direction']='out'
	plt.rcParams['xtick.major.width']=1.5
	plt.rcParams['xtick.minor.width']=1
	plt.rcParams['xtick.major.size']=4
	plt.rcParams['xtick.minor.size']=3
	plt.rcParams['ytick.major.width']=1.5
	plt.rcParams['ytick.minor.width']=1
	plt.rcParams['ytick.major.size']=4
	plt.rcParams['ytick.minor.size']=3

	## LEGEND
	plt.rcParams['legend.frameon']='False'


def set_mod_defaults():
	## FIGURE
	plt.rcParams["text.usetex"] = "True"
	#plt.rcParams['figure.figsize']=(8, 8) # MNRAS columnwidth

	## FONT
	plt.rcParams['font.serif']=['cm']
	plt.rcParams['font.family']='serif'	
	# plt.rcParams['font.serif']=['cm']

	# plt.rcParams['mathtext.fontset'] = 'cm'
	# plt.rcParams['mathtext.rm']='serif'
	plt.rcParams['text.latex.preamble']=r'\usepackage{amsmath}'

	plt.rcParams['font.size']=18
	plt.rcParams['xtick.labelsize']=15
	plt.rcParams['ytick.labelsize']=15
	plt.rcParams['legend.fontsize']=14
	plt.rcParams['axes.titlesize']=16
	plt.rcParams['axes.labelsize']=16
	plt.rcParams['axes.linewidth']=2
	plt.rcParams["lines.linewidth"] = 2.2
	## TICKS
	plt.rcParams['xtick.top']='True'
	plt.rcParams['xtick.bottom']='True'
	plt.rcParams['xtick.minor.visible']='True'
	plt.rcParams['xtick.direction']='out'
	plt.rcParams['ytick.left']='True'
	plt.rcParams['ytick.right']='True'
	plt.rcParams['ytick.minor.visible']='True'
	plt.rcParams['ytick.direction']='out'
	plt.rcParams['xtick.major.width']=1.5
	plt.rcParams['xtick.minor.width']=1
	plt.rcParams['xtick.major.size']=4
	plt.rcParams['xtick.minor.size']=3
	plt.rcParams['ytick.major.width']=1.5
	plt.rcParams['ytick.minor.width']=1
	plt.rcParams['ytick.major.size']=4
	plt.rcParams['ytick.minor.size']=3

	## LEGEND
	#plt.rcParams['legend.frameon']='False'

def set_amy_defaults():
	## FIGURE
	plt.rcParams["text.usetex"] = "True"
	plt.rcParams['figure.figsize']=(8, 8) # MNRAS columnwidth

	## FONT
	plt.rcParams['font.serif']=['cm']
	plt.rcParams['font.family']='serif'	
	# plt.rcParams['font.serif']=['cm']

	# plt.rcParams['mathtext.fontset'] = 'cm'
	# plt.rcParams['mathtext.rm']='serif'
	plt.rcParams['text.latex.preamble']=r'\usepackage{amsmath}'

	plt.rcParams['font.size']=18
	plt.rcParams['xtick.labelsize']=15
	plt.rcParams['ytick.labelsize']=15
	plt.rcParams['legend.fontsize']=18
	plt.rcParams['axes.titlesize']=18
	plt.rcParams['axes.labelsize']=18
	plt.rcParams['axes.linewidth']=2
	## TICKS
	plt.rcParams['xtick.top']='True'
	plt.rcParams['xtick.bottom']='True'
	plt.rcParams['xtick.minor.visible']='True'
	plt.rcParams['xtick.direction']='out'
	plt.rcParams['ytick.left']='True'
	plt.rcParams['ytick.right']='True'
	plt.rcParams['ytick.minor.visible']='True'
	plt.rcParams['ytick.direction']='out'
	plt.rcParams['xtick.major.width']=1.5
	plt.rcParams['xtick.minor.width']=1
	plt.rcParams['xtick.major.size']=4
	plt.rcParams['xtick.minor.size']=3
	plt.rcParams['ytick.major.width']=1.5
	plt.rcParams['ytick.minor.width']=1
	plt.rcParams['ytick.major.size']=4
	plt.rcParams['ytick.minor.size']=3

	## LEGEND
	plt.rcParams['legend.frameon']='False'

def set_color_cycler(style="default"):
	#if type(style) == str:
	import matplotlib.style
	if style == "default":
		matplotlib.style.use(style)
	else:
		plt.rcParams['axes.prop_cycle'] = matplotlib.style.library[style]['axes.prop_cycle'] 


