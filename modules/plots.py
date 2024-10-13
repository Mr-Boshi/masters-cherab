import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D

from cherab.core.math import sample3d 
from cherab.core.atomic import hydrogen, carbon

from raysect.optical import Point3D

def plot_beam(beam_length,beam_full,beam_half = None, beam_third = None, norm = False, figsize=(15,12), axis = 'z', scale = 'log'):
	fig, ax = plt.subplots(2,1, figsize=figsize, constrained_layout=True)

	x_start, x_end, y_start, y_end, z_start = 0, 0, 0, 0, 0
	x_num, y_num  = 1, 1
	z_end   = beam_length
	z_num   = 20

	ax_min = z_start
	ax_max = z_end

	if axis == 'x':
		x_start         = -0.3
		x_end           = 0.3
		x_num           = 200
		z_start, z_end  = 0, 0
		z_num           = 1
		ax_min          = x_start
		ax_max          = x_end

	if axis == 'y':
		y_start         = -0.3
		y_end           = 0.3
		y_num           = 200
		z_start, z_end  = 0, 0
		z_num           = 1
		ax_min          = y_start
		ax_max          = y_end

	# Sampling main component  
	x1, y1, z1, beam_density1D_full  = sample3d(beam_full.density, (x_start, x_end, x_num), (y_start, y_end, y_num), (z_start, z_end, z_num))
	x2, _, z2, beam_density2D_full = sample3d(beam_full.density, (-0.3, 0.3, 100), (0, 0, 1), (0, beam_length, 200))

	# Preallocation
	beam_density2D_half  = np.zeros(beam_density2D_full.shape)
	beam_density2D_third = np.zeros(beam_density2D_full.shape)

	# Plotting main component density
	if norm == True:
		n_coef_1 = np.max(beam_density1D_full)
	else:
		n_coef_1  = 1

	axis_grid = z1
	if axis == 'x':
		axis_grid = x1
	if axis == 'y':
		axis_grid = y1
	
	if scale == 'log':
		ax[1].semilogy(axis_grid,np.squeeze(np.squeeze(beam_density1D_full / n_coef_1)),label='60 кэВ')
	elif scale == 'linear':
		ax[1].plot(axis_grid,np.squeeze(np.squeeze(beam_density1D_full / n_coef_1)),label='E0')
	
	# Sampling E0/2
	if beam_half != None:
		x1, y1, z1, beam_density1D_half  = sample3d(beam_half.density, (x_start, x_end, x_num), (y_start, y_end, y_num), (z_start, z_end, z_num))
		x2, _, z2, beam_density2D_half = sample3d(beam_half.density, (-0.3, 0.3, 100), (0, 0, 1), (0, beam_length, 200))
		if norm == True:
			n_coef_2 = np.max(beam_density1D_half)
		else:
			n_coef_2  = 1
	
		if scale == 'log':
			ax[1].semilogy(axis_grid,np.squeeze(np.squeeze(beam_density1D_half / n_coef_2)),label='80 кэВ')
		elif scale == 'linear':
			ax[1].plot(axis_grid,np.squeeze(np.squeeze(beam_density1D_half / n_coef_2)),label='E0/2')


	# Sampling E0/3	
	if beam_third != None:
		x1, y1, z1, beam_density1D_third = sample3d(beam_third.density,(x_start, x_end, x_num), (y_start, y_end, y_num), (z_start, z_end, z_num))
		x2, _, z2, beam_density2D_third = sample3d(beam_third.density, (-0.3, 0.3, 100), (0, 0, 1), (0, beam_length, 200))
		if norm == True:
			n_coef_3 = np.max(beam_density1D_third)
		else:
			n_coef_3  = 1

		if scale == 'log':
			ax[1].semilogy(axis_grid,np.squeeze(np.squeeze(beam_density1D_third / n_coef_3)),label='100 кэВ')
		elif scale == 'linear':	
			ax[1].plot(axis_grid,np.squeeze(np.squeeze(beam_density1D_third / n_coef_3)),label='E0/3')

	ax[1].set_xlabel('Расстояние от фланца инжектора, м', fontsize=16)
	if norm == False:
		ax[1].set_ylabel(r'Плотность пучка, м$^{-3}$', fontsize=16)
	else:
		ax[1].set_ylabel('Доля начальной плотности пучка', fontsize=16)
		if scale == 'linear':
			ax[1].yaxis.set_major_locator(MultipleLocator(0.1))

	ax[1].set_xlim(ax_min,ax_max)
	ax[1].tick_params(axis='x', labelsize=14)
	ax[1].tick_params(axis='y', labelsize=14)

	if axis == 'z':
		ax[1].xaxis.set_major_locator(MultipleLocator(0.1))
	else:
		ax[1].xaxis.set_major_locator(MultipleLocator(0.01))

	beam_density2D = beam_density2D_full + beam_density2D_half + beam_density2D_third
	cs = ax[0].contourf(z2,x2,(np.squeeze(beam_density2D)),35,cmap="inferno")
	cbar = fig.colorbar(cs,ax=ax[0])
	cbar.set_label('beam full density, m-3')

	ax[0].set_xlabel('z of beam, m')
	ax[0].set_ylabel('x of beam, m')
	ax[0].set_xlim(0,beam_length)
	ax[0].xaxis.set_major_locator(MultipleLocator(0.1))
	ax[0].grid(visible=True, which='both', axis='both', color = 'white')

	ax[1].grid(visible=True, which='both', axis='both')

	plt.legend()

	return axis_grid, np.squeeze(np.squeeze(beam_density1D_full / n_coef_1))



def plot_beam_toroidal(list_of_beams, equilibrium, norm = False, z_position = None, fig_size=(15,15), scale = 'log'):

	MA_r , MA_z  = equilibrium.magnetic_axis
	R_min, R_max = equilibrium.r_range
	Z_min, Z_max = equilibrium.z_range

	if not z_position:
		z_position = MA_z

	x_start, x_end, x_num = -R_max, R_max, 2000
	y_start, y_end, y_num = -R_max, R_max, 2000

	beam_density = np.zeros((x_num, y_num))
	xpts = np.linspace(x_start, x_end, x_num)
	ypts = np.linspace(y_start, y_end, y_num)

	# Adding port central lines
	angles = np.arange(0, 361, 22.5)
	rad_1 = np.ones(len(angles)) * 1.5
	rad_2 = np.ones(len(angles)) * 2.93

	x1 = rad_1 * np.cos(np.deg2rad(angles))
	y1 = rad_1 * np.sin(np.deg2rad(angles))
	x2 = rad_2 * np.cos(np.deg2rad(angles))
	y2 = rad_2 * np.sin(np.deg2rad(angles))
	

	for beam in list_of_beams:
		for i, xpt in enumerate(xpts):
			for j, ypt in enumerate(ypts):
				pt = Point3D(xpt, ypt, z_position).transform(beam.to_local())
				beam_density[i, j] = beam_density[i, j] + beam.density(pt.x, pt.y, pt.z)

	figure, axes = plt.subplots(figsize=fig_size)
	plt.rcParams.update({'font.size': 16})
	owter_wall = plt.Circle(( 0. , 0. ), 2.925, color='w', fill = False, linewidth=2) 
	inner_wall = plt.Circle(( 0. , 0. ), 1.5,   color='w', fill = False, linewidth=2) 
	magnetic_axis = plt.Circle(( 0. , 0. ), 2.15, color='r', fill = False, linewidth=2) 
	outer_plasma_border = plt.Circle(( 0. , 0. ), 2.686, color='c', fill = False, linewidth=3) 
	inner_plasma_border = plt.Circle(( 0. , 0. ), 1.588, color='c', fill = False, linewidth=3)
	if scale == 'log':
		plotted_beam_density = np.log10(beam_density)
		vmin_setting = 12
	else:
		plotted_beam_density = beam_density
		vmin_setting = 0

	plt.imshow(np.transpose(plotted_beam_density), extent=[x_start, x_end, y_start, y_end], origin='lower', cmap='inferno', vmin = vmin_setting)
	plt.yticks(fontsize=16)
	plt.xticks(fontsize=16)
	axes.add_artist(owter_wall)
	axes.add_artist(inner_wall)
	axes.add_artist(magnetic_axis)
	axes.add_artist(outer_plasma_border)
	axes.add_artist(inner_plasma_border)
	axes.set_facecolor("black")
	for i in range(len(x2)):
		plt.plot([x1[i],x2[i]], [y1[i],y2[i]], linestyle='--', color='w')
	plt.colorbar()
	plt.axis('equal')
	return figure
	



def plot_plasma_on_R(plasma,equilibrium, main_ion = hydrogen, main_ion_charge = 1,
					   impurity_ion = carbon, impurity_charge = 6,
					   impurity_only = False):

	ion=plasma.composition.get(main_ion,main_ion_charge)
	imp=plasma.composition.get(impurity_ion,impurity_charge)

	_, MA_z = equilibrium.magnetic_axis
	R_min, R_max =  equilibrium.r_range

	if not impurity_only:
		R, _, _, ne = sample3d(plasma.electron_distribution.density,(R_min,R_max,100),(0,0,1),(MA_z,MA_z,1))
		R, _, _, ni = sample3d(plasma.ion_density,(R_min,R_max,100),(0,0,1),(MA_z,MA_z,1))
		ne = np.squeeze(np.squeeze(ne))
		ni = np.squeeze(np.squeeze(ni))

		R, _, _, Te = sample3d(plasma.electron_distribution.effective_temperature,(R_min,R_max,100),(0,0,1),(MA_z,MA_z,1))
		R, _, _, Ti_ion = sample3d(ion.distribution.effective_temperature,(R_min,R_max,100),(0,0,1),(MA_z,MA_z,1))
		R, _, _, Ti_imp = sample3d(imp.distribution.effective_temperature,(R_min,R_max,100),(0,0,1),(MA_z,MA_z,1))
		Te = np.squeeze(np.squeeze(Te))
		Ti_ion = np.squeeze(np.squeeze(Ti_ion))
		Ti_imp = np.squeeze(np.squeeze(Ti_imp))
	   
		fig, ax = plt.subplots(1,2, figsize=(15,10))
		ax[0].plot(R,ne,label = 'ne')
		ax[0].plot(R,ni,label='sum(ni)')
		ax[0].set_xlabel('R, m')
		ax[0].set_ylabel('density, m-3')
		ax[0].legend()
		ax[0].grid(True)

		ax[1].plot(R,Te,label = 'Te')
		ax[1].plot(R,Ti_ion,label='Ti_'+main_ion.name+str(main_ion_charge)+'+')
		ax[1].plot(R,Ti_imp,label='Ti_'+impurity_ion.name+str(impurity_charge)+'+')
		ax[1].set_xlabel('R, m')
		ax[1].set_ylabel('temperature, eV')
		ax[1].legend()
		ax[1].grid(True)
	else:
		R, _, _, nz = sample3d(imp.distribution.density,(R_min,R_max,100),(0,0,1),(MA_z,MA_z,1))
		R, _, _, Ti_imp = sample3d(imp.distribution.effective_temperature,(R_min,R_max,100),(0,0,1),(MA_z,MA_z,1))
		nz = np.squeeze(np.squeeze(nz))
		Ti_imp = np.squeeze(np.squeeze(Ti_imp))
	   
		fig, ax = plt.subplots(1,2, figsize=(15,10))
		ax[0].plot(R,nz,label=impurity_ion.name+str(impurity_charge)+'+'+' density')
		ax[0].set_xlabel('R, m')
		ax[0].set_ylabel('density, m-3')
		ax[0].legend()
		ax[0].grid(True)

		ax[1].plot(R,Ti_imp,label='Ti_'+impurity_ion.name+str(impurity_charge)+'+')
		ax[1].set_xlabel('R, m')
		ax[1].set_ylabel('temperature, eV')
		ax[1].legend()
		ax[1].grid(True)

def plot_3D_chords(los,plasma = None):
	rho_min=1.5-0.67
	rho_max=1.5+0.67
	phi2d=np.arange(-90.0,90.0,1)/180.0*np.pi

	fig = plt.figure()
	ax = Axes3D(fig)
	
	ax.plot(rho_min*np.cos(phi2d),rho_min*np.sin(phi2d),0.0*phi2d,'-k',zorder=1)
	ax.plot(rho_max*np.cos(phi2d),rho_max*np.sin(phi2d),0.0*phi2d,'-k',zorder=1)
	
	for i,sys in enumerate(los.system_name):
		lens_pos=los.lens_position[i]
		xlens=lens_pos.x
		ylens=lens_pos.y
		zlens=lens_pos.z
		ax.plot(xlens,ylens,zlens,'rx')
		ax.text(xlens,ylens,zlens,sys)
		for k in range(los.number_of_channels[i]):
			target=los.viewing_target[i][k]
			x=target.x
			y=target.y
			z=target.y
			ax.plot(x,y,z,'ro')
			ax.plot([xlens,x],[ylens,y],[zlens, z],'c-')
	ax.set_xlabel('x, m')
	ax.set_ylabel('y, m')
	ax.set_zlabel('z, m')
	ax.set_xlim(0,2.2)
	ax.set_ylim(-2.2,2.2)
	ax.set_zlim(-2.2,2.2)
	ax.set_aspect('auto')
	ax.set_box_aspect((0.5,1,1))


def plot_spectra(los,spectra):
	for i,sys in enumerate(los.system_name):
		naxes = int( np.ceil(np.sqrt(los.number_of_channels[i])) )
		fig =plt.figure()
		wavelength = spectra[i]['emission']['wavelength']
		for k,target in enumerate(los.viewing_target[i]):
			ax=fig.add_subplot(naxes,naxes,k+1)
			for key in spectra[i]['emission'].keys():
				if key !='wavelength':
					intensity = spectra[i]['emission'][key][k]
					ax.plot(wavelength,intensity,label=key)
			title_str='target={'+str(round(target.x,2))+','+str(round(target.y,2))+','+str(round(target.z,2))+'}'
			ax.legend()
			ax.set_title(title_str)
			ax.set_xlabel('wavelength, nm')
			ax.set_ylabel('intensity, photons/s/m^2/nm/sr')
		fig.canvas.manager.set_window_title('system '+sys+': '+spectra[i]['type'])

