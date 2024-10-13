import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from scipy.interpolate import interp1d
from raysect.core.math import Point3D
from raysect.optical import translate, rotate_basis, Vector3D

from cherab.core import Beam
from cherab.core.model import SingleRayAttenuator
from cherab.core.math import sample3d
from cherab.openadas import OpenADAS


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


def neutralisation_efficiency(energy):
        # Efficiency of neutralisation in percent
        filename            = r'data/neutralization.dat'
        neutralization_data = np.loadtxt(filename)
        beam_energy         = neutralization_data[:,0]
        efficiensy          = neutralization_data[:,1] / 100
        efficiensy_int      = interp1d(beam_energy, efficiensy, kind='cubic', bounds_error=False, fill_value=0.25)

        if energy > 1.e4:
            return efficiensy_int(energy/1e3)
        else:
            print('Assuming the energy is in keV')
            return efficiensy_int(energy)

def create_beam(plasma, element, energy, start, radius, power, length=None):

    beam_start     = Point3D(start, 0, 0)
    dnb_target     = Point3D(0, 0, 0)
    dnb_forward    =  beam_start.vector_to(dnb_target).normalise()
    transform      = translate(beam_start.x, beam_start.y, beam_start.z) * rotate_basis(dnb_forward, Vector3D(0, 0, 1))

    if length:
        beam_length = length
    else:
        beam_length = start
    
    beam = Beam(parent=plasma.parent, transform=transform)

    beam.plasma                 = plasma
    beam.atomic_data            = OpenADAS(permit_extrapolation=True, missing_rates_return_null=True)
    beam.energy                 = energy
    beam.power                  = power
    beam.temperature            = 100
    beam.element                = element
    beam.sigma                  = radius
    beam.divergence_x           = 0.3
    beam.divergence_y           = 0.3
    beam.length                 = beam_length
    beam.attenuator             = SingleRayAttenuator(clamp_to_zero=False)
    beam.models                 = []
    beam.integrator.step        = 0.0025
    beam.integrator.min_samples = 5
    
    return beam

def create_beams(plasma, beam_gas, main_energy, beam_start, beam_radius, full_power, atom_composition, length = None):
    power = np.array(atom_composition) * full_power
    beam_e0 = create_beam(plasma, beam_gas, main_energy,   beam_start, beam_radius, power[0], length)
    beam_e2 = create_beam(plasma, beam_gas, main_energy/2, beam_start, beam_radius, power[1], length)
    beam_e3 = create_beam(plasma, beam_gas, main_energy/3, beam_start, beam_radius, power[2], length)

    print('Beams are created!')
    return beam_e0, beam_e2, beam_e3


def plot_beam_attenuation(beam_full,beam_half = None, beam_third = None, limits = None, norm = False, figsize=(15,12), axis = 'z', scale = 'log'):
    fig, ax = plt.subplots(2,1, figsize=figsize, constrained_layout=True)
    
    beam_length = beam_full.length

    x_start, x_end, y_start, y_end, z_start, z_end = 0, 0, 0, 0, 0, 0
    x_num, y_num, z_num  = 1, 1, 1

    radius = 3 * beam_full.sigma

    if axis == 'x':
        x_start         = -radius if limits is None else limits[0]
        x_end           =  radius if limits is None else limits[-1]
        x_num           = 200
        ax_min          = x_start
        ax_max          = x_end

    elif axis == 'y':
        y_start         = -radius if limits is None else limits[0]
        y_end           =  radius if limits is None else limits[-1]
        y_num           = 200
        ax_min          = y_start
        ax_max          = y_end
    
    else:
        z_start         = 0  if limits is None else limits[0]
        z_end           = beam_length if limits is None else limits[-1]
        z_num           = 200
        ax_min          = z_start
        ax_max          = z_end


    # Sampling main component  
    x1, y1, z1, beam_density1D_full  = sample3d(beam_full.density, (x_start, x_end, x_num), (y_start, y_end, y_num), (z_start, z_end, z_num))
    x2, _, z2, beam_density2D_full = sample3d(beam_full.density, (-radius, radius, 100), (0, 0, 1), (0, beam_length, 500))

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
        ax[1].semilogy(axis_grid,np.squeeze(np.squeeze(beam_density1D_full / n_coef_1)),label=f'{(beam_full.energy/1e3):.1f} кэВ')
    elif scale == 'linear':
        ax[1].plot(axis_grid,np.squeeze(np.squeeze(beam_density1D_full / n_coef_1)),label='E0')
    
    # Sampling E0/2
    if beam_half != None:
        x1, y1, z1, beam_density1D_half  = sample3d(beam_half.density, (x_start, x_end, x_num), (y_start, y_end, y_num), (z_start, z_end, z_num))
        x2, _, z2, beam_density2D_half = sample3d(beam_half.density, (-radius, radius, 100), (0, 0, 1), (0, beam_length, 500))
        if norm == True:
            n_coef_2 = np.max(beam_density1D_half)
        else:
            n_coef_2  = 1
    
        if scale == 'log':
            ax[1].semilogy(axis_grid,np.squeeze(np.squeeze(beam_density1D_half / n_coef_2)),label=f'{(beam_half.energy/1e3):.1f} кэВ')
        elif scale == 'linear':
            ax[1].plot(axis_grid,np.squeeze(np.squeeze(beam_density1D_half / n_coef_2)),label='E0/2')


    # Sampling E0/3	
    if beam_third != None:
        x1, y1, z1, beam_density1D_third = sample3d(beam_third.density,(x_start, x_end, x_num), (y_start, y_end, y_num), (z_start, z_end, z_num))
        x2, _, z2, beam_density2D_third = sample3d(beam_third.density, (-radius, radius, 100), (0, 0, 1), (0, beam_length, 500))
        if norm == True:
            n_coef_3 = np.max(beam_density1D_third)
        else:
            n_coef_3  = 1

        if scale == 'log':
            ax[1].semilogy(axis_grid,np.squeeze(np.squeeze(beam_density1D_third / n_coef_3)),label=f'{(beam_third.energy/1e3):.1f} кэВ')
        elif scale == 'linear':	
            ax[1].plot(axis_grid,np.squeeze(np.squeeze(beam_density1D_third / n_coef_3)),label='E0/3')

    ax[1].set_xlabel('Расстояние от фланца инжектора, м', fontsize=16)
    if norm == False:
        ax[1].set_ylabel(r'Плотность пучка, м$^{-3}$', fontsize=16)
    else:
        ax[1].set_ylabel('Доля начальной плотности пучка', fontsize=16)
        if scale == 'linear':
            ax[1].yaxis.set_major_locator(MultipleLocator(beam_length / 20))

    ax[1].set_xlim(ax_min,ax_max)
    ax[1].tick_params(axis='x', labelsize=14)
    ax[1].tick_params(axis='y', labelsize=14)

    if axis == 'z':
        ax[1].xaxis.set_major_locator(MultipleLocator(beam_length / 20))
    else:
        ax[1].xaxis.set_major_locator(MultipleLocator(radius / 20))

    beam_density2D = beam_density2D_full + beam_density2D_half + beam_density2D_third
    cs = ax[0].contourf(z2,x2,(np.squeeze(beam_density2D)),35,cmap="inferno")
    cbar = fig.colorbar(cs,ax=ax[0])
    cbar.set_label(r'Полная концентрация пучка, м$^{-3}$')

    ax[0].set_xlabel('z пучка, м', fontsize=16)
    ax[0].set_ylabel('x пучка, м', fontsize=16)
    ax[0].set_xlim(0,beam_length)
    ax[0].xaxis.set_major_locator(MultipleLocator(beam_length / 20))
    ax[0].grid(visible=True, which='both', axis='both', color = 'white')

    ax[0].tick_params(axis='x', labelsize=14)
    ax[0].tick_params(axis='y', labelsize=14)

    ax[1].grid(visible=True, which='both', axis='both')

    plt.legend()

    # return axis_grid, np.squeeze(np.squeeze(beam_density1D_full / n_coef_1))