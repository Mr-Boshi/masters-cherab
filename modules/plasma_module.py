import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import electron_mass, atomic_mass

from raysect.primitive import Cylinder
from raysect.optical import translate, Point3D, Vector3D, rotate_basis
from raysect.optical.observer import PinholeCamera, PowerPipeline2D

from cherab.core import Species, Maxwellian, Plasma
from cherab.core.math import sample3d
from cherab.openadas import OpenADAS



class NeutralFunction:
    """A neutral profile that is constant outside the plasma,
       then exponentially decays inside the LCFS."""

    def __init__(self, peak_value, sigma, magnetic_axis, lcfs_radius=1):

        self.peak = peak_value
        self.sigma = sigma
        self.lcfs_radius = lcfs_radius
        self._constant = (2*self.sigma*self.sigma)

        self.r_axis = magnetic_axis[0]
        self.z_axis = magnetic_axis[1]

    def __call__(self, x, y, z):

        # calculate r in r-z space
        r = np.sqrt(x**2 + y**2)

        # calculate radius of coordinate from magnetic axis
        radius_from_axis = np.sqrt((r - self.r_axis)**2 + (z - self.z_axis)**2)

        if radius_from_axis <= self.lcfs_radius:
            return self.peak * np.exp(-((radius_from_axis - self.lcfs_radius)**2) / self._constant)
        else:
            return self.peak

class IonFunction:
    """An approximate toroidal plasma profile that follows a double
       quadratic between the LCFS and magnetic axis."""

    def __init__(self, v_core, v_lcfs, magnetic_axis, p=4, q=3, lcfs_radius=1):

        self.v_core = v_core
        self.v_lcfs = v_lcfs
        self.p = p
        self.q = q
        self.lcfs_radius = lcfs_radius

        self.r_axis = magnetic_axis[0]
        self.z_axis = magnetic_axis[1]

    def __call__(self, x, y, z):

        # calculate r in r-z space
        r = np.sqrt(x**2 + y**2)

        # calculate radius of coordinate from magnetic axis
        radius_from_axis = np.sqrt((r - self.r_axis)**2 + (z - self.z_axis)**2)

        # evaluate pedestal-> core function
        if radius_from_axis <= self.lcfs_radius:
            return ((self.v_core - self.v_lcfs) *
                    np.power((1 - np.power(radius_from_axis / self.lcfs_radius, self.p)), self.q) + self.v_lcfs)
        else:
            return 0


def create_plasma(parent, main_gas, peak_density, peak_temperature, major_radius, minor_radius):
    
    magnetic_axis = (major_radius, 0)

    plasma = Plasma(parent=parent)
    plasma.atomic_data = OpenADAS(permit_extrapolation=True)
    plasma.geometry = Cylinder(10, 10, transform=translate(0, 0, -1.1))
    plasma.geometry_transform = translate(0, 0, -1.1)
    
    # No net velocity for any species
    zero_velocity = Vector3D(0, 0, 0)
    
    # define neutral species distribution
    d0_density = NeutralFunction(peak_density, 0.1, magnetic_axis, lcfs_radius = minor_radius)
    d0_temperature = 0.5  # constant 0.5eV temperature for all neutrals
    d0_distribution = Maxwellian(d0_density, d0_temperature, zero_velocity,
                                main_gas.atomic_weight * atomic_mass)
    d0_species = Species(main_gas, 0, d0_distribution)
    
    # define main_gas ion species distribution
    d1_density = IonFunction(peak_density, 0, magnetic_axis, lcfs_radius = minor_radius)
    d1_temperature = IonFunction(peak_temperature, 0, magnetic_axis, lcfs_radius = minor_radius)
    d1_distribution = Maxwellian(d1_density, d1_temperature, zero_velocity,
                                main_gas.atomic_weight * atomic_mass)
    d1_species = Species(main_gas, 1, d1_distribution)
    
    # define the electron distribution
    e_density = IonFunction(peak_density, 0, magnetic_axis, lcfs_radius = minor_radius)
    e_temperature = IonFunction(peak_temperature, 0, magnetic_axis, lcfs_radius = minor_radius)
    e_distribution = Maxwellian(e_density, e_temperature, zero_velocity, electron_mass)
    
    # define species
    plasma.b_field = Vector3D(0.1, 0.1, 0.1)
    plasma.electron_distribution = e_distribution
    plasma.composition = [d0_species, d1_species]

    print('Plasma created!')
    return plasma


def plasma_temperature_rz(plasma, main_gas, r_range, z_range):
    plasma_ions = plasma.composition.get(main_gas, 1)
    d1_temperature_distribution = plasma_ions.distribution.effective_temperature
    
    r, _, z, t_samples = sample3d(d1_temperature_distribution, (r_range[0], r_range[1], 200), (0, 0, 1), (z_range[0], z_range[1], 200))
    
    plt.imshow(np.transpose(np.squeeze(t_samples)), extent=[r_range[0], r_range[1], z_range[0], z_range[1]])
    plt.colorbar()
    # plt.axis('equal')
    plt.xlabel('r, м', fontsize=14)
    plt.ylabel('z, м', fontsize=14)
    plt.title(r"$T_i$(r,z), эВ")

def plasma_temperature_xy(plasma, main_gas, x_range):
    plasma_ions = plasma.composition.get(main_gas, 1)
    d1_temperature_distribution = plasma_ions.distribution.effective_temperature

    r, _, z, t_samples = sample3d(d1_temperature_distribution, (x_range[0], x_range[1], 400), (x_range[0], x_range[1], 400), (0, 0, 1))
    
    plt.figure()
    plt.imshow(np.transpose(np.squeeze(t_samples)), extent=[x_range[0], x_range[1], x_range[0], x_range[1]])
    plt.colorbar()
    # plt.axis('equal')
    plt.xlabel('x, м', fontsize=14)
    plt.ylabel('y, м', fontsize=14)
    plt.title(r"$T_i$(x,y), эВ")

def plasma_density_rz(plasma, main_gas, r_range, z_range):
    plasma_ions = plasma.composition.get(main_gas, 1)
    d1_density_distribution = plasma_ions.distribution.density

    r, _, z, t_samples = sample3d(d1_density_distribution, (r_range[0], r_range[1], 200), (0, 0, 1), (z_range[0], z_range[1], 200))

    plt.figure()
    plt.imshow(np.transpose(np.squeeze(t_samples)), extent=[r_range[0], r_range[1], z_range[0], z_range[1]])
    plt.colorbar()
    plt.axis('equal')
    plt.xlabel('r axis', fontsize=14)
    plt.ylabel('z axis', fontsize=14)
    plt.title(r"$n_i$(r,z), м$^{-3}$")

def plasma_emission(plasma, start, target):
    camera_pupil   = Point3D(start[0], start[1], start[2])
    camera_target  = Point3D(target[0], target[1], target[2])

    forward = camera_pupil.vector_to(camera_target).normalise()

    pipeline = PowerPipeline2D(display_update_time = 60)
    camera = PinholeCamera((256, 256), pipelines=[pipeline], parent=plasma.parent)
    camera.transform = (translate(camera_pupil.x, camera_pupil.y, camera_pupil.z) * rotate_basis(forward, Vector3D(0, 0, 1)))
    camera.pixel_samples = 1

    plt.ion()
    camera.observe()
    plt.ioff()
    plt.show()