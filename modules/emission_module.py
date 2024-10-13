from raysect.core.math import Point3D
from raysect.optical import Ray

from cherab.core.model import ExcitationLine, BeamEmissionLine
from cherab.core.atomic import Line

# import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.ticker import MultipleLocator


def beam_line(element):
    return BeamEmissionLine(Line(element, 0, (3, 2)))

def excitation_line(element):
    return ExcitationLine(Line(element, 0, (3, 2)))

def trace_beam_emission(beam, emission_model, start, where_to, wavelengths):

    pupil   = Point3D(start[0], 0, start[1])
    target  = Point3D(where_to[0], 0, where_to[1])
    forward = pupil.vector_to(target).normalise()

    beam.models = [emission_model]
    ray = Ray(origin=pupil, direction=forward,
           min_wavelength=wavelengths[0], max_wavelength=wavelengths[1], bins=10000)
    spectrum = ray.trace(beam.parent)
    beam.models = []
    return spectrum.wavelengths, spectrum.samples

def plot_beam_spectra(lambdas, e0_emission, e2_emission=[None], e3_emission=[None]):
    fig = plt.figure(figsize=[10,7.5])
    plt.rcParams.update({'font.size': 14})
    
    if e3_emission[0] is not None or e2_emission[0]  is not None:
        plt.plot(lambdas, e0_emission + e2_emission + e3_emission)

    plt.plot(lambdas, e0_emission)
    
    if e2_emission[0] is not None:
        plt.plot(lambdas, e2_emission)

    if e3_emission[0] is not None:
        plt.plot(lambdas, e3_emission)



    plt.grid(True)
    plt.xlabel(r'$\lambda$, нм')
    plt.ylabel(r'Спектральная плотность мощности, Вт/(м$^2\cdot$ср$\cdot$нм)')
    

    