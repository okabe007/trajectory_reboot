
import numpy as np
from scipy.optimize import fsolve

def volume_um3_from_ul(volume_ul):
    return volume_ul * 1e9

def calc_cube_size(volume_um3):
    edge = volume_um3 ** (1/3)
    return {"edge_length": edge}

def calc_drop_size(volume_um3):
    radius = ((3 * volume_um3) / (4 * np.pi)) ** (1/3)
    return {"drop_radius": radius}

def spot_volume_eq(R, angle_rad, volume):
    h = R * (1 - np.cos(angle_rad))
    return (np.pi * h**2 * (3*R - h)) / 3 - volume

def calc_spot_size(volume_um3, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    r_guess = ((3 * volume_um3) / (np.pi * angle_rad**2))**(1/3)
    R_solution = fsolve(spot_volume_eq, r_guess, args=(angle_rad, volume_um3))[0]
    h = R_solution * (1 - np.cos(angle_rad))
    bottom_R = R_solution * np.sin(angle_rad)
    return {
        "spot_R": R_solution,
        "spot_height": h,
        "spot_bottom_R": bottom_R
    }
