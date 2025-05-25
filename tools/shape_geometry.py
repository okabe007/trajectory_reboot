
import numpy as np

def volume_um3_from_ul(volume_ul):
    return volume_ul * 1e9

def calc_cube_size(volume_um3):
    edge = volume_um3 ** (1/3)
    return {"edge_length": edge}


def spot_volume_eq(R, angle_rad, volume):
    h = R * (1 - np.cos(angle_rad))
    return (np.pi * h**2 * (3*R - h)) / 3 - volume

def calc_spot_size(volume_um3, angle_deg):
    angle_rad = np.deg2rad(angle_deg)

    def cap_volume(R):
        h = R * (1 - np.cos(angle_rad))
        return np.pi * h * h * (3 * R - h) / 3

    low = 0.0
    high = max(volume_um3 ** (1 / 3), 1.0)
    while cap_volume(high) < volume_um3:
        high *= 2.0

    for _ in range(60):
        mid = (low + high) / 2.0
        if cap_volume(mid) < volume_um3:
            low = mid
        else:
            high = mid

    R_solution = (low + high) / 2.0
    h = R_solution * (1 - np.cos(angle_rad))
    bottom_R = R_solution * np.sin(angle_rad)
    return {
        "spot_R": R_solution,
        "spot_height": h,
        "spot_bottom_R": bottom_R,
    }
