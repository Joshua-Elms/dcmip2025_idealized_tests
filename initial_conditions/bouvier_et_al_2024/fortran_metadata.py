import numpy as np


def p_to_eta(p, ps: float = 1013.25):
    """
    Convert pressure levels to eta levels
    Formula: eta = (p/ps), where ps = surface pressure
    """
    return p/ps


def eta_to_p(eta, ps: float = 1013.25):
    """
    Convert eta levels to pressure levels
    Formula: p = eta*ps, where ps = surface pressure
    """
    return eta*ps


def write_levels(output_path, p_levels=None, eta_levels=None):
    """
    Use p or eta to find the other and write both to a file
    """
    if p_levels is None and eta_levels is None:
        raise ValueError("Either p_levels or eta_levels must be provided")

    elif p_levels is None:
        p_levels = eta_to_p(eta_levels)

    elif eta_levels is None:
        eta_levels = p_to_eta(p_levels)

    merged = np.vstack((p_levels, eta_levels)).T
    np.savetxt(output_path, merged, delimiter=",",
               fmt="%.15f", header="p,eta", comments="")
    print(f"Levels written to {output_path}")
    
    
def gen_and_write_lats(save_path, lat_north = 90, lat_south = -90, step_in_deg = 0.25):
    start, stop, step = (lat_north, lat_south, -step_in_deg)
    lat_deg = np.arange(start, stop+step, step)
    lat_rad = np.deg2rad(lat_deg)
    merged = np.vstack((lat_deg, lat_rad)).T
    np.savetxt(save_path, merged, delimiter=",", fmt="%.10f", header="lat_deg,lat_rad", comments="")
    print(f"Wrote latitudes to {output_path}")


def _fmtd_arr_to_fortran_str(array: list, name: str, precision):
    """
    Convert a (pseudo) 2D array to a string that can be used in Fortran code.
    """
    out = ""
    out += f"{name} = ["
    nchar = len(out)
    lines = [", ".join([f"{float(val):.{precision}f}" for val in sub if val != "."]) for sub in array]
    out += f", &\n&{' ' * (nchar - 1)}".join(lines)
    out += "]"

    return out

def numeric_arr_to_fort_str(arr: list, name: str, precision: int = 15, row_width: int = 3):
    arr += ["." for _ in range(row_width - (((len(arr)) % row_width)))]
    twoD_arr = np.array(arr).reshape(-1, row_width).tolist()
    return _fmtd_arr_to_fortran_str(twoD_arr, name, precision)


if __name__ == "__main__":
    ### step 1: output p/eta levels for gen_IC_FCN.F90
    
    # primary levels for FCN are: [1013.25, 1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
    # superset of those and additional levels to assist with integration of specific humidity:
    """
    [
        1013.25, 1000, 990, 980, 970, 960, 950, 940, 930, 925, 920, 910, 900, 890, 880, 870, 860, 850, 840, 830, 
        820, 810, 800, 790, 780, 770, 760, 750, 740, 730, 720, 710, 700, 690, 680, 670, 660, 650, 640, 630, 620, 
        610, 600, 590, 580, 570, 560, 550, 540, 530, 520, 510, 500, 490, 480, 470, 460, 450, 440, 430, 420, 410, 
        400, 390, 380, 370, 360, 350, 340, 330, 320, 310, 300, 290, 280, 270, 260, 250, 240, 230, 220, 210, 200, 
        190, 180, 170, 160, 150, 140, 130, 120, 110, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10
    ]
    """
    p_levels = np.array([1013.25, 1000, 925, 850, 700, 600,
                        500, 400, 300, 250, 200, 150, 100, 50])
    p_levels_full = np.array([
        1013.25, 1000, 990, 980, 970, 960, 950, 940, 930, 925, 920, 910, 900, 890, 880, 870, 860, 850, 840, 830,
        820, 810, 800, 790, 780, 770, 760, 750, 740, 730, 720, 710, 700, 690, 680, 670, 660, 650, 640, 630, 620,
        610, 600, 590, 580, 570, 560, 550, 540, 530, 520, 510, 500, 490, 480, 470, 460, 450, 440, 430, 420, 410,
        400, 390, 380, 370, 360, 350, 340, 330, 320, 310, 300, 290, 280, 270, 260, 250, 240, 230, 220, 210, 200,
        190, 180, 170, 160, 150, 140, 130, 120, 110, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10
    ])
    output_path = "levels_full.txt"
    write_levels(output_path, p_levels=p_levels_full)
    
    ### step 2: output latitudes for gen_IC_FCN.F90
    gen_and_write_lats("/glade/work/jmelms/data/dcmip2025_idealized_tests/initial_conditions/metadata/lat")
    
    ### step 3: format outputs from steps 1 and 2 as Fortran lists to paste into gen_IC_FCN.F90
   
