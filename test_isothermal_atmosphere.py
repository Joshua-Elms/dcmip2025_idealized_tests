import datetime as dt
import torch
import numpy as np
import logging
from earth2mip import networks
import dcmip2025_helper_funcs as dcmip
import dotenv

# load the earth2mip environment variables
dotenv.load_dotenv()

logging.basicConfig(level=logging.INFO)

def generate_isothermal_atmosphere_at_rest():
    """" Generates an SFNO-compatible xarray for an isothermal atmosphere at rest. """
 
    # initialize an sfno xarray
    isothermal_ds = dcmip.create_empty_sfno_ds()

    # winds and humidity variables to zero
    isothermal_ds["U"][:] = 0.0
    isothermal_ds["V"][:] = 0.0
    isothermal_ds["R"][:] = 0.0
    isothermal_ds["VAR_10U"][:] = 0.0
    isothermal_ds["VAR_10V"][:] = 0.0
    isothermal_ds["VAR_100U"][:] = 0.0
    isothermal_ds["VAR_100V"][:] = 0.0
    isothermal_ds["TCW"][:] = 0.0

    temp = 300.0
    isothermal_ds["T"][:] = temp
    isothermal_ds["VAR_2T"][:] = 0.0

    # surface pressure
    p0 = 100000.0
    isothermal_ds["SP"][:] = p0
    isothermal_ds["MSL"][:] = p0

    # infer the geopotential height
    p_lev = np.array(dcmip.sfno_levels)*100.0
    g = 9.81 # m/s^2
    R_d = 287.0 # J/kg/K
    T = temp
    H = R_d*T/g
    z = H*np.log(p0/p_lev)
    isothermal_ds["Z"][:] = z[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]

    return isothermal_ds

# load the model
device = "cuda:0"
#device = "cpu"
model_name = "fcnv2_sm"
print("Loading model.")
model = networks.get_model(model_name).to(device)
print("Model loaded.")

print("Initializing model.")
# generate the initial condidtion
ds = generate_isothermal_atmosphere_at_rest()
# put the initial condition into a format compatible with the SFNO
x = dcmip.pack_sfno_state(ds, device=device)

# run the model
data_list = []
print("Model initialized.")
iterator = model(dt.datetime(1850,1,1), x)
for k, (time, data, _) in enumerate(iterator):
    print(f"Step {k+1} completed.")
    data_list.append(data)
    if k == 1:
        break

# stack the data
data = torch.stack(data_list)

# unpack the data
ds = dcmip.unpack_sfno_state(data)

# save the data
ds.squeeze().to_netcdf("isothermal_at_rest.nc")
