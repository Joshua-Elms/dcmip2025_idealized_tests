import xarray as xr
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

def generate_isothermal_atmosphere_at_rest(
        account_for_topography = True,
):
    """" Generates an SFNO-compatible xarray for an isothermal atmosphere at rest. """

    # set some constants
    g = 9.81 # m/s^2
    R_d = 287.0 # J/kg/K

    # set parameters
    T = 300.0 # atmospheric temperature [K]
    p0 = 100000.0 # surface pressure at sea level [Pa]
    H = R_d*T/g # scale height [m]
 
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

    isothermal_ds["T"][:] = T
    isothermal_ds["VAR_2T"][:] = 0.0

    if account_for_topography:
        # get the topography
        topo_file = "/glade/campaign/collections/rda/data/ds633.0/e5.oper.invariant/197901/e5.oper.invariant.128_129_z.ll025sc.1979010100_1979010100.nc"
        topo_xr = xr.open_dataset(topo_file, chunks = -1).squeeze()
        zs = topo_xr["Z"].values/g

    else:
        zs = np.zeros_like(isothermal_ds["SP"].values)

    # set surface pressure based on topography
    ps = p0*np.exp(-zs/H)
    isothermal_ds["SP"][:] = ps
    # set mean sea level pressure
    isothermal_ds["MSL"][:] = p0

    # infer the geopotential height
    p_lev = np.array(dcmip.sfno_levels)*100.0
    z = H*np.log(ps[np.newaxis,:,:]/p_lev[:,np.newaxis,np.newaxis])
    isothermal_ds["Z"][:] = z[np.newaxis, np.newaxis, ...]

    return isothermal_ds

# load the model
device = "cuda:0"
#device = "cpu"
print("Loading model.")
model = networks.get_model("fcnv2_sm").to(device)
print("Model loaded.")

print("Initializing model.")
# generate the initial condidtion
ds = generate_isothermal_atmosphere_at_rest()
# put the initial condition into a format compatible with the SFNO
x = dcmip.pack_sfno_state(ds, device=device)
print("Model initialized.")

# run the model
data_list = []
iterator = model(dt.datetime(1850,1,1), x)
for k, (time, data, _) in enumerate(iterator):
    print(f"Step {k+1} completed.")
    data_list.append(data)
    if k == 5:
        break

# stack the output data by time
data = torch.stack(data_list)

# unpack the data into an xarray object
ds = dcmip.unpack_sfno_state(data)

# save the data
ds.squeeze().to_netcdf("isothermal_at_rest.nc")
