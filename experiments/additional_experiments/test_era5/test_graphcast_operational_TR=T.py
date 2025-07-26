import datetime
import dotenv
import xarray as xr
import datetime as dt
from pathlib import Path
from utils import inference_graphcast_oper
from earth2mip.networks import get_model # :ignore

dotenv.load_dotenv()

model = get_model("e2mip://graphcast_operational", device="cuda:0")
output_path = "graphcast_operational_TR=T.nc"
date = dt.datetime(2018, 1, 1, 0)
print(f"output_path: {output_path}")
print(f"Running tendency reversion test for {date}")
tendency_path = Path("graphcast_operational_tendency.nc")
if tendency_path.exists():
    print(f"Loading cached tendencies from {tendency_path}.")
    tendency = xr.open_dataset(tendency_path)
    
else:
    print(f"Computing tendencies and saving to {tendency_path}.")
    tds = inference_graphcast_oper.single_IC_inference(
        model=model, 
        n_timesteps=1, 
        init_time=date,
        device="cuda:0",
        vocal=True,
    )
    tendency = tds.isel(time=1) - tds.isel(time=0)
    tendency.to_netcdf(tendency_path)
    
print(f"tendency: {tendency}")
rpert = - tendency

# run model with recurrent perturbation
ds = inference_graphcast_oper.single_IC_inference(
    model=model,
    n_timesteps=6,
    init_time=date,
    device="cuda:0",
    recurrent_perturbation=rpert,
    vocal=True,
)

# save output
ds.to_netcdf(output_path)
print(f"Saved output to {output_path}")
print(f"output: {ds}")
