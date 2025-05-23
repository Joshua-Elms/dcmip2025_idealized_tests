import datetime
import dotenv
import datetime as dt
import torch
from utils import inference_graphcast_small
from earth2mip.networks import get_model # :ignore

dotenv.load_dotenv()

model = get_model("e2mip://graphcast_small", device="cuda:0")
output_path = "graphcast_small_0.nc"
date = dt.datetime(2018, 1, 1, 0)
print(f"output_path: {output_path}")
ds = inference_graphcast_small.single_IC_inference(
    model=model, 
    n_timesteps=1, 
    init_time=date,
    device="cuda:0",
    vocal=True,
)
ds.to_netcdf(output_path)
print(f"output: {ds}")