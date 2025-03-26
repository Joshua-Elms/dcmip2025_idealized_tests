import datetime
import dotenv
from earth2mip.networks import get_model
from earth2mip.initial_conditions import cds, get_data_source
from earth2mip.inference_ensemble import run_basic_inference
from earth2mip import schema
dotenv.load_dotenv()

time_loop  = get_model("e2mip://graphcast_small", device="cuda:0")
print('channel_names',time_loop.in_channel_names)

output_path = "graphcast_small_output.nc"
print(f"output_path: {output_path}")

data_source = cds.DataSource(time_loop.in_channel_names)
print('data_source type',type(data_source))
    

ds = run_basic_inference(time_loop, n=2, data_source=data_source, time=datetime.datetime(2018, 1, 2))
ds.to_netcdf(output_path)
print(f"output: {ds}")
